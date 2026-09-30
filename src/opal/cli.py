"""``opal`` command-line interface.

    opal info                              GPU / TensorFlow / RAM check
    opal config  -c cfg.json [--set k=v]   print the resolved configuration
    opal plan    -c cfg.json               memory-guard dry run (which mode would run)
    opal run     -c cfg.json --out DIR     simulate and save frames
    opal pass    -c cfg.json --out DIR     satellite pass with exact r0 switching
    opal dataset -c cfg.json --out DIR     labelled ML dataset (sweeps x realizations)
    opal analyze r0|fiber DIR              r0 / fiber-coupling statistics of a run
    opal export  -c cfg.json --out DIR     TensorFlow SavedModel
    opal view    DIR [--save movie.mp4]    animate a run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ._version import __version__

MODES = ("auto", "device", "host", "chunked", "no_save")


def _parse_value(text):
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    try:
        f = float(text)
    except ValueError:
        return text
    return int(f) if f.is_integer() and "." not in text and "e" not in low else f


def _overrides(args):
    out = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _parse_value(v)
    for flag, key in (("steps", "steps"), ("grid", "N"), ("layers", "layers"), ("r0", "r0"),
                      ("seed", "seed")):
        v = getattr(args, flag, None)
        if v is not None:
            out[key] = v
    return out


def _load(args):
    from .config import load_config

    return load_config(args.config if args.config else {}, overrides=_overrides(args))


def _add_config_args(p, steps=True):
    p.add_argument("-c", "--config", help="JSON config file (default: built-in defaults)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a parameter, e.g. --set r0=0.1 --set wind=10 (repeatable)")
    p.add_argument("--grid", type=int, help="grid size N")
    p.add_argument("--layers", type=int, help="number of layers")
    p.add_argument("--r0", type=float, help="total path r0 (m)")
    p.add_argument("--seed", type=int, help="master seed")
    if steps:
        p.add_argument("--steps", type=int, help="number of ticks")


def _quiet_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _printer(args):
    return None if getattr(args, "quiet", False) else (lambda line: print(line, flush=True))


# ----------------------------------------------------------------------
def cmd_info(args):
    _quiet_tf()
    import numpy as np
    import tensorflow as tf

    from . import memory as mem

    gpus = tf.config.list_physical_devices("GPU")
    print(f"OPAL {__version__}")
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | tensorflow {tf.__version__}")
    print(f"built with CUDA: {tf.test.is_built_with_cuda()}")
    if gpus:
        total = mem.gpu_total_memory_bytes()
        print(f"GPU: {mem.gpu_name()} ({total / mem.GIB:.1f} GiB)" if total else f"GPU: {mem.gpu_name()}")
    else:
        print("GPU: none visible (OPAL will run on CPU, slowly)")
    ram = mem.host_available_bytes()
    if ram:
        print(f"host RAM available: {ram / mem.GIB:.1f} GiB")
    return 0


def cmd_config(args):
    from .config import save_config, to_plain

    cfg = _load(args)
    if args.output:
        save_config(cfg, args.output)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(to_plain(cfg), indent=2))
    return 0


def cmd_plan(args):
    _quiet_tf()
    from . import memory as mem
    from .simulator import GpuSimulator

    cfg = _load(args)
    sim = GpuSimulator(cfg)
    T = int(args.steps or sim.steps)
    baseline = mem.measure_baseline_peak_bytes(sim)
    est = mem.estimate_memory(T, sim.nx_size, baseline, requested_chunk_size=args.chunk_size)
    selected, chunk, reason = mem.select_mode(args.mode, est, live=True, has_output_dir=not args.no_out)
    for line in mem.format_plan(est, args.mode, selected, chunk):
        print(line)
    print(f"  decision: {reason}")
    d = sim.describe()
    print(f"channel: N={d['grid']} dx={d['pixel_size_m']} m aperture={d['aperture_m']:.2f} m "
          f"r0={d['total_r0_m']} m D/r0={d['D_over_r0']:.1f} layers={d['num_layers']} seed={d['seed']}")
    for layer in d["layers"]:
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in layer.items()))
    return 0


def cmd_run(args):
    _quiet_tf()
    from . import runners
    from .savedmodel import load
    from .simulator import GpuSimulator

    log = _printer(args)
    if args.model:
        sim = load(args.model)
    else:
        cfg = _load(args)
        sim = GpuSimulator(cfg)
        if log:
            d = sim.describe()
            log(f"OPAL {__version__} | N={d['grid']} layers={d['num_layers']} r0={d['total_r0_m']} m "
                f"D/r0={d['D_over_r0']:.1f} seed={d['seed']}")
    state_in, state_out = args.state_in, args.state_out
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.exists() and state_in is None:
            state_in = ckpt
        state_out = state_out or ckpt
    captures = [int(x) for x in args.capture.split(",")] if args.capture else None
    result = runners.run(
        sim, args.steps, mode=args.mode, schedule=args.schedule, schedule_kind=args.schedule_kind,
        out_dir=args.out, chunk_size=args.chunk_size, capture_indices=captures,
        state_in=state_in, state_out=state_out, log=log,
    )
    if args.quiet:
        print(json.dumps({"mode": result.mode, "steps": result.num_steps,
                          "run_time_s": round(result.run_time_s, 3), "out_dir": str(result.out_dir)}))
    return 0


def cmd_pass(args):
    _quiet_tf()
    from .passes import run_pass

    cfg = _load(args)
    if args.segments:
        cfg.setdefault("pass_params", {})["num_segments"] = args.segments
    if args.captures:
        cfg.setdefault("pass_params", {})["wavefront_captures"] = args.captures
    if args.window is not None:
        cfg.setdefault("pass_params", {})["pass_window_s"] = args.window
    run_pass(cfg, args.out, measure_r0=not args.no_r0, log=_printer(args))
    return 0


def cmd_dataset(args):
    _quiet_tf()
    from .dataset import generate_dataset

    cfg = _load(args)
    generate_dataset(
        cfg, args.out, sweep=args.sweep, realizations=args.realizations, num_steps=args.steps,
        stride=args.stride, crop=args.crop, targets=[t.strip() for t in args.targets.split(",")],
        seed=args.seed, label_aperture=args.label_aperture, r0_frames=args.r0_frames,
        compress=args.compress, resume=not args.no_resume, log=_printer(args),
    )
    return 0


class _ChunkedFrames:
    """Read-only ``[T, N, N]`` view over the chunk files of a chunked run."""

    def __init__(self, run_dir, chunks):
        import numpy as np

        self._np = np
        self.parts = [(c["start_step"], np.load(Path(run_dir) / Path(c["file"]).name, mmap_mode="r"))
                      for c in chunks]
        first = self.parts[0][1]
        self.shape = (sum(p.shape[0] for _, p in self.parts), *first.shape[1:])

    def _one(self, t):
        for start, arr in self.parts:
            if start <= t < start + arr.shape[0]:
                return arr[t - start]
        raise IndexError(t)

    def __getitem__(self, idx):
        if isinstance(idx, (int, self._np.integer)):
            return self._one(int(idx))
        return self._np.stack([self._one(int(t)) for t in idx])


def _run_frames(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    import numpy as np

    for name in ("frames.npy", "wavefronts.npy"):
        if (run_dir / name).exists():
            return manifest, np.load(run_dir / name, mmap_mode="r")
    chunk_manifest = run_dir / "frames_chunk_manifest.json"
    if chunk_manifest.exists():
        chunks = json.loads(chunk_manifest.read_text(encoding="utf-8"))["chunks"]
        if chunks and chunks[0].get("file"):
            return manifest, _ChunkedFrames(run_dir, chunks)
    raise SystemExit(f"{run_dir} holds no saved frames (no_save run?)")


def cmd_analyze(args):
    import numpy as np

    from .analysis.fiber import (compute_fiber_coupling, coupling_efficiency_theoretical_vk)
    from .analysis.r0 import estimate_r0_from_wavefront

    manifest, frames = _run_frames(args.run_dir)
    ch = manifest["config"]["channel_params"]
    dx, L0, r0 = float(ch["pixel_size"]), float(ch["L0"]), float(ch["total_r0"])
    N = frames.shape[-1]
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, min(args.frames, frames.shape[0])).round().astype(int))
    if args.what == "r0":
        aperture = args.aperture
        vals = np.array([estimate_r0_from_wavefront(np.asarray(frames[i]), dx, aperture, L0=L0)["measured_r0"]
                         for i in idx])
        ok = vals[np.isfinite(vals)]
        print(f"r0 over {len(ok)} frames: median {np.median(ok):.4f} m, mean {ok.mean():.4f} m, "
              f"std {ok.std():.4f} m (configured path r0 {r0} m, ratio {np.median(ok) / r0:.3f})")
    else:
        aperture = args.aperture or 0.4 * N * dx
        if aperture >= 0.47 * N * dx:
            raise SystemExit(f"aperture must be < {0.47 * N * dx:.2f} m (inside the absorbing boundary)")
        eta = compute_fiber_coupling(np.asarray(frames[idx]), dx, a=args.a, aperture_diameter=aperture)["eta"]
        theory = coupling_efficiency_theoretical_vk(aperture, r0, L0, a=args.a)
        print(f"fiber coupling over {len(idx)} frames (D={aperture:.3f} m, D/r0={aperture / r0:.2f}, a={args.a}): "
              f"mean {eta.mean():.4f}, std {eta.std():.4f}; von Karman theory {theory:.4f} "
              f"({100 * (eta.mean() / theory - 1):+.1f} %)")
    print("note: frames of one run are correlated in time; for ensemble statistics use independent "
          "realizations (opal dataset -k K or GpuSimulator.reseed)")
    return 0


def cmd_export(args):
    _quiet_tf()
    from .savedmodel import export
    from .simulator import GpuSimulator

    sim = GpuSimulator(_load(args))
    export(sim, args.out)
    print(f"exported SavedModel (seed {sim.seed}) to {args.out}")
    return 0


def cmd_view(args):
    from .viewer import animate_run

    out = animate_run(args.run_dir, save=args.save, stride=args.stride, fps=args.fps,
                      max_frames=args.max_frames)
    if out:
        print(f"wrote {out}")
    return 0


# ----------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="opal",
        description="OPAL: GPU-accelerated multilayer atmospheric turbulence simulator "
                    "for space-to-ground optical links.",
    )
    ap.add_argument("--version", action="version", version=f"opal {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show GPU / TensorFlow / RAM status")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("config", help="print or save the resolved configuration")
    _add_config_args(p)
    p.add_argument("-o", "--output", help="write the resolved config to this file")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("plan", help="memory-guard dry run: which mode would run and why")
    _add_config_args(p)
    p.add_argument("--mode", choices=MODES, default="auto")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--no-out", action="store_true", help="plan as if no output directory were given")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="simulate and (optionally) save frames")
    _add_config_args(p)
    p.add_argument("--mode", choices=MODES, default="auto",
                   help="auto (default) lets the memory guard pick device/host/chunked/no_save")
    p.add_argument("-o", "--out", help="run directory for frames + manifest")
    p.add_argument("--chunk-size", type=int, default=100, help="ticks per chunk in chunked mode")
    p.add_argument("--schedule", help=".npy/.npz/.csv/.json turbulence schedule [T or 1, layers] or [T]")
    p.add_argument("--schedule-kind", choices=("b_scale", "cn2", "r0"), default="b_scale")
    p.add_argument("--capture", help="comma-separated ticks at which to capture all layer screens")
    p.add_argument("--state-in", help="start from this checkpoint (.npz)")
    p.add_argument("--state-out", help="save the final state here (.npz)")
    p.add_argument("--checkpoint", help="resume from this file if it exists, then overwrite it")
    p.add_argument("--model", help="run an exported SavedModel instead of building from a config")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pass", help="satellite pass with exact per-segment r0 switching")
    _add_config_args(p)
    p.add_argument("-o", "--out", help="output directory")
    p.add_argument("--segments", type=int, help="number of r0 segments (divides steps)")
    p.add_argument("--captures", type=int, help="number of wavefront captures")
    p.add_argument("--window", type=float, help="only simulate +-window/2 s around zenith")
    p.add_argument("--no-r0", action="store_true", help="skip per-capture r0 measurement")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_pass)

    p = sub.add_parser("dataset", help="generate a labelled ML dataset")
    _add_config_args(p)
    p.add_argument("-o", "--out", required=True, help="dataset directory")
    p.add_argument("--sweep", action="append", metavar="KEY=SPEC",
                   help="r0=0.05:0.2:4 (lo:hi:n) or wind=5,10,20; repeat for a Cartesian product")
    p.add_argument("-k", "--realizations", type=int, default=1, help="realizations per sweep point")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th tick")
    p.add_argument("--crop", type=int, help="centre-crop stored frames to crop x crop")
    p.add_argument("--targets", default="intensity,phase", help="field,intensity,phase")
    p.add_argument("--label-aperture", type=float, help="aperture (m) for eta/scint/tilt labels")
    p.add_argument("--r0-frames", type=int, default=4, help="frames per sample for the measured-r0 label")
    p.add_argument("--compress", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="regenerate existing samples")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("analyze", help="r0 / fiber-coupling statistics of a saved run")
    p.add_argument("what", choices=("r0", "fiber"))
    p.add_argument("run_dir")
    p.add_argument("--frames", type=int, default=20, help="number of frames to analyse")
    p.add_argument("--aperture", type=float, help="aperture diameter (m)")
    p.add_argument("--a", type=float, default=1.12, help="fiber coupling geometry D/(2 w0)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("export", help="export a TensorFlow SavedModel")
    _add_config_args(p, steps=True)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("view", help="animate a saved run (needs matplotlib)")
    p.add_argument("run_dir")
    p.add_argument("--save", help="write .mp4 or .gif instead of opening a window")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=300)
    p.set_defaults(func=cmd_view)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
