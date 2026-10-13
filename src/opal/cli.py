"""``opal`` command-line interface.

    opal info                                     GPU / TensorFlow / RAM check
    opal config  [-c cfg.json] [--set k=v]        print or save the resolved configuration
    opal plan    [-c cfg.json]                    memory-guard dry run (which mode would run)
    opal run     [-c cfg.json] [-o DIR]           simulate and save frames
    opal pass    [-c cfg.json] [-o DIR]           satellite pass with exact r0 switching
    opal dataset [-c cfg.json] [-o DIR]           labelled ML dataset (sweeps x realizations)
    opal analyze fiber|r0 RUN [--frames N|all]    per-frame statistics, saved to RUN/analysis/
    opal export  [-c cfg.json] [-o DIR]           TensorFlow SavedModel
    opal view    PATH [--save movie.gif]          interactive viewer (runs, passes, datasets)

Without ``-o`` the writers create a new directory named after the date and the
main parameters, e.g. ``runs/2026-07-16_143205_N1024_L10_r0-0.15_T1000_seed42``
(``datasets/`` and ``exports/`` for the other commands, relative to the
current directory). Every log line is stamped with the time since the command
was launched, so the phase times add up to the wall-clock time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

from . import _timing
from ._version import __version__

MODES = ("auto", "device", "host", "chunked", "no_save")
OUTPUT_ROOTS = {"run": "runs", "pass": "runs", "dataset": "datasets", "export": "exports"}
CONFIG_FLAGS = ("config", "set", "grid", "layers", "r0", "seed")


class CLIError(Exception):
    """A user error, reported as ``opal <command>: error: ...`` without a traceback."""


def _parse_value(text):
    low = text.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        f = float(text)
    except ValueError:
        return text
    return int(f) if f.is_integer() and "." not in text and "e" not in low else f


def _overrides(args):
    out = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise CLIError(f"--set expects KEY=VALUE, got {item!r}")
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


def _resolve_seed(cfg):
    """Draw the random seed up front (when the config has none) so it can name the output."""
    from .config import get_value, set_value

    if get_value(cfg, "seed") is None:
        import numpy as np

        set_value(cfg, "seed", int(np.random.default_rng().integers(1, 2**31 - 1)))
    return int(get_value(cfg, "seed"))


def _given_config_flags(args):
    return [f"--{name.replace('_', '-')}" if name != "config" else "-c/--config"
            for name in CONFIG_FLAGS if getattr(args, name, None) not in (None, [], False)]


def _add_config_args(p, steps=True):
    p.add_argument("-c", "--config", help="JSON config file (default: built-in defaults)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any parameter, e.g. --set wind=10 --set pass_params.num_segments=20 "
                        "(repeatable)")
    p.add_argument("--grid", type=int, help="grid size N (pixels)")
    p.add_argument("--layers", type=int, help="number of layers")
    p.add_argument("--r0", type=float, help="total path r0 (m)")
    p.add_argument("--seed", type=int, help="master seed (default: config seed, else random)")
    if steps:
        p.add_argument("--steps", type=int, help="number of ticks")


def _quiet_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ----------------------------------------------------------------------
# Output naming and phase timing
# ----------------------------------------------------------------------
def output_name(kind, parts):
    """``<YYYY-MM-DD_HHMMSS>_<parts...>`` for a new output directory."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return "_".join([stamp] + [str(p) for p in parts if p])


def default_output_dir(kind, parts, root=None):
    """A new, non-existing directory under ``runs/``, ``datasets/`` or ``exports/``."""
    base = Path(root) if root is not None else Path(OUTPUT_ROOTS[kind])
    name = output_name(kind, parts)
    path, k = base / name, 2
    while path.exists():
        path, k = base / f"{name}_{k}", k + 1
    return path


def _param_parts(cfg, T=None, skip=()):
    from .config import get_value

    parts = {
        "N": f"N{int(get_value(cfg, 'N'))}",
        "layers": f"L{int(get_value(cfg, 'layers'))}",
        "r0": f"r0-{float(get_value(cfg, 'r0')):g}",
        "steps": f"T{int(T if T is not None else get_value(cfg, 'steps'))}",
        "seed": f"seed{get_value(cfg, 'seed')}",
    }
    return [v for k, v in parts.items() if k not in skip]


class _Session:
    """Console reporter and phase timer of one CLI command."""

    def __init__(self, args):
        self.quiet = bool(getattr(args, "quiet", False))
        self.sink = None if self.quiet else (lambda line: print(line, flush=True))
        self.phases = {}

    def say(self, msg):
        if self.sink is not None:
            self.sink(f"[opal {time.time() - _timing.clock():7.2f}s] {msg}")

    def plain(self, msg):
        if self.sink is not None:
            self.sink(msg)

    def startup(self):
        """Import TensorFlow, initialise the GPU list and report the start-up time."""
        _quiet_tf()
        from . import runners

        self.phases["startup"] = time.time() - _timing.clock()
        self.say(f"startup {self.phases['startup']:.1f} s (Python, NumPy/SciPy and TensorFlow import)")
        runners.warn_if_no_gpu(self.say)
        return runners

    def build(self, cfg, seed=None):
        from .config import get_value
        from .simulator import GpuSimulator

        N, L = int(get_value(cfg, "N")), int(get_value(cfg, "layers"))
        self.say(f"building simulator: N={N}, {L} layers (per-layer covariance, Cholesky and SVD on "
                 "the CPU; done once per run)")
        tb = time.time()
        sim = GpuSimulator(cfg, seed=seed, progress=_timing.layer_progress(self.say, N))
        self.phases["build"] = time.time() - tb
        rows = int(sum(sim.tf_extr_per_tick.numpy()))
        self.say(f"built simulator in {self.phases['build']:.1f} s | extrusion {rows} rows/tick over all "
                 f"layers (wind multiplier x{sim.extrusion_multiplier}; per-layer table: opal plan)")
        return sim

    def total(self, **later):
        total = time.time() - _timing.clock()
        phases = {**self.phases, **{k: v for k, v in later.items() if v is not None}}
        other = total - sum(phases.values())
        labels = {"plan": "plan (1 traced test step)"}
        text = " + ".join(f"{labels.get(k, k)} {v:.1f}" for k, v in phases.items())
        self.say(f"total wall time {total:.1f} s = {text} + other {max(other, 0.0):.1f} s")
        return total


def _header(cfg):
    from .config import get_value

    N, dx, r0 = int(get_value(cfg, "N")), float(get_value(cfg, "dx")), float(get_value(cfg, "r0"))
    return (f"OPAL {__version__} | N={N} layers={int(get_value(cfg, 'layers'))} r0={r0:g} m "
            f"D/r0={N * dx / r0:.1f} seed={get_value(cfg, 'seed')}")


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
        print("GPU: none visible - OPAL would run on the CPU (10-100x slower). Check that the CUDA and "
              "cuDNN libraries are on PATH (see the README install section).")
    ram = mem.host_available_bytes()
    if ram:
        print(f"host RAM available: {ram / mem.GIB:.1f} GiB")
    print(f"default output folders: {Path.cwd() / 'runs'}, datasets, exports")
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
    s = _Session(args)
    cfg = _load(args)
    _resolve_seed(cfg)
    s.plain(_header(cfg))
    s.startup()
    from . import memory as mem

    sim = s.build(cfg)
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
    print(f"  extrusion: {d['rows_per_tick_total']} rows per tick over all layers "
          f"(multiplier x{d['extrusion_multiplier']}: physical rows_per_tick = v*dt/dx is scaled so the "
          "lowest layer moves >= 1 row, then rounded up; effective_wind_mps is the resulting speed). "
          "Step time grows with this row count, not with N.")
    return 0


def cmd_run(args):
    s = _Session(args)
    if args.no_save and args.out:
        raise CLIError("--no-save and -o/--out cannot be combined")
    if args.model:
        given = _given_config_flags(args)
        if given:
            raise CLIError(f"--model runs an exported simulator with the config it was exported with, so "
                           f"{', '.join(given)} would be ignored. Re-export with the new settings "
                           "(opal export ...) or drop --model.")
        runners = s.startup()
        from .savedmodel import load

        tb = time.time()
        sim = load(args.model)
        s.phases["load"] = time.time() - tb
        s.say(f"loaded SavedModel {args.model} in {s.phases['load']:.1f} s")
        T = int(args.steps or sim.steps)
        if sim.config:
            from .config import get_value

            r0 = float(get_value(sim.config, "r0"))
            parts = [f"N{sim.nx_size}", f"L{sim.num_layers}", f"r0-{r0:g}", f"T{T}", f"seed{sim.seed}"]
        else:
            parts = [f"model-{Path(args.model).name}", f"T{T}"]
    else:
        cfg = _load(args)
        _resolve_seed(cfg)
        T = int(args.steps) if args.steps else None
        parts = _param_parts(cfg, T)
        s.plain(_header(cfg))
    out = None if args.no_save else (Path(args.out) if args.out else default_output_dir("run", parts))
    s.say(f"output: {out}" if out is not None else "output: none (--no-save, frames are discarded)")
    if not args.model:
        runners = s.startup()
        sim = s.build(cfg)

    state_in, state_out = args.state_in, args.state_out
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.exists() and state_in is None:
            state_in = ckpt
        state_out = state_out or ckpt
    try:
        captures = [int(x) for x in args.capture.split(",")] if args.capture else None
    except ValueError:
        raise CLIError(f"--capture expects comma-separated integers, got {args.capture!r}") from None
    timing = {f"{k}_s": v for k, v in s.phases.items()}
    result = runners.run(
        sim, args.steps, mode=args.mode, schedule=args.schedule, schedule_kind=args.schedule_kind,
        out_dir=out, chunk_size=args.chunk_size, capture_indices=captures,
        state_in=state_in, state_out=state_out, log=s.sink, timing=timing,
    )
    t = result.manifest["timing"]
    total = s.total(plan=t["plan_s"], simulate=t["simulate_s"], write=t["write_frames_s"] or None)
    if out is not None:
        s.say(f"output: {out}")
    if args.quiet:
        print(json.dumps({"mode": result.mode, "steps": result.num_steps,
                          "run_time_s": round(result.run_time_s, 3), "total_s": round(total, 3),
                          "out_dir": str(out) if out is not None else None}))
    return 0


def cmd_pass(args):
    s = _Session(args)
    if args.no_save and args.out:
        raise CLIError("--no-save and -o/--out cannot be combined")
    cfg = _load(args)
    pp = cfg.setdefault("pass_params", {})
    if args.segments:
        pp["num_segments"] = args.segments
    if args.captures:
        pp["wavefront_captures"] = args.captures
    if args.window is not None:
        pp["pass_window_s"] = args.window
    from .config import load_config

    cfg = load_config(cfg)
    _resolve_seed(cfg)
    out = None if args.no_save else (Path(args.out) if args.out else
                                     default_output_dir("pass", ["pass"] + _param_parts(cfg)))
    s.plain(_header(cfg))
    s.say(f"output: {out}" if out is not None else "output: none (--no-save)")
    s.startup()
    from .passes import run_pass

    res = run_pass(cfg, out, measure_r0=not args.no_r0, log=s.sink)
    t = res["manifest"]["timing"]
    s.phases["build"] = t.get("build_s")
    total = s.total(simulate=t.get("simulate_s"), write=t.get("write_s"))
    if out is not None:
        s.say(f"output: {out}")
    if args.quiet:
        m = res["manifest"]
        print(json.dumps({"steps": m["num_steps"], "segments": m["num_segments"],
                          "captures": m["wavefront_captures"], "run_time_s": round(m["run_time_s"], 3),
                          "r0_ratio_median": m["r0_ratio_median"], "total_s": round(total, 3),
                          "out_dir": str(out) if out is not None else None}))
    return 0


def cmd_dataset(args):
    from .config import get_value, set_value
    from .dataset import parse_sweep

    s = _Session(args)
    cfg = _load(args)
    stored = Path(args.out) / "manifest.json" if args.out else None
    if (stored is not None and stored.exists() and not args.no_resume
            and get_value(cfg, "seed") is None):
        try:
            base_seed = json.loads(stored.read_text(encoding="utf-8")).get("base_seed")
        except (OSError, ValueError):
            base_seed = None
        if base_seed is not None:
            set_value(cfg, "seed", int(base_seed))
            s.say(f"resuming {args.out}: using its base seed {base_seed} (pass --seed to override)")
    seed = _resolve_seed(cfg)
    sweeps = [parse_sweep(spec) for spec in (args.sweep or [])]
    if args.out:
        out = Path(args.out)
    else:
        swept = {k for k, _ in sweeps}
        skip = {"r0"} if swept & {"r0", "total_r0"} else set()
        tags = [f"k{args.realizations}"]
        if sweeps:
            tags.append("sweep-" + "-".join(f"{k}x{len(v)}" for k, v in sweeps))
        T = int(args.steps or get_value(cfg, "steps"))
        out = default_output_dir("dataset", _param_parts(cfg, T, skip=skip) + tags)
    s.plain(_header(cfg))
    s.say(f"output: {out}")
    s.startup()
    from .dataset import generate_dataset

    manifest = generate_dataset(
        cfg, out, sweep=args.sweep, realizations=args.realizations, num_steps=args.steps,
        stride=args.stride, crop=args.crop, targets=[t.strip() for t in args.targets.split(",")],
        seed=seed, label_aperture=args.label_aperture, r0_frames=args.r0_frames,
        compress=args.compress, resume=not args.no_resume, log=s.sink,
    )
    s.total(dataset=manifest["wall_time_s"])
    s.say(f"output: {out}")
    if s.quiet:
        print(json.dumps({"samples": manifest["num_samples"], "resumed": manifest["samples_resumed"],
                          "simulation_time_s": round(manifest["simulation_time_s"], 3),
                          "wall_time_s": round(manifest["wall_time_s"], 3), "out_dir": str(out)}))
    return 0


def _parse_frames(text):
    if text is None or str(text).lower() == "all":
        return "all"
    try:
        n = int(text)
    except ValueError:
        raise CLIError(f"--frames expects a number or 'all', got {text!r}") from None
    if n < 1:
        raise CLIError("--frames must be >= 1 (or 'all')")
    return n


def cmd_analyze(args):
    from .analysis.runs import NOTE, analyze_run
    from .frames import MIB

    s = _Session(args)
    summary = analyze_run(
        args.run_dir, args.what, frames=_parse_frames(args.frames), stride=args.stride,
        start=args.start, stop=args.stop, aperture=args.aperture, a=args.a, out_dir=args.out,
        save=not args.no_save, plot=args.plot, max_batch_bytes=int(args.batch_mb * MIB), log=s.say,
    )
    st = summary["statistics"]
    n = st.get("count", 0)
    if not n:
        print(f"{args.what}: no finite values over {summary['frames_analysed']} frames")
    elif args.what == "fiber":
        th = summary["theory"]
        theory = th.get("eta", th.get("mean"))
        per_frame = "" if "eta" in th else " (mean over per-frame target r0)"
        print(f"fiber coupling over {n} frames (D={summary['aperture_m']:.3f} m, "
              f"D/r0={summary['D_over_r0']:.2f}, a={summary['a']}): mean {st['mean']:.4f}, "
              f"std {st['std']:.4f}, min {st['min']:.4f}, max {st['max']:.4f}; von Karman theory{per_frame} "
              f"{theory:.4f} ({100 * (st['mean'] / theory - 1):+.1f} %)")
    else:
        r0 = summary["configured_total_r0_m"]
        tgt = summary.get("target", {})
        ratio = tgt.get("ratio_median")
        model = tgt.get("model", "")
        if model.startswith("pass"):
            ref = "pass target r0"
        elif model.startswith("turbulence schedule"):
            ref = f"schedule target path r0 (median {tgt['r0_m_median']:.4f} m)"
        else:
            ref = f"configured path r0 {r0} m"
        print(f"r0 over {n} frames: median {st['median']:.4f} m, mean {st['mean']:.4f} m, "
              f"std {st['std']:.4f} m (median ratio to {ref}: "
              + (f"{ratio:.3f})" if ratio is not None else "n/a)"))
    skipped = int(summary.get("frames_analysed", n)) - int(n)
    if n and skipped > 0:
        print(f"warning: {skipped} of {summary['frames_analysed']} frames gave no finite value and are excluded "
              "from the statistics (non-finite frames"
              + (", or no 1/e crossing of the coherence function" if args.what == "r0" else "")
              + "); see the CSV for which ones")
    for kind, path in (summary.get("files") or {}).items():
        print(f"wrote {path}")
    print(f"note: {NOTE}")
    return 0


def cmd_export(args):
    s = _Session(args)
    cfg = _load(args)
    _resolve_seed(cfg)
    out = Path(args.out) if args.out else default_output_dir("export", _param_parts(cfg))
    s.plain(_header(cfg))
    s.say(f"output: {out}")
    s.startup()
    from .savedmodel import export

    sim = s.build(cfg)
    te = time.time()
    export(sim, out)
    s.say(f"exported SavedModel (seed {sim.seed}) to {out}")
    s.total(export=time.time() - te)
    return 0


def cmd_view(args):
    from .viewer import view

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None
    if views == ["all"]:
        views = "all"
    try:
        out = view(args.path, views=views, start=args.start, stop=args.stop, stride=args.stride,
                   frame=args.frame, tick=args.tick, sample=args.sample, series=not args.no_series,
                   save=args.save, fps=args.fps, max_frames=args.max_frames)
    except RuntimeError as exc:
        raise CLIError(str(exc)) from None
    if out:
        print(f"wrote {out}")
    return 0


# ----------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="opal",
        description="OPAL: GPU-accelerated multilayer atmospheric turbulence simulator "
                    "for space-to-ground optical links.",
        epilog="Run `opal <command> -h` for the options of a command. Errors are reported without a "
               "traceback; set OPAL_DEBUG=1 to see it.",
    )
    ap.add_argument("--version", action="version", version=f"opal {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("info", help="show GPU / TensorFlow / RAM status")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("config", help="print or save the resolved configuration",
                       description="Print the configuration after defaults, the -c file and the flags/--set "
                                   "overrides are applied (and validated), or save it with -o to start a "
                                   "new config file.")
    _add_config_args(p)
    p.add_argument("-o", "--output", help="write the resolved config to this file")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("plan", help="memory-guard dry run: which mode would run and why",
                       description="Build the simulator, measure its baseline GPU memory and report which "
                                   "mode `opal run` would use and the predicted peak of every mode, "
                                   "without simulating.")
    _add_config_args(p)
    p.add_argument("--mode", choices=MODES, default="auto",
                   help="mode to check (default auto: the one the memory guard picks)")
    p.add_argument("--chunk-size", type=int, default=100, help="ticks per chunk for the chunked estimate")
    p.add_argument("--no-out", action="store_true", help="plan as if the run were started with --no-save")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="simulate and save the received fields",
                       description="Simulate and save the received fields. Without -o the run is written "
                                   "to runs/<date>_<time>_N.._L.._r0-.._T.._seed..")
    _add_config_args(p)
    p.add_argument("--mode", choices=MODES, default="auto",
                   help="auto (default) lets the memory guard pick device/host/chunked/no_save")
    p.add_argument("-o", "--out", help="run directory (default: a new folder under runs/)")
    p.add_argument("--no-save", action="store_true",
                   help="do not write anything (benchmarking; frames are discarded)")
    p.add_argument("--chunk-size", type=int, default=100, help="ticks per chunk in chunked mode")
    p.add_argument("--schedule", help=".npy/.npz/.csv/.json turbulence schedule: [T, layers], [T, 1] or [T] "
                                      "(one value per tick for all layers); a single row is held constant")
    p.add_argument("--schedule-kind", choices=("b_scale", "cn2", "r0", "path_r0"), default="b_scale",
                   help="b_scale: innovation multiplier b; cn2: Cn^2 multiplier; r0: per-layer r0 (m); "
                        "path_r0: total path r0 (m) per tick (default b_scale)")
    p.add_argument("--capture", help="comma-separated ticks at which to capture all layer screens")
    p.add_argument("--state-in", help="start from this checkpoint (.npz)")
    p.add_argument("--state-out", help="save the final state here (.npz)")
    p.add_argument("--checkpoint", help="resume from this file if it exists, then overwrite it")
    p.add_argument("--model", help="run an exported SavedModel instead of building from a config "
                                   "(cannot be combined with config flags)")
    p.add_argument("-q", "--quiet", action="store_true", help="print one JSON summary line only")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pass", help="satellite pass with exact per-segment r0 switching",
                       description="Simulate a satellite pass. Without -o the output is written to "
                                   "runs/<date>_<time>_pass_N.._L.._r0-.._T.._seed..")
    _add_config_args(p)
    p.add_argument("-o", "--out", help="output directory (default: a new folder under runs/)")
    p.add_argument("--no-save", action="store_true", help="keep the captures in memory only")
    p.add_argument("--segments", type=int, help="number of r0 segments (must divide the steps)")
    p.add_argument("--captures", type=int, help="number of wavefront captures")
    p.add_argument("--window", type=float, help="only simulate +-window/2 s around zenith")
    p.add_argument("--no-r0", action="store_true", help="skip the per-capture r0 measurement")
    p.add_argument("-q", "--quiet", action="store_true", help="print one JSON summary line only")
    p.set_defaults(func=cmd_pass)

    p = sub.add_parser("dataset", help="generate a labelled ML dataset",
                       description="Generate a labelled ML dataset. Without -o it is written to "
                                   "datasets/<date>_<time>_N.._L.._T.._seed.._k... Pass the same -o "
                                   "again to resume an interrupted dataset.")
    _add_config_args(p)
    p.add_argument("-o", "--out", help="dataset directory (default: a new folder under datasets/)")
    p.add_argument("--sweep", action="append", metavar="KEY=SPEC",
                   help="r0=0.05:0.2:4 (lo:hi:n) or wind=5,10,20; repeat for a Cartesian product")
    p.add_argument("-k", "--realizations", type=int, default=1, help="realizations per sweep point")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th tick")
    p.add_argument("--crop", type=int, help="centre-crop stored frames to crop x crop")
    p.add_argument("--targets", default="intensity,phase", help="any of field,intensity,phase")
    p.add_argument("--label-aperture", type=float, help="aperture (m) for eta/scint/tilt labels")
    p.add_argument("--r0-frames", type=int, default=4, help="frames per sample for the measured-r0 label")
    p.add_argument("--compress", action="store_true",
                   help="store samples with np.savez_compressed (smaller, slower to write and read)")
    p.add_argument("--no-resume", action="store_true",
                   help="delete existing samples in -o and regenerate everything (by default an existing "
                        "dataset with identical settings is resumed; different settings are an error)")
    p.add_argument("-q", "--quiet", action="store_true", help="print one JSON summary line only")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("analyze", help="fiber coupling / r0 per frame of a saved run",
                       description="Measure fiber coupling or r0 frame by frame. Frames are read in "
                                   "bounded batches, so any run size works. Results go to "
                                   "RUN/analysis/<what>.csv and <what>_summary.json (and <what>.png "
                                   "with --plot); `opal view RUN` shows them next to the frames.")
    p.add_argument("what", choices=("fiber", "r0"))
    p.add_argument("run_dir", help="run or pass directory (or a frames .npy file)")
    p.add_argument("--frames", default="20",
                   help="how many evenly spaced frames of the selected range, or 'all' (default 20)")
    p.add_argument("--start", type=int, help="first frame index of the range")
    p.add_argument("--stop", type=int, help="end of the range (exclusive)")
    p.add_argument("--stride", type=int, default=1, help="use every n-th frame of the range")
    p.add_argument("--aperture", type=float,
                   help="aperture diameter (m); fiber default 0.4*N*dx, r0 default the full grid")
    p.add_argument("--a", type=float, default=1.12, help="fiber coupling geometry D/(2 w0) (default 1.12)")
    p.add_argument("-o", "--out", help="results directory (default RUN/analysis)")
    p.add_argument("--no-save", action="store_true", help="print the statistics only")
    p.add_argument("--plot", action="store_true", help="also save a time-series + histogram PNG")
    p.add_argument("--batch-mb", type=float, default=256, help="RAM budget per batch in MiB (default 256)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("export", help="export a TensorFlow SavedModel",
                       description="Export a TensorFlow SavedModel. Without -o it is written to "
                                   "exports/<date>_<time>_N.._L.._r0-.._T.._seed..")
    _add_config_args(p, steps=True)
    p.add_argument("-o", "--out", help="export directory (default: a new folder under exports/)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("view", help="interactive viewer for runs, passes and datasets",
                       description="Open an interactive viewer (play/pause, slider, step, go to tick, "
                                   "snapshots) or render a GIF/MP4/PNG with --save. Works on run and pass "
                                   "directories, chunked runs, datasets, dataset samples (.npz) and "
                                   "frame .npy files. Frames are read one at a time.")
    p.add_argument("path", help="run / pass / dataset directory, sample_XXXXX.npz or frames .npy")
    p.add_argument("--views", default="intensity,phase",
                   help="panels: intensity, log_intensity, phase, phase_tt (tip/tilt removed) or all "
                        "(default intensity,phase)")
    p.add_argument("--start", type=int, help="first frame index")
    p.add_argument("--stop", type=int, help="end frame index (exclusive)")
    p.add_argument("--stride", type=int, default=1, help="show every n-th frame")
    p.add_argument("--frame", type=int, help="open at (or, with --save x.png, render) this frame index")
    p.add_argument("--tick", type=int, help="like --frame, but by simulation tick")
    p.add_argument("--sample", type=int, default=0, help="dataset sample to open (default 0)")
    p.add_argument("--no-series", action="store_true", help="hide the time-series strip")
    p.add_argument("--save", help="write .gif/.mp4 (animation) or .png (one frame) instead of a window")
    p.add_argument("--fps", type=int, default=20, help="playback / saved animation frame rate (default 20)")
    p.add_argument("--max-frames", type=int, default=300,
                   help="frame limit for --save animations (default 300; 0 = no limit)")
    p.set_defaults(func=cmd_view)
    return ap


def main(argv=None):
    _timing.mark_start(None if argv is None else time.time())
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print(f"opal {args.command}: interrupted", file=sys.stderr, flush=True)
        return 130
    except (CLIError, ValueError, KeyError, FileNotFoundError, NotADirectoryError) as exc:
        if os.environ.get("OPAL_DEBUG"):
            raise
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"opal {args.command}: error: {msg}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ != "LinAlgError" or os.environ.get("OPAL_DEBUG"):
            raise
        print(f"opal {args.command}: error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
