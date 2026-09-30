"""Run modes, state checkpoints, schedules and run outputs.

:func:`run` is the single entry point used by the CLI and recommended for
scripts. It measures the simulator's static GPU footprint, lets the memory
guard (:mod:`opal.memory`) choose a run mode, executes it with automatic
out-of-memory fallback, and writes a self-describing run directory::

    <out_dir>/
      frames.npy                  complex64 [T, N, N]   (device / host)
      frames_chunk_000.npy ...    complex64 [chunk, N, N] (chunked)
      frames_chunk_manifest.json
      final_frame.npy             complex64 [N, N]      (no_save)
      captures.npz                layer screens / r0 at the capture ticks
      manifest.json               config, seed, mode, timings, memory
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import memory as mem
from ._version import __version__

__all__ = [
    "RunResult",
    "run",
    "save_state",
    "load_state",
    "load_schedule",
    "build_schedule",
    "load_frames",
    "iter_frames",
    "save_npy",
]

SCHEDULE_KINDS = ("b_scale", "cn2", "r0")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _np(value):
    if isinstance(value, dict):
        return {k: _np(v) for k, v in value.items()}
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
        fh.write("\n")


class _Log:
    """Timestamped console reporter (``log=None`` silences it)."""

    def __init__(self, sink):
        self.sink = sink
        self.t0 = time.time()

    def __call__(self, msg):
        if self.sink is None:
            return
        self.sink(f"[opal {time.time() - self.t0:7.2f}s] {msg}")


def _default_sink(line):
    print(line, flush=True)


# ----------------------------------------------------------------------
# State checkpoints
# ----------------------------------------------------------------------
def save_state(simulator, path):
    """Write the mutable simulator state (screens, RNG states, step, b-scales) to ``.npz``.

    Restoring it with :func:`load_state` continues the run bit-exactly.
    """
    state = _np(simulator.get_state())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        phase_screens=state["phase_screens"],
        rng_states=state["rng_states"],
        global_step=np.asarray(state["global_step"], dtype=np.int64),
        b_scales=state["b_scales"],
    )
    return path


def load_state(simulator, path):
    """Restore a state written by :func:`save_state`."""
    data = np.load(Path(path))
    expected = (int(simulator.num_layers), int(simulator.nx_size), int(simulator.nx_size))
    if tuple(data["phase_screens"].shape) != expected:
        raise ValueError(
            f"Checkpoint {path} holds screens of shape {data['phase_screens'].shape}, "
            f"simulator expects {expected}"
        )
    return simulator.set_state(
        tf.convert_to_tensor(data["phase_screens"], dtype=tf.float32),
        tf.convert_to_tensor(data["rng_states"], dtype=tf.int64),
        tf.convert_to_tensor(np.asarray(data["global_step"]).reshape(()), dtype=tf.int64),
        tf.convert_to_tensor(data["b_scales"], dtype=tf.float32),
    )


# ----------------------------------------------------------------------
# Turbulence-strength schedules
# ----------------------------------------------------------------------
def load_schedule(path):
    """Load a ``[steps, layers]`` schedule from ``.npy``, ``.npz``, ``.json`` or ``.csv``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        data = archive[archive.files[0]]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported schedule file type: {path}")
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError("A schedule must have shape [steps, layers].")
    return data


def build_schedule(schedule, num_steps, num_layers, base_r0=None, kind="b_scale"):
    """Turn a schedule into the per-tick ``b_scale`` array the graphs consume.

    Parameters
    ----------
    schedule : None, array-like or path
        ``[steps, layers]`` values (a single row is held constant). Shorter
        schedules hold their last row; longer ones are truncated.
    kind : {"b_scale", "cn2", "r0"}
        ``b_scale``: innovation multiplier b (Cn^2 scales by b^2).
        ``cn2``: Cn^2 multiplier m (b = sqrt(m)).
        ``r0``: absolute per-layer r0 in metres (b = (r0_base / r0)^(5/6)).
    """
    if schedule is None:
        return np.ones((num_steps, num_layers), dtype=np.float32)
    if isinstance(schedule, (str, os.PathLike)):
        schedule = load_schedule(schedule)
    schedule = np.asarray(schedule, dtype=np.float32)
    if schedule.ndim == 1:
        schedule = schedule.reshape(1, -1)
    if schedule.shape[1] != num_layers:
        raise ValueError(f"Schedule has {schedule.shape[1]} layers, simulator has {num_layers}.")

    if kind == "b_scale":
        out = schedule
    elif kind == "cn2":
        out = np.sqrt(np.maximum(schedule, 1e-24))
    elif kind == "r0":
        if base_r0 is None:
            raise ValueError("kind='r0' needs the base per-layer r0 array")
        base = np.asarray(base_r0, dtype=np.float32).reshape(1, -1)
        out = np.power(base / np.maximum(schedule, 1e-24), 5.0 / 6.0)
    else:
        raise ValueError(f"Unknown schedule kind {kind!r}; choose from {SCHEDULE_KINDS}")

    if out.shape[0] > num_steps:
        out = out[:num_steps]
    elif out.shape[0] < num_steps:
        out = np.concatenate([out, np.repeat(out[-1:], num_steps - out.shape[0], axis=0)], axis=0)
    return out.astype(np.float32)


# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------
def save_npy(path, array, max_write_bytes=256 * mem.MIB):
    """Write a large ``.npy`` in bounded slices through a temp file, then rename."""
    start = time.time()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=array.dtype, shape=array.shape)
        if array.ndim == 0:
            mm[...] = array
        else:
            rows = max(1, int(max_write_bytes // max(1, int(array[0].nbytes))))
            for i in range(0, array.shape[0], rows):
                mm[i:i + rows] = array[i:i + rows]
        mm.flush()
        del mm
        if path.exists():
            path.unlink()
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return {"path": str(path), "write_time_s": time.time() - start}


def _clear_frame_outputs(out_dir):
    for pattern in ("frames.npy", "frames.npy.tmp", "frames_chunk_*.npy", "frames_chunk_*.npy.tmp",
                    "frames_chunk_manifest.json", "final_frame.npy"):
        for p in Path(out_dir).glob(pattern):
            if p.is_file():
                p.unlink()


def load_frames(run_dir, mmap=True):
    """Load the frames of a run directory as one ``[T, N, N]`` array.

    Device/host runs are memory-mapped (no RAM cost). Chunked runs are
    concatenated, so prefer :func:`iter_frames` for very large ones.
    """
    run_dir = Path(run_dir)
    single = run_dir / "frames.npy"
    if single.exists():
        return np.load(single, mmap_mode="r" if mmap else None)
    manifest = run_dir / "frames_chunk_manifest.json"
    if manifest.exists():
        chunks = json.loads(manifest.read_text(encoding="utf-8"))["chunks"]
        return np.concatenate([np.load(run_dir / Path(c["file"]).name) for c in chunks], axis=0)
    final = run_dir / "final_frame.npy"
    if final.exists():
        return np.load(final)[None]
    raise FileNotFoundError(f"No frames found in {run_dir}")


def iter_frames(run_dir, stride=1):
    """Yield ``(step, frame)`` from any run directory without loading it all."""
    run_dir = Path(run_dir)
    single = run_dir / "frames.npy"
    if single.exists():
        frames = np.load(single, mmap_mode="r")
        for t in range(0, frames.shape[0], stride):
            yield t, np.asarray(frames[t])
        return
    manifest = run_dir / "frames_chunk_manifest.json"
    if manifest.exists():
        chunks = json.loads(manifest.read_text(encoding="utf-8"))["chunks"]
        for c in chunks:
            frames = np.load(run_dir / Path(c["file"]).name, mmap_mode="r")
            for t in range(c["start_step"], c["end_step_exclusive"]):
                if t % stride == 0:
                    yield t, np.asarray(frames[t - c["start_step"]])
        return
    raise FileNotFoundError(f"No frame series found in {run_dir}")


# ----------------------------------------------------------------------
# Run modes
# ----------------------------------------------------------------------
@dataclass
class RunResult:
    """Outcome of :func:`run`.

    ``frames`` is a ``[T, N, N]`` complex64 array (memory-mapped when written to
    disk) for device/host runs, ``None`` otherwise. ``final_frame`` is always set
    when the mode produced one.
    """

    mode: str
    num_steps: int
    run_time_s: float
    frames: np.ndarray | None = None
    final_frame: np.ndarray | None = None
    captures: dict | None = None
    out_dir: Path | None = None
    manifest: dict = field(default_factory=dict)

    @property
    def steps_per_second(self):
        return self.num_steps / self.run_time_s if self.run_time_s > 0 else float("inf")


def _progress_every(num_steps):
    return max(1, num_steps // 10)


def _run_device(sim, T, sched, captures, out_dir, log):
    t = tf.constant(T, dtype=tf.int32)
    s = tf.convert_to_tensor(sched, dtype=tf.float32)
    start = time.time()
    if captures:
        res = _np(sim.simulate_steps_capture(t, s, tf.convert_to_tensor(captures, dtype=tf.int32)))
        frames = res.pop("frames")
        cap = res
    else:
        frames = _np(sim.simulate_steps(t, s))
        cap = None
    run_time = time.time() - start
    log(f"device: {T} steps in {run_time:.2f} s ({1e3 * run_time / T:.2f} ms/step)")
    files = {}
    if out_dir is not None:
        w = save_npy(Path(out_dir) / "frames.npy", frames)
        files["frames"] = "frames.npy"
        log(f"wrote frames.npy ({frames.nbytes / mem.GIB:.2f} GiB) in {w['write_time_s']:.1f} s")
        frames = np.load(Path(out_dir) / "frames.npy", mmap_mode="r")
    return {"frames": frames, "final_frame": np.asarray(frames[-1]), "captures": cap,
            "run_time_s": run_time, "files": files}


def _run_host(sim, T, sched, captures, out_dir, log):
    N = int(sim.nx_size)
    files = {}
    if out_dir is not None:
        final_path = Path(out_dir) / "frames.npy"
        tmp_path = final_path.with_name("frames.npy.tmp")
        frames = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.complex64, shape=(T, N, N))
    else:
        frames = np.empty((T, N, N), dtype=np.complex64)

    every = _progress_every(T)
    start = time.time()

    def sink(step, frame):
        frames[step] = frame
        done = step + 1
        if done % every == 0 or done == T:
            el = time.time() - start
            gpu = mem.memory_info()["current"] / mem.GIB
            log(f"host: step {done:>6d}/{T} ({100 * done / T:5.1f}%) "
                f"{done / max(el, 1e-9):7.1f} steps/s | GPU in use {gpu:.2f} GiB")

    res = sim.simulate_steps_offload(T, sched, capture_indices=captures or None, frame_callback=sink)
    run_time = time.time() - start
    cap = None
    if captures:
        cap = {k: v for k, v in res.items() if k != "frames"}

    if out_dir is not None:
        frames.flush()
        del frames
        if final_path.exists():
            final_path.unlink()
        tmp_path.replace(final_path)
        frames = np.load(final_path, mmap_mode="r")
        files["frames"] = "frames.npy"
    return {"frames": frames, "final_frame": np.asarray(frames[-1]), "captures": cap,
            "run_time_s": run_time, "files": files}


def _run_chunked(sim, T, sched, chunk, out_dir, log):
    chunk = max(1, int(chunk))
    records = []
    pending = None
    last = None
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for idx, s0 in enumerate(range(0, T, chunk)):
            s1 = min(s0 + chunk, T)
            c0 = time.time()
            arr = _np(sim.simulate_steps(tf.constant(s1 - s0, dtype=tf.int32),
                                         tf.convert_to_tensor(sched[s0:s1], dtype=tf.float32)))
            compute = time.time() - c0
            last = arr[-1].copy()
            if pending is not None:
                pending.result()
            rec = {"index": idx, "start_step": s0, "end_step_exclusive": s1, "steps": s1 - s0,
                   "compute_time_s": compute, "file": None}
            if out_dir is not None:
                name = f"frames_chunk_{idx:03d}.npy"
                rec["file"] = name
                pending = pool.submit(save_npy, Path(out_dir) / name, arr)
            records.append(rec)
            log(f"chunked: chunk {idx:3d} steps {s0}-{s1 - 1} in {compute:.2f} s "
                f"({(s1 - s0) / max(compute, 1e-9):.1f} steps/s)"
                + (f" -> {rec['file']}" if rec["file"] else " (not saved)"))
        if pending is not None:
            pending.result()
    run_time = time.time() - start
    files = {}
    if out_dir is not None:
        _write_json(Path(out_dir) / "frames_chunk_manifest.json",
                    {"total_steps": T, "chunk_size": chunk, "dtype": "complex64",
                     "frame_shape": [int(sim.nx_size)] * 2, "chunks": records})
        files["chunks"] = "frames_chunk_manifest.json"
    return {"frames": None, "final_frame": last, "captures": None, "run_time_s": run_time,
            "files": files, "chunks": records}


def _run_no_save(sim, T, sched, out_dir, log):
    start = time.time()
    final = _np(sim.simulate_steps_no_save(tf.constant(T, dtype=tf.int32),
                                           tf.convert_to_tensor(sched, dtype=tf.float32)))
    run_time = time.time() - start
    log(f"no_save: {T} steps in {run_time:.2f} s ({T / max(run_time, 1e-9):.1f} steps/s)")
    files = {}
    if out_dir is not None:
        np.save(Path(out_dir) / "final_frame.npy", final)
        files["final_frame"] = "final_frame.npy"
    return {"frames": None, "final_frame": final, "captures": None, "run_time_s": run_time,
            "files": files}


def _is_oom(exc):
    return isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError, OSError))


def run(
    simulator,
    num_steps=None,
    *,
    mode="auto",
    schedule=None,
    schedule_kind="b_scale",
    out_dir=None,
    chunk_size=mem.DEFAULT_CHUNK_SIZE,
    capture_indices=None,
    state_in=None,
    state_out=None,
    log=_default_sink,
):
    """Run a simulation in the requested (or automatically chosen) mode.

    Parameters
    ----------
    simulator : GpuSimulator or LoadedSimulator
    num_steps : int, optional
        Ticks to simulate (default: the config's ``num_steps``).
    mode : {"auto", "device", "host", "chunked", "no_save"}
        ``auto`` lets the memory guard decide. Explicit modes are downgraded
        if they would not fit.
    schedule, schedule_kind
        Optional turbulence-strength schedule, see :func:`build_schedule`.
    out_dir : path, optional
        Run directory. Without it frames are returned in memory only.
    chunk_size : int
        Requested chunk length for ``chunked`` (capped by the guard).
    capture_indices : sequence of int, optional
        Ticks at which to also capture all layer screens and r0 values
        (device and host modes).
    state_in : path, optional
        Start from a checkpoint instead of the base state.
    state_out : path, optional
        Write the final state here so a later run can continue bit-exactly.
    log : callable or None
        Line sink for progress messages (default: print).
    """
    log = _Log(log)
    T = int(num_steps if num_steps is not None else simulator.steps)
    if T < 1:
        raise ValueError("num_steps must be >= 1")
    L = int(simulator.num_layers)
    N = int(simulator.nx_size)
    live = hasattr(simulator, "simulate_steps_offload")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_frame_outputs(out_dir)

    base_r0 = _np(simulator.get_state())["base_r0"]
    sched = build_schedule(schedule, T, L, base_r0=base_r0, kind=schedule_kind)
    captures = sorted({int(i) for i in (capture_indices or []) if 0 <= int(i) < T})

    baseline = mem.measure_baseline_peak_bytes(simulator)
    estimate = mem.estimate_memory(T, N, baseline, requested_chunk_size=chunk_size)
    selected, chunk, reason = mem.select_mode(mode, estimate, live=live,
                                              has_output_dir=out_dir is not None)
    for line in mem.format_plan(estimate, mode, selected, chunk):
        log(line)
    if reason != "requested":
        log(f"guard: {reason}")

    current, chunk_now = selected, (chunk or min(int(chunk_size), T))
    fallbacks = []
    while True:
        if state_in is not None:
            load_state(simulator, state_in)
        else:
            simulator.reset_to_base()
        start_step = int(_np(simulator.get_state())["global_step"])
        mem.reset_memory_stats()
        mem_before = mem.memory_sample()
        log(f"running {current} mode: {T} steps, N={N}, {L} layers"
            + (f", chunk={chunk_now}" if current == "chunked" else "")
            + (f", from checkpoint step {start_step}" if state_in is not None else ""))
        try:
            if current == "device":
                res = _run_device(simulator, T, sched, captures, out_dir, log)
            elif current == "host":
                res = _run_host(simulator, T, sched, captures, out_dir, log)
            elif current == "chunked":
                if captures:
                    log("chunked mode does not collect captures; use device or host mode for them")
                res = _run_chunked(simulator, T, sched, chunk_now, out_dir, log)
            else:
                res = _run_no_save(simulator, T, sched, out_dir, log)
            break
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            log(f"{current} failed ({type(exc).__name__}: {msg[:120]})")
            fallbacks.append({"mode": current, "error": msg[:500]})
            if out_dir is not None:
                _clear_frame_outputs(out_dir)
            if current == "device":
                current = "host" if live else "chunked"
            elif current == "host":
                current = "chunked"
            elif current == "chunked" and chunk_now > 1:
                chunk_now = max(1, chunk_now // 2)
            elif current != "no_save":
                current = "no_save"
            else:
                raise
            log(f"falling back to {current}" + (f" (chunk={chunk_now})" if current == "chunked" else ""))

    mem_after = mem.memory_sample()
    peak = mem.memory_info()["peak"]
    run_time = res["run_time_s"]
    end_step = int(_np(simulator.get_state())["global_step"])
    log(f"done: {T} steps in {run_time:.2f} s | {T / max(run_time, 1e-9):.1f} steps/s | "
        f"peak GPU {peak / mem.GIB:.2f} GiB")

    files = dict(res["files"])
    if state_out is not None:
        save_state(simulator, state_out)
        files["state_out"] = str(state_out)
        log(f"saved state (step {end_step}) to {state_out}")

    if out_dir is not None and res["captures"]:
        cap = res["captures"]
        np.savez(out_dir / "captures.npz",
                 capture_indices=np.asarray(cap.get("capture_indices", captures)),
                 captured_screens=cap["captured_screens"],
                 captured_r0=cap["captured_r0"],
                 captured_b_scales=cap["captured_b_scales"])
        files["captures"] = "captures.npz"

    config = getattr(simulator, "config", None)
    manifest = {
        "opal_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": getattr(simulator, "seed", None),
        "config": config,
        "grid": N,
        "num_layers": L,
        "num_steps": T,
        "start_step": start_step,
        "end_step": end_step,
        "frame_dtype": "complex64",
        "frame_shape": [N, N],
        "mode_requested": mode,
        "mode_selected": selected,
        "mode_final": current,
        "guard_reason": reason,
        "chunk_size": chunk_now if current == "chunked" else None,
        "fallbacks": fallbacks,
        "run_time_s": run_time,
        "steps_per_second": T / max(run_time, 1e-9),
        "gpu": {"name": mem.gpu_name(), "total_bytes": estimate["gpu_total_bytes"],
                "baseline_peak_bytes": baseline, "run_peak_bytes": peak,
                "before": mem_before, "after": mem_after},
        "memory_estimate": estimate,
        "schedule": {"kind": schedule_kind,
                     "source": str(schedule) if isinstance(schedule, (str, os.PathLike)) else
                     ("array" if schedule is not None else None),
                     "constant": bool(np.all(sched == sched[0]))},
        "capture_indices": captures,
        "state_in": str(state_in) if state_in is not None else None,
        "files": files,
    }
    if "chunks" in res:
        manifest["chunks"] = res["chunks"]
    if out_dir is not None:
        _write_json(out_dir / "manifest.json", manifest)
        log(f"wrote {out_dir / 'manifest.json'}")

    return RunResult(
        mode=current,
        num_steps=T,
        run_time_s=run_time,
        frames=res["frames"],
        final_frame=res["final_frame"],
        captures=res["captures"],
        out_dir=out_dir,
        manifest=manifest,
    )
