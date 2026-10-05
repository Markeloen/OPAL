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

from . import _timing
from . import memory as mem
from ._version import __version__
from .frames import iter_frames, load_frames, open_frames  # noqa: F401  (re-exported)

__all__ = [
    "RunResult",
    "run",
    "save_state",
    "load_state",
    "load_schedule",
    "build_schedule",
    "load_frames",
    "iter_frames",
    "open_frames",
    "save_npy",
    "warn_if_no_gpu",
]

SCHEDULE_KINDS = ("b_scale", "cn2", "r0", "path_r0")


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
    """Timestamped console reporter (``log=None`` silences it).

    Timestamps count from the process start when the CLI set it
    (:mod:`opal._timing`), otherwise from the creation of this reporter.
    """

    def __init__(self, sink, t0=None):
        self.sink = sink
        self.t0 = _timing.clock(t0)

    def __call__(self, msg):
        if self.sink is None:
            return
        self.sink(f"[opal {time.time() - self.t0:7.2f}s] {msg}")


def _default_sink(line):
    print(line, flush=True)


_GPU_WARNED = False


def warn_if_no_gpu(log):
    """Log a loud warning (once per process) when TensorFlow sees no GPU; it silently uses the CPU."""
    global _GPU_WARNED
    if not tf.config.list_physical_devices("GPU"):
        if not _GPU_WARNED:
            log("WARNING: TensorFlow sees no GPU - running on the CPU, typically 10-100x slower. "
                "Run `opal info` and see the README install notes (CUDA/cuDNN on PATH).")
            _GPU_WARNED = True
        return False
    return True


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
    """Load a schedule array from ``.npy``, ``.npz``, ``.json`` or ``.csv``.

    The array is returned as stored (0-, 1- or 2-D); :func:`build_schedule`
    decides how it maps onto ticks and layers.
    """
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
    if data.ndim > 2:
        raise ValueError(f"A schedule must have shape [steps, layers]; {path} has shape {data.shape}.")
    return data


def build_schedule(schedule, num_steps, num_layers, base_r0=None, kind="b_scale"):
    """Turn a schedule into the per-tick ``b_scale`` array the graphs consume.

    Parameters
    ----------
    schedule : None, array-like or path
        ``[steps, layers]`` values, one row per tick and one column per
        layer. A single row is held constant for the whole run; shorter
        schedules hold their last row; longer ones are truncated. A single
        column ``[steps, 1]`` applies the same value to every layer. A 1-D
        array of length ``layers`` is one constant row; any other 1-D length
        is a per-tick series applied to every layer. A scalar is constant.
    kind : {"b_scale", "cn2", "r0", "path_r0"}
        ``b_scale``: innovation multiplier b (Cn^2 scales by b^2).
        ``cn2``: Cn^2 multiplier m (b = sqrt(m)).
        ``r0``: absolute per-layer r0 in metres (b = (r0_base / r0)^(5/6)).
        ``path_r0``: total path r0 in metres, one value per tick (shape
        ``[steps]`` or ``[steps, 1]``); every layer is scaled by the same
        b = (r0_path_base / r0_path)^(5/6).
    base_r0 : array-like, optional
        The simulator's per-layer r0 (needed for ``r0`` and ``path_r0``).
    """
    if kind not in SCHEDULE_KINDS:
        raise ValueError(f"Unknown schedule kind {kind!r}; choose from {SCHEDULE_KINDS}")
    if schedule is None:
        return np.ones((num_steps, num_layers), dtype=np.float32)
    if isinstance(schedule, (str, os.PathLike)):
        schedule = load_schedule(schedule)
    schedule = np.asarray(schedule, dtype=np.float32)
    if schedule.ndim == 0:
        schedule = schedule.reshape(1, 1)
    elif schedule.ndim == 1:
        per_layer_row = schedule.size == num_layers and kind != "path_r0"
        schedule = schedule.reshape(1, -1) if per_layer_row else schedule.reshape(-1, 1)
    if schedule.ndim != 2 or schedule.shape[0] == 0:
        raise ValueError(f"A schedule must have shape [steps, layers]; got {schedule.shape}.")
    if kind == "path_r0" and schedule.shape[1] != 1:
        raise ValueError(f"kind='path_r0' takes one path r0 per tick, shape [steps] or [steps, 1]; got {schedule.shape}.")
    if schedule.shape[1] == 1 and num_layers > 1:
        schedule = np.repeat(schedule, num_layers, axis=1)
    if schedule.shape[1] != num_layers:
        raise ValueError(
            f"Schedule has shape {schedule.shape} ({schedule.shape[1]} layer columns) but the simulator has "
            f"{num_layers} layers. Use [steps, {num_layers}], [steps, 1] or a 1-D per-tick series for all layers."
        )
    positive = kind in ("cn2", "r0", "path_r0")
    if not np.all(np.isfinite(schedule)) or np.any(schedule <= 0 if positive else schedule < 0):
        raise ValueError(f"Schedule values must be finite and {'positive' if positive else 'non-negative'} "
                         f"for kind={kind!r}.")

    if kind == "b_scale":
        out = schedule
    elif kind == "cn2":
        out = np.sqrt(schedule)
    else:
        if base_r0 is None:
            raise ValueError(f"kind={kind!r} needs the simulator's per-layer r0 (base_r0)")
        base = np.asarray(base_r0, dtype=np.float64).reshape(1, -1)
        if kind == "path_r0":
            ok = np.isfinite(base) & (base > 0)
            base = np.full_like(base, np.sum(base[ok] ** (-5.0 / 3.0)) ** (-3.0 / 5.0))
        out = np.power(base / schedule.astype(np.float64), 5.0 / 6.0)

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
                    "frames_chunk_manifest.json", "final_frame.npy", "schedule.npy", "captures.npz"):
        for p in Path(out_dir).glob(pattern):
            if p.is_file():
                p.unlink()


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
    write_time = 0.0
    if out_dir is not None:
        w = save_npy(Path(out_dir) / "frames.npy", frames)
        write_time = w["write_time_s"]
        files["frames"] = "frames.npy"
        log(f"wrote frames.npy ({frames.nbytes / mem.GIB:.2f} GiB) in {write_time:.1f} s")
        frames = np.load(Path(out_dir) / "frames.npy", mmap_mode="r")
    return {"frames": frames, "final_frame": np.asarray(frames[-1]), "captures": cap,
            "run_time_s": run_time, "write_time_s": write_time, "files": files}


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

    write_time = 0.0
    if out_dir is not None:
        # The frames were written into a memory-mapped file while running; the
        # OS may still hold part of it in RAM, and flushing it can take a while.
        w0 = time.time()
        frames.flush()
        del frames
        if final_path.exists():
            final_path.unlink()
        tmp_path.replace(final_path)
        write_time = time.time() - w0
        frames = np.load(final_path, mmap_mode="r")
        files["frames"] = "frames.npy"
        log(f"wrote frames.npy ({frames.nbytes / mem.GIB:.2f} GiB): final flush to disk took "
            f"{write_time:.1f} s")
    return {"frames": frames, "final_frame": np.asarray(frames[-1]), "captures": cap,
            "run_time_s": run_time, "write_time_s": write_time, "files": files}


def _run_chunked(sim, T, sched, chunk, out_dir, log, captures=None):
    chunk = max(1, int(chunk))
    records = []
    pending = None
    last = None
    write_wait = 0.0
    caps = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for idx, s0 in enumerate(range(0, T, chunk)):
            s1 = min(s0 + chunk, T)
            c0 = time.time()
            n = tf.constant(s1 - s0, dtype=tf.int32)
            s = tf.convert_to_tensor(sched[s0:s1], dtype=tf.float32)
            local = [c - s0 for c in (captures or []) if s0 <= c < s1]
            if local:
                res = _np(sim.simulate_steps_capture(n, s, tf.convert_to_tensor(local, dtype=tf.int32)))
                arr = res.pop("frames")
                caps.append({k: res[k] for k in ("captured_screens", "captured_r0", "captured_b_scales")})
            else:
                arr = _np(sim.simulate_steps(n, s))
            compute = time.time() - c0
            last = arr[-1].copy()
            if pending is not None:
                w0 = time.time()
                pending.result()
                write_wait += time.time() - w0
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
            w0 = time.time()
            pending.result()
            write_wait += time.time() - w0
    run_time = time.time() - start
    files = {}
    if out_dir is not None:
        _write_json(Path(out_dir) / "frames_chunk_manifest.json",
                    {"total_steps": T, "chunk_size": chunk, "dtype": "complex64",
                     "frame_shape": [int(sim.nx_size)] * 2, "chunks": records})
        files["chunks"] = "frames_chunk_manifest.json"
        log(f"chunked: {len(records)} chunk files written; waiting on disk writes took {write_wait:.1f} s "
            "(writes overlap with the next chunk's compute)")
    # run_time includes the write waits here; report them separately.
    cap = None
    if caps:
        cap = {k: np.concatenate([c[k] for c in caps]) for k in caps[0]}
        cap["capture_indices"] = np.asarray(captures, dtype=np.int32)
    return {"frames": None, "final_frame": last, "captures": cap, "run_time_s": run_time - write_wait,
            "write_time_s": write_wait, "files": files, "chunks": records}


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
            "write_time_s": 0.0, "files": files}


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
    timing=None,
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
        (device, host and chunked modes; written to ``captures.npz``).
    state_in : path, optional
        Start from a checkpoint instead of the base state.
    state_out : path, optional
        Write the final state here so a later run can continue bit-exactly.
    log : callable or None
        Line sink for progress messages (default: print).
    timing : dict, optional
        Earlier phases to record in the manifest's ``timing`` block, e.g.
        ``{"startup_s": ..., "build_s": ...}`` (the CLI passes these).
    """
    log = _Log(log)
    t_run = time.time()
    T = int(num_steps if num_steps is not None else simulator.steps)
    if T < 1:
        raise ValueError("num_steps must be >= 1")
    L = int(simulator.num_layers)
    N = int(simulator.nx_size)
    live = hasattr(simulator, "simulate_steps_offload")
    warn_if_no_gpu(log)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        old = [p.name for p in sorted(out_dir.glob("frames*.npy")) + sorted(out_dir.glob("final_frame.npy"))]
        if old:
            shown = ", ".join(old[:3]) + (f" (+{len(old) - 3} more)" if len(old) > 3 else "")
            log(f"note: {out_dir} already holds a run; replacing {shown}")
        _clear_frame_outputs(out_dir)

    base_r0 = _np(simulator.get_state())["base_r0"]
    sched = build_schedule(schedule, T, L, base_r0=base_r0, kind=schedule_kind)
    requested = sorted({int(i) for i in (capture_indices or [])})
    captures = [i for i in requested if 0 <= i < T]
    if len(captures) < len(requested):
        log(f"note: capture ticks {[i for i in requested if not 0 <= i < T]} are outside 0..{T - 1} "
            "and are ignored")

    baseline = mem.measure_baseline_peak_bytes(simulator)
    estimate = mem.estimate_memory(T, N, baseline, requested_chunk_size=chunk_size)
    selected, chunk, reason = mem.select_mode(mode, estimate, live=live,
                                              has_output_dir=out_dir is not None)
    for line in mem.format_plan(estimate, mode, selected, chunk):
        log(line)
    if reason != "requested":
        log(f"guard: {reason}")
    ram = estimate["host_ram_available"]
    if selected == "host" and out_dir is not None and ram and estimate["full_output_bytes"] > 0.5 * ram:
        log(f"note: the {estimate['full_output_bytes'] / mem.GIB:.2f} GiB output is written through a "
            f"memory-mapped file and {ram / mem.GIB:.2f} GiB RAM is available; the OS writes it back to "
            "disk while running, so later steps can slow down and the final flush can take a while. "
            "--mode chunked writes bounded files instead.")
    t_plan = time.time() - t_run

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
                can_capture = hasattr(simulator, "simulate_steps_capture")
                if captures and not can_capture:
                    log("note: this model has no capture graph; captures are skipped")
                res = _run_chunked(simulator, T, sched, chunk_now, out_dir, log,
                                   captures=captures if can_capture else None)
            else:
                if captures:
                    log("note: no_save mode does not collect captures; use device, host or chunked mode")
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
    write_time = float(res.get("write_time_s", 0.0))
    end_step = int(_np(simulator.get_state())["global_step"])
    log(f"done: {T} steps simulated in {run_time:.2f} s | {T / max(run_time, 1e-9):.1f} steps/s | "
        f"peak GPU {mem.format_bytes(peak)}"
        + (f" | writing frames {write_time:.1f} s" if write_time >= 0.05 else ""))
    warnings_out = []
    if res["final_frame"] is not None and not np.all(np.isfinite(res["final_frame"])):
        warnings_out.append(
            "the final frame contains non-finite values: a layer's extrusion diverged, which happens when "
            "the outer scale is too large for the pixel scale. The frames of this run are not valid; lower "
            "channel_params.L0 or use a coarser channel_params.pixel_size.")
        log("WARNING: " + warnings_out[-1])

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

    if out_dir is not None and schedule is not None:
        np.save(out_dir / "schedule.npy", sched)
        files["schedule"] = "schedule.npy"

    config = getattr(simulator, "config", None)
    channel = simulator.describe() if hasattr(simulator, "describe") else None
    manifest = {
        "opal_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": getattr(simulator, "seed", None),
        "config": config,
        "channel": channel,
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
        "timing": None,
        "gpu": {"name": mem.gpu_name(), "total_bytes": estimate["gpu_total_bytes"],
                "baseline_peak_bytes": baseline, "run_peak_bytes": peak,
                "before": mem_before, "after": mem_after},
        "memory_estimate": estimate,
        "schedule": {"kind": schedule_kind,
                     "source": str(schedule) if isinstance(schedule, (str, os.PathLike)) else
                     ("array" if schedule is not None else None),
                     "constant": bool(np.all(sched == sched[0])),
                     "file": "schedule.npy (applied b_scale, [steps, layers])"
                     if out_dir is not None and schedule is not None else None},
        "capture_indices": captures,
        "state_in": str(state_in) if state_in is not None else None,
        "files": files,
        "warnings": warnings_out,
    }
    if "chunks" in res:
        manifest["chunks"] = res["chunks"]
    t_end = time.time()
    timing_block = {k: round(float(v), 3) for k, v in (timing or {}).items() if v is not None}
    timing_block.update(plan_s=round(t_plan, 3), simulate_s=round(run_time, 3),
                        write_frames_s=round(write_time, 3), run_call_s=round(t_end - t_run, 3))
    if _timing.started() is not None:
        timing_block["since_process_start_s"] = round(t_end - _timing.started(), 3)
    manifest["timing"] = timing_block
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
