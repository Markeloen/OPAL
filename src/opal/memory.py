"""GPU memory measurement and the automatic run-mode guard.

The guard measures the simulator's own static footprint (layer screens,
extrusion operators, propagator constants and FFT workspace) by running a
single step, then predicts each run mode's peak memory:

=========  ====================================================
mode       predicted GPU peak
=========  ====================================================
device     baseline + 2 x (T x frame) + margin
host       baseline + 4 x frame + margin          (flat in T)
chunked    baseline + 2 x (chunk x frame) + margin
no_save    baseline + margin
=========  ====================================================

with ``frame = N^2 x 8`` bytes (complex64) and a safety margin of
max(1.5 GiB, 20 % of the card).
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import tensorflow as tf

from . import _tfsetup  # noqa: F401

__all__ = [
    "GIB",
    "MIB",
    "COMPLEX64_BYTES",
    "DEFAULT_CHUNK_SIZE",
    "MODES",
    "setup_tensorflow",
    "gpu_name",
    "gpu_total_memory_bytes",
    "host_available_bytes",
    "memory_info",
    "memory_sample",
    "reset_memory_stats",
    "measure_baseline_peak_bytes",
    "estimate_memory",
    "select_mode",
    "format_plan",
]

GIB = 1024**3
MIB = 1024**2
COMPLEX64_BYTES = 8
DEFAULT_CHUNK_SIZE = 100
MAX_CHUNK_FILE_BYTES = 800 * MIB
HOST_RAM_FRACTION = 0.6
MODES = ("auto", "device", "host", "chunked", "no_save")


def setup_tensorflow(require_gpu=True):
    """Enable memory growth and return the visible GPUs.

    Raises ``RuntimeError`` if ``require_gpu`` and no GPU is visible.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError(
            "TensorFlow did not detect a GPU. OPAL needs a CUDA-capable GPU; "
            "see the installation notes in the README."
        )
    return gpus


def gpu_name():
    try:
        details = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0])
        return details.get("device_name", "GPU:0")
    except Exception:
        return "GPU:0"


def gpu_total_memory_bytes():
    """Total memory of the first GPU (GPUtil, then nvidia-smi); 0 if unknown."""
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            return int(gpus[0].memoryTotal * MIB)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(float(result.stdout.strip().splitlines()[0].strip()) * MIB)
    except Exception:
        return 0


def host_available_bytes():
    """Available host RAM in bytes (0 if psutil is missing)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def memory_info():
    """TensorFlow allocator statistics for GPU:0 in bytes: ``{"current", "peak"}``."""
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {"current": int(info["current"]), "peak": int(info["peak"])}
    except Exception:
        return {"current": 0, "peak": 0}


def memory_sample():
    info = memory_info()
    return {
        "current_bytes": info["current"],
        "peak_bytes": info["peak"],
        "current_gib": info["current"] / GIB,
        "peak_gib": info["peak"] / GIB,
    }


def reset_memory_stats():
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def measure_baseline_peak_bytes(simulator):
    """Peak GPU bytes of a single no-save step (the simulator's static footprint).

    Resets the simulator to its base state before and after.
    """
    simulator.reset_to_base()
    reset_memory_stats()
    schedule = tf.ones([1, int(simulator.num_layers)], dtype=tf.float32)
    out = simulator.simulate_steps_no_save(tf.constant(1, dtype=tf.int32), schedule)
    np.asarray(out)
    info = memory_info()
    baseline = max(info["current"], info["peak"])
    simulator.reset_to_base()
    reset_memory_stats()
    return baseline


def estimate_memory(
    num_steps,
    nx_size,
    baseline_peak_bytes,
    gpu_total_bytes=None,
    requested_chunk_size=DEFAULT_CHUNK_SIZE,
    host_available=None,
):
    """Predict the GPU (and host) memory needed by every run mode."""
    if gpu_total_bytes is None:
        gpu_total_bytes = gpu_total_memory_bytes()
    if host_available is None:
        host_available = host_available_bytes()

    frame_bytes = int(nx_size) * int(nx_size) * COMPLEX64_BYTES
    full_output_bytes = int(num_steps) * frame_bytes
    safety_margin = (
        max(int(1.5 * GIB), int(0.20 * gpu_total_bytes)) if gpu_total_bytes else int(1.5 * GIB)
    )

    device_required = baseline_peak_bytes + 2 * full_output_bytes + safety_margin
    host_required = baseline_peak_bytes + 4 * frame_bytes + safety_margin

    requested_chunk_size = max(1, int(requested_chunk_size))
    if gpu_total_bytes:
        numerator = gpu_total_bytes - baseline_peak_bytes - safety_margin
        max_chunk_steps = math.floor(numerator / (2.0 * frame_bytes)) if numerator > 0 else 0
    else:
        max_chunk_steps = requested_chunk_size
    max_io_chunk_steps = max(1, math.floor(MAX_CHUNK_FILE_BYTES / frame_bytes))
    chunk_steps = (
        min(requested_chunk_size, max_chunk_steps, max_io_chunk_steps, int(num_steps))
        if max_chunk_steps >= 1
        else 0
    )
    chunk_required = baseline_peak_bytes + 2 * chunk_steps * frame_bytes + safety_margin

    host_ram_budget = int(HOST_RAM_FRACTION * host_available) if host_available else 0

    def fits(required):
        return bool(gpu_total_bytes) and required < gpu_total_bytes

    return {
        "num_steps": int(num_steps),
        "nx_size": int(nx_size),
        "frame_bytes": frame_bytes,
        "full_output_bytes": full_output_bytes,
        "baseline_peak_bytes": int(baseline_peak_bytes),
        "gpu_total_bytes": int(gpu_total_bytes),
        "safety_margin": safety_margin,
        "device_required": device_required,
        "device_safe": fits(device_required),
        "host_required": host_required,
        "host_safe": fits(host_required),
        "requested_chunk_size": requested_chunk_size,
        "max_chunk_steps": int(max_chunk_steps),
        "max_io_chunk_steps": int(max_io_chunk_steps),
        "chunk_steps": int(chunk_steps),
        "chunk_required": chunk_required,
        "chunk_safe": bool(chunk_steps >= 1 and (not gpu_total_bytes or chunk_required < gpu_total_bytes)),
        "host_ram_available": int(host_available),
        "host_ram_budget": host_ram_budget,
        "output_fits_ram": bool(host_ram_budget and full_output_bytes <= host_ram_budget),
    }


def select_mode(requested, estimate, *, live=True, has_output_dir=False):
    """Pick the run mode for ``requested`` given a memory ``estimate``.

    ``auto`` prefers, in order: ``device`` (fastest; output must fit on the
    GPU and in RAM), ``host`` (flat GPU memory; needs RAM or an output
    directory for the memory-mapped file), ``chunked`` (bounded GPU memory,
    streamed to disk) and finally ``no_save``. An explicit request is honoured
    when it is safe and otherwise downgraded along the same chain.

    Returns
    -------
    (mode, chunk_size, reason)
    """
    if requested not in MODES:
        raise ValueError(f"Unknown mode {requested!r}; choose from {MODES}")

    fits_ram = estimate["output_fits_ram"]
    host_ok = live and estimate["host_safe"] and (has_output_dir or fits_ram)
    device_ok = estimate["device_safe"] and fits_ram
    chunk_ok = estimate["chunk_safe"]

    if requested == "no_save":
        return "no_save", 0, "requested"
    if requested == "device" and device_ok:
        return "device", 0, "requested"
    if requested == "host" and host_ok:
        return "host", 0, "requested"
    if requested == "chunked" and chunk_ok:
        return "chunked", estimate["chunk_steps"], "requested"

    chain = {
        "auto": ("device", "host", "chunked"),
        "device": ("host", "chunked"),
        "host": ("device", "chunked"),
        "chunked": (),
    }[requested]
    for mode in chain:
        if mode == "device" and device_ok:
            return "device", 0, "auto: full output fits on the GPU"
        if mode == "host" and host_ok:
            return "host", 0, "auto: streaming frames to host keeps GPU memory flat"
        if mode == "chunked" and chunk_ok and has_output_dir:
            return "chunked", estimate["chunk_steps"], "auto: output exceeds host RAM, streaming chunks to disk"
    return "no_save", 0, "no safe mode can keep the frames; running without saving"


def format_plan(estimate, requested, selected, chunk_size=0):
    """Human-readable memory plan (list of lines)."""
    e = estimate
    g = lambda b: f"{b / GIB:.2f} GiB"  # noqa: E731
    lines = [
        f"memory plan: N={e['nx_size']} steps={e['num_steps']} requested={requested} -> {selected}"
        + (f" (chunk={chunk_size})" if selected == "chunked" else ""),
        f"  frame {e['frame_bytes'] / MIB:.1f} MiB | output {g(e['full_output_bytes'])} | "
        f"baseline {g(e['baseline_peak_bytes'])} | margin {g(e['safety_margin'])} | "
        f"GPU {g(e['gpu_total_bytes'])} | RAM avail {g(e['host_ram_available'])}",
        f"  predicted GPU peak: device {g(e['device_required'])}"
        f"{'' if e['device_safe'] else ' (too big)'} | host {g(e['host_required'])} | "
        f"chunked {g(e['chunk_required'])} (chunk<={e['chunk_steps']})",
    ]
    return lines
