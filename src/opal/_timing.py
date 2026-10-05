"""Process clock shared by the CLI and the ``[opal  X s]`` log lines.

The CLI marks the start of the process, so every timestamp it prints is the
time since the command was launched (Python and TensorFlow start-up and the
simulator build included) and matches an external stopwatch. Library calls
made without the CLI keep a per-call clock.
"""

from __future__ import annotations

import time

_start = None


def process_start_time():
    """Creation time of this process (falls back to "now")."""
    try:
        import psutil

        return float(psutil.Process().create_time())
    except Exception:
        return time.time()


def mark_start(t=None):
    """Use ``t`` (default: the process creation time) as time zero."""
    global _start
    _start = process_start_time() if t is None else float(t)
    return _start


def started():
    """Time zero set by :func:`mark_start`, or ``None``."""
    return _start


def clock(t0=None):
    """``t0`` if given, else the process clock if set, else now."""
    if t0 is not None:
        return float(t0)
    return _start if _start is not None else time.time()


def layer_progress(say, nx_size, min_size=512):
    """``GpuSimulator(progress=...)`` callback that reports each layer build.

    Only used for grids of at least ``min_size`` where building a layer takes
    seconds; returns ``None`` otherwise.
    """
    if say is None or int(nx_size) < min_size:
        return None

    def report(i, n, r0):
        say(f"  layer {i + 1:2d}/{n}: r0 {r0:.3f} m (covariance, Cholesky, SVD on the CPU)")

    return report
