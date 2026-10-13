"""OPAL: GPU-accelerated multilayer atmospheric turbulence simulator for
space-to-ground optical links.

Quick start::

    import opal

    sim = opal.GpuSimulator(opal.load_config("configs/quickstart_256.json"))
    result = opal.run(sim, mode="auto", out_dir="runs/quickstart")
    frames = result.frames            # [T, N, N] complex64 received fields

The TensorFlow-backed names are imported lazily, so ``import opal`` and the
TF-free helpers (``opal.config``, ``opal.analysis``) stay light.
"""

from ._version import __version__
from .config import load_config, save_config

_LAZY = {
    "GpuSimulator": ("simulator", "GpuSimulator"),
    "run": ("runners", "run"),
    "RunResult": ("runners", "RunResult"),
    "save_state": ("runners", "save_state"),
    "load_state": ("runners", "load_state"),
    "build_schedule": ("runners", "build_schedule"),
    "load_frames": ("frames", "load_frames"),
    "iter_frames": ("frames", "iter_frames"),
    "open_frames": ("frames", "open_frames"),
    "FrameSeries": ("frames", "FrameSeries"),
    "export": ("savedmodel", "export"),
    "load": ("savedmodel", "load"),
    "run_pass": ("passes", "run_pass"),
    "plan_pass": ("passes", "plan_pass"),
    "generate_dataset": ("dataset", "generate_dataset"),
    "load_dataset": ("dataset", "load_dataset"),
    "iter_samples": ("dataset", "iter_samples"),
    "as_tf_dataset": ("dataset", "as_tf_dataset"),
    "analyze_run": ("analysis.runs", "analyze_run"),
    "load_analysis": ("analysis.runs", "load_analysis"),
}

_SUBMODULES = {"analysis", "config", "dataset", "frames", "geometry", "layer", "memory", "passes",
               "phase_screens", "profiles", "propagation", "runners", "savedmodel", "simulator", "viewer"}

__all__ = ["__version__", "load_config", "save_config", *_LAZY]


def __getattr__(name):
    import importlib

    if name in _LAZY:
        module, attr = _LAZY[name]
        value = getattr(importlib.import_module(f".{module}", __name__), attr)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'opal' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
