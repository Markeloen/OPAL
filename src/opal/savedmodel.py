"""SavedModel export and loading.

An exported model contains the layer state, the extrusion operators and the
propagator, and exposes the simulation graphs as signatures, so it can be
driven from any TensorFlow runtime without the Python package (e.g. TF
Serving, C++, or a training input pipeline). The host-streaming graph embeds
a Python callback and is therefore not exported; exported models support the
``device``, ``chunked`` and ``no_save`` modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import _tfsetup  # noqa: F401
from .simulator import saved_model_signatures

__all__ = ["export", "load", "LoadedSimulator"]

CONFIG_FILE = "opal_config.json"


def export(simulator, export_dir):
    """Export ``simulator`` in its base state to ``export_dir``.

    The config and seed are written next to the model as ``opal_config.json``.
    """
    export_dir = Path(export_dir)
    simulator.reset_to_base()
    tf.saved_model.save(simulator, str(export_dir), signatures=saved_model_signatures(simulator))
    meta = {
        "config": simulator.config,
        "seed": simulator.seed,
        "nx_size": simulator.nx_size,
        "num_layers": simulator.num_layers,
        "steps": simulator.steps,
        "base_r0": [float(np.asarray(r).item()) for r in simulator.geometry.r0_array],
    }
    (export_dir / CONFIG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return export_dir


def _out(result):
    if isinstance(result, dict):
        return result.get("output_0", next(iter(result.values())))
    return result


class LoadedSimulator:
    """Thin wrapper giving a loaded SavedModel the :class:`GpuSimulator` call API.

    Works with :func:`opal.run`, :func:`opal.save_state` and
    :func:`opal.load_state`.
    """

    def __init__(self, export_dir):
        self.export_dir = Path(export_dir)
        self.model = tf.saved_model.load(str(self.export_dir))
        self.signatures = self.model.signatures
        meta_path = self.export_dir / CONFIG_FILE
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        state = self.get_state()
        self.num_layers = int(state["phase_screens"].shape[0])
        self.nx_size = int(state["phase_screens"].shape[1])
        self.steps = int(self.meta.get("steps", 1000))
        self.seed = self.meta.get("seed")
        self.config = self.meta.get("config")

    def simulate_steps(self, num_steps, b_scale_schedule):
        return _out(self.signatures["simulate_steps"](
            num_steps=tf.cast(num_steps, tf.int32),
            b_scale_schedule=tf.cast(b_scale_schedule, tf.float32)))

    def simulate_steps_no_save(self, num_steps, b_scale_schedule):
        return _out(self.signatures["simulate_steps_no_save"](
            num_steps=tf.cast(num_steps, tf.int32),
            b_scale_schedule=tf.cast(b_scale_schedule, tf.float32)))

    def simulate_steps_capture(self, num_steps, b_scale_schedule, capture_indices):
        return self.signatures["simulate_steps_capture"](
            num_steps=tf.cast(num_steps, tf.int32),
            b_scale_schedule=tf.cast(b_scale_schedule, tf.float32),
            capture_indices=tf.cast(capture_indices, tf.int32))

    def simulate_steps_sparse_capture(self, num_steps, b_scale_schedule, capture_indices,
                                      wavefront_indices):
        return self.signatures["simulate_steps_sparse_capture"](
            num_steps=tf.cast(num_steps, tf.int32),
            b_scale_schedule=tf.cast(b_scale_schedule, tf.float32),
            capture_indices=tf.cast(capture_indices, tf.int32),
            wavefront_indices=tf.cast(wavefront_indices, tf.int32))

    def reset_to_base(self):
        return self.signatures["reset_to_base"]()

    def get_state(self):
        return self.signatures["get_state"]()

    def set_state(self, phase_screens, rng_states, global_step, b_scales):
        return self.signatures["set_state"](
            phase_screens=tf.cast(phase_screens, tf.float32),
            rng_states=tf.cast(rng_states, tf.int64),
            global_step=tf.cast(global_step, tf.int64),
            b_scales=tf.cast(b_scales, tf.float32))

    def set_b_scale_profile(self, b_scales):
        return _out(self.signatures["set_b_scale_profile"](b_scales=tf.cast(b_scales, tf.float32)))


def load(export_dir):
    """Load an exported simulator as a :class:`LoadedSimulator`."""
    return LoadedSimulator(export_dir)
