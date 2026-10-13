"""GPU-resident phase-screen layer."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from . import _tfsetup  # noqa: F401
from .phase_screens import PhaseScreenVonKarman

__all__ = ["GpuLayer"]


class GpuLayer(tf.Module):
    """One turbulent layer: an infinite von Karman screen extruded on the GPU.

    The screen, the extrusion operators ``A``/``B``, the per-layer r0 and the
    layer's Philox RNG state are TensorFlow variables, so the whole layer can
    be checkpointed, restored, and exported in a SavedModel.

    Parameters
    ----------
    nx_size : int
        Screen size in pixels.
    pixel_scale : float
        Pixel pitch (m).
    r0 : float
        Layer Fried parameter (m).
    L0 : float
        Outer scale (m).
    random_seed : int, optional
        Seed of the initial screen and of the extrusion noise. ``None`` draws
        a random seed.
    """

    def __init__(
        self,
        nx_size: int,
        pixel_scale: float,
        r0: float,
        L0: float = 20.0,
        random_seed: int | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if random_seed is None:
            random_seed = int(np.random.default_rng().integers(1, 2**31 - 1))

        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.random_seed = int(random_seed)
        self._phase_screen_object = PhaseScreenVonKarman(
            nx_size, pixel_scale, r0, L0, random_seed=random_seed,
        )

        initial_screen = tf.cast(self._phase_screen_object.scrn, tf.float32)
        self.tf_scrn = tf.Variable(initial_screen, trainable=False, name="scrn")
        self.tf_base_scrn = tf.Variable(initial_screen, trainable=False, name="base_scrn")
        # r0 at which the base screen was drawn (screens scale exactly as r0^(-5/6))
        self.base_scrn_r0 = float(r0)

        self.tf_stencil_coords = tf.constant(
            self._phase_screen_object.stencil_coords,
            dtype=tf.int32,
        )
        self.tf_A_mat = tf.Variable(
            tf.cast(self._phase_screen_object.A_mat, tf.float32),
            trainable=False,
            name="A_mat",
        )
        self.tf_B_base = tf.Variable(
            tf.cast(self._phase_screen_object.B_mat, tf.float32),
            trainable=False,
            name="B_base",
        )
        # The float64 host copies are no longer needed once on the device.
        self._phase_screen_object.release_operators()

        self.tf_b_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="b_scale")
        self.tf_base_r0 = tf.Variable(float(r0), trainable=False, dtype=tf.float32, name="base_r0")
        self.tf_r0 = tf.Variable(float(r0), trainable=False, dtype=tf.float32, name="r0")
        self.tf_base_cn2_integrated = tf.Variable(
            tf.pow(tf.constant(float(r0), dtype=tf.float32), -5.0 / 3.0),
            trainable=False,
            name="base_cn2_integrated",
        )
        self.tf_cn2_integrated = tf.Variable(
            self.tf_base_cn2_integrated.read_value(),
            trainable=False,
            name="cn2_integrated",
        )

        self.tf_nx_size = tf.constant(self._phase_screen_object.nx_size, dtype=tf.int32)
        self.tf_stencil_length = self.tf_nx_size

        self.tf_rng = tf.random.Generator.from_seed(random_seed)
        self.tf_base_rng_state = tf.Variable(
            self.tf_rng.state.read_value(),
            trainable=False,
            name="base_rng_state",
        )

    @property
    def scrn(self):
        """Current phase screen as a NumPy array (radians)."""
        return self.tf_scrn.numpy()

    # ------------------------------------------------------------------
    # Soft (weighted) turbulence-strength control
    # ------------------------------------------------------------------
    @tf.function
    def set_b_scale(self, b_scale):
        """Scale the innovation term: ``Cn^2 -> b^2 Cn^2``, ``r0 -> r0 / b^(6/5)``."""
        b_scale = tf.cast(b_scale, tf.float32)
        b_scale = tf.maximum(b_scale, tf.constant(1e-12, dtype=tf.float32))
        self.tf_b_scale.assign(b_scale)
        self.tf_cn2_integrated.assign(self.tf_base_cn2_integrated * tf.square(b_scale))
        self.tf_r0.assign(self.tf_base_r0 / tf.pow(b_scale, 6.0 / 5.0))
        return self.tf_b_scale

    @tf.function
    def reset_to_base(self):
        """Restore the initial screen, RNG state and turbulence strength."""
        self.tf_scrn.assign(self.tf_base_scrn)
        self.tf_b_scale.assign(1.0)
        self.tf_r0.assign(self.tf_base_r0)
        self.tf_cn2_integrated.assign(self.tf_base_cn2_integrated)
        self.tf_rng.state.assign(self.tf_base_rng_state)
        return self.tf_scrn

    @tf.function
    def set_state(self, screen, rng_state, b_scale):
        self.tf_scrn.assign(tf.cast(screen, tf.float32))
        self.tf_rng.state.assign(tf.cast(rng_state, self.tf_rng.state.dtype))
        return self.set_b_scale(b_scale)

    # ------------------------------------------------------------------
    # Extrusion
    # ------------------------------------------------------------------
    @tf.function
    def get_new_row(self):
        random_data = self.tf_rng.normal(
            shape=[self.tf_nx_size],
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32,
        )
        stencil_data = tf.gather_nd(self.tf_scrn, self.tf_stencil_coords)
        new_row_part1 = tf.linalg.matvec(self.tf_A_mat, stencil_data)
        new_row_part2 = self.tf_b_scale * tf.linalg.matvec(self.tf_B_base, random_data)
        new_row = new_row_part1 + new_row_part2
        return tf.reshape(new_row, [1, self.tf_nx_size])

    @tf.function
    def add_row(self):
        new_row = self.get_new_row()
        updated_scrn = tf.concat([new_row, self.tf_scrn], axis=0)
        self.tf_scrn.assign(updated_scrn[:self.tf_stencil_length, :self.tf_nx_size])
        return self.tf_scrn

    # ------------------------------------------------------------------
    # Exact (hard-switch) turbulence-strength control
    # ------------------------------------------------------------------
    def replace_matrices_for_r0(self, new_r0, *, regenerate_scrn=False, screen_seed=None):
        """Rebuild the extrusion operators ``A``/``B`` for a new layer r0.

        Unlike :meth:`set_b_scale`, the new statistics are exact von Karman
        statistics at ``new_r0`` for every subsequent row. Runs on the host
        (covariance + SVD), so it is called between graph segments.

        Parameters
        ----------
        new_r0 : float
            New Fried parameter (m).
        regenerate_scrn : bool
            Also replace the current and base screens with a fresh screen at
            ``new_r0``. By default the current screen is kept so the
            wavefront stays continuous and relaxes to the new statistics as
            new rows are extruded.
        screen_seed : int, optional
            Seed of the regenerated screen (default: the layer's current
            screen seed), so regeneration is reproducible.
        """
        new_r0 = float(new_r0)
        if not np.isfinite(new_r0) or new_r0 <= 0.0:
            raise ValueError(f"replace_matrices_for_r0 needs new_r0 > 0, got {new_r0!r}")

        seed = self._phase_screen_object.random_seed if screen_seed is None else int(screen_seed)
        new_ps = PhaseScreenVonKarman(
            self._phase_screen_object.nx_size,
            self._phase_screen_object.pixel_scale,
            new_r0,
            self._phase_screen_object.L0,
            random_seed=seed,
            initial_screen=regenerate_scrn,
        )

        if new_ps.A_mat.shape != tuple(self.tf_A_mat.shape):
            raise RuntimeError(
                f"A_mat shape mismatch: existing {tuple(self.tf_A_mat.shape)}, new {new_ps.A_mat.shape}"
            )
        if new_ps.B_mat.shape != tuple(self.tf_B_base.shape):
            raise RuntimeError(
                f"B_mat shape mismatch: existing {tuple(self.tf_B_base.shape)}, new {new_ps.B_mat.shape}"
            )

        self.tf_A_mat.assign(tf.cast(new_ps.A_mat, tf.float32))
        self.tf_B_base.assign(tf.cast(new_ps.B_mat, tf.float32))

        new_r0_const = tf.constant(new_r0, dtype=tf.float32)
        self.tf_base_r0.assign(new_r0_const)
        self.tf_r0.assign(new_r0_const)
        self.tf_base_cn2_integrated.assign(tf.pow(new_r0_const, -5.0 / 3.0))
        self.tf_cn2_integrated.assign(self.tf_base_cn2_integrated)
        # Replacement owns turbulence strength; b_scale can still be applied on top.
        self.tf_b_scale.assign(tf.constant(1.0, dtype=tf.float32))

        if regenerate_scrn:
            new_scrn = tf.cast(new_ps.scrn, tf.float32)
            self.tf_scrn.assign(new_scrn)
            self.tf_base_scrn.assign(new_scrn)
            self.base_scrn_r0 = new_r0

        new_ps.release_operators()
        self._phase_screen_object = new_ps
        self.r0 = new_r0

        return self.tf_r0

    # ------------------------------------------------------------------
    # New realization without rebuilding the graph
    # ------------------------------------------------------------------
    def reseed(self, random_seed):
        """Draw a new initial screen and RNG stream from ``random_seed``.

        The result is identical to constructing a new layer with the same
        seed, provided the layer's operators still correspond to its current
        r0 (always true unless :meth:`replace_matrices_for_r0` was used).
        """
        random_seed = int(random_seed)
        ps = self._phase_screen_object
        ps.random_seed = random_seed
        ps.make_initial_screen()
        screen = tf.cast(ps.scrn, tf.float32)
        ps._scrn = None
        self.tf_scrn.assign(screen)
        self.tf_base_scrn.assign(screen)
        self.base_scrn_r0 = float(ps.r0)

        rng_state = tf.random.Generator.from_seed(random_seed).state.read_value()
        self.tf_rng.state.assign(rng_state)
        self.tf_base_rng_state.assign(rng_state)

        self.tf_b_scale.assign(1.0)
        self.tf_r0.assign(self.tf_base_r0)
        self.tf_cn2_integrated.assign(self.tf_base_cn2_integrated)
        self.random_seed = random_seed
