"""The OPAL GPU simulator.

:class:`GpuSimulator` owns the layer stack and the propagator and runs the
whole time loop inside TensorFlow graphs: every tick it applies the per-layer
turbulence-strength profile, propagates the field through all layers, and
advances each layer by its wind-driven number of rows. All mutable state (phase
screens, RNG states, turbulence scales, step counter) lives in ``tf.Variable``
objects, so runs can be checkpointed, resumed bit-exactly, and exported as a
SavedModel.

Entry points
------------
``simulate_steps``                  all frames kept on the GPU (device mode)
``simulate_steps_capture``          + layer-state captures at chosen ticks
``simulate_steps_sparse_capture``   only selected wavefronts / screens
``simulate_steps_no_save``          final frame only (benchmarks, in-graph metrics)
``simulate_steps_offload``          every frame streamed to host RAM (host mode)
``simulate_steps_sparse_capture_host``  sparse captures streamed to a CPU buffer
``simulate_pass_replacement``       multi-segment pass with exact r0 switches
"""

import json
import logging
import os
import time

import numpy as np
import tensorflow as tf

from .config import normalize_config, to_plain
from .geometry import GPUGeometry, derive_layer_seeds
from .propagation import Propagator, tf_rect

__all__ = ["GpuSimulator", "saved_model_signatures"]


class _HostOffloadRuntime:
    """Plain (non-trackable) holder for the host-offload graph and frame sink.

    Deliberately kept invisible to tf.Module tracking: the offload graph
    embeds a tf.py_function, which cannot be serialized into a SavedModel,
    and the Python-side sink must never become a checkpoint dependency.
    """

    def __init__(self):
        self.graph_fn = None
        self.sink = None

    def dispatch(self, step, frame):
        self.sink(int(step), frame.numpy())
        return step


class _HostCaptureRuntime:
    """Plain (non-trackable) holder for the host-capture graph and CPU buffer.

    Like _HostOffloadRuntime, deliberately invisible to tf.Module tracking:
    the wavefront buffer can reach multiple GB (capacity x nx^2 x 8 B) and
    must never become a checkpoint/SavedModel dependency. Capacity growth
    swaps in a new Variable, which forces a retrace -- the graph_fn is
    rebuilt alongside the buffer so callers never see stale captures.
    """

    def __init__(self):
        self.graph_fn = None
        self.wavefront_buffer = None  # tf.Variable [capacity, nx, nx] complex64 on /CPU:0
        self.capacity = 0


class GpuSimulator(tf.Module):
    """Multilayer GPU turbulence simulator.

    Parameters
    ----------
    config : dict
        Configuration (see :func:`opal.load_config`). Plain values and
        ``{"value": ...}`` entries are both accepted.
    name : str, optional
        ``tf.Module`` name.
    seed : int, optional
        Master seed. Falls back to ``simulation_params.seed`` and then to a
        random seed; the seed in use is always available as ``self.seed``.
    log_dir : str or os.PathLike, optional
        If given, the layer geometry and diagnostics are logged to
        ``<log_dir>/opal.log``.
    """

    def __init__(self, config, name=None, seed=None, log_dir=None):
        super().__init__(name=name)

        config = normalize_config(config)
        self._config_json = json.dumps(to_plain(config))
        channel_params = config["channel_params"]
        simulation_params = config["simulation_params"]
        if seed is None:
            seed = simulation_params["seed"]["value"]
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))

        self.steps = int(simulation_params["num_steps"]["value"])
        self.nx_size = int(channel_params["phase_screen_size"]["value"])
        self.px_delta = float(channel_params["pixel_size"]["value"])
        self.total_r0 = float(channel_params["total_r0"]["value"])
        self.num_layers = int(channel_params["num_layers"]["value"])
        self.per_tick_simulation = float(simulation_params["time_per_tick"]["value"])
        self.simulation_time = self.steps * self.per_tick_simulation
        self.seed = int(seed)

        self.geometry = GPUGeometry(
            nx_size=self.nx_size,
            pixel_scale=self.px_delta,
            wvl=float(channel_params["wavefront_wavelength"]["value"]),
            r0=self.total_r0,
            number_of_layers=self.num_layers,
            whole_simulation_time=self.simulation_time,
            per_tick_simulation=self.per_tick_simulation,
            satellite_orbit=float(channel_params["satellite_orbit"]["value"]),
            ground_wind_speed=float(channel_params["ground_wind_speed"]["value"]),
            satellite_zenith_angle_starting=float(channel_params["satellite_zenith"]["value"]),
            L0=float(channel_params["L0"]["value"]),
            seed=self.seed,
        )

        self.log_dir = None if log_dir is None else os.fspath(log_dir)
        self.logger = self._setup_logger()
        self.geometry.show_object_info(self.logger)

        self.propagator = Propagator(
            self.nx_size,
            self.geometry.wvl,
            self.px_delta,
            self.px_delta,
            self.geometry.layer_height_array,
        )

        self.tf_layer_object_arr = self.geometry.layer_object_array
        self.tf_number_of_layers = tf.constant(self.num_layers, dtype=tf.int32)
        self.tf_current_b_scales = tf.Variable(
            tf.ones([self.num_layers], dtype=tf.float32),
            trainable=False,
            name="current_b_scales",
        )
        self.tf_base_r0_array = tf.constant(
            np.asarray(self.geometry.r0_array, dtype=np.float32),
            dtype=tf.float32,
        )
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")

        self._offload_runtime = _HostOffloadRuntime()
        self._host_capture_runtime = _HostCaptureRuntime()

        first_extr = self.geometry.number_extrusions_array_per_tick[0]
        extrusion_mult = np.ceil(1 / first_extr) if first_extr < 1 else np.ceil(first_extr)
        self.tf_extr_per_tick = tf.constant(
            np.ceil(
                extrusion_mult
                * np.asarray(self.geometry.number_extrusions_array_per_tick, dtype=np.float32)
            ),
            dtype=tf.float32,
        )

    def _setup_logger(self):
        logger = logging.getLogger(f"opal.simulator.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            if self.log_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)
                handler = logging.FileHandler(os.path.join(self.log_dir, "opal.log"))
                handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            else:
                handler = logging.NullHandler()
            logger.addHandler(handler)
        return logger

    def close_logger(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    @property
    def config(self):
        """The plain-valued configuration this simulator was built from."""
        return json.loads(self._config_json)

    def describe(self):
        """Summary of the channel: grid, seeds and the per-layer profile."""
        return {
            "seed": self.seed,
            "grid": self.nx_size,
            "pixel_size_m": self.px_delta,
            "aperture_m": self.nx_size * self.px_delta,
            "total_r0_m": self.total_r0,
            "D_over_r0": self.nx_size * self.px_delta / self.total_r0,
            "num_layers": self.num_layers,
            "time_per_tick_s": self.per_tick_simulation,
            "rows_per_tick": self.tf_extr_per_tick.numpy().astype(int).tolist(),
            "layers": self.geometry.describe(),
        }

    def reseed(self, seed=None):
        """Start a new independent realization without rebuilding any graph.

        The result is identical to constructing a new simulator with the same
        config and ``seed``, but the traced graphs and extrusion operators are
        reused, which makes ensemble and dataset generation cheap. Layers whose
        operators were replaced during a pass are first restored to their base
        r0. Returns the seed in use.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        seed = int(seed)
        layer_seeds = derive_layer_seeds(seed, self.num_layers)
        for layer, base_r0, layer_seed in zip(
            self.tf_layer_object_arr, self.geometry.r0_array, layer_seeds
        ):
            base_r0 = float(np.asarray(base_r0).item())
            if float(layer.r0) != base_r0:
                layer.replace_matrices_for_r0(base_r0)
            layer.reseed(layer_seed)
        self.geometry.seed = seed
        self.geometry.layer_seeds = layer_seeds
        self.tf_current_b_scales.assign(tf.ones([self.num_layers], dtype=tf.float32))
        self.global_step.assign(0)
        self.seed = seed
        return seed

    @tf.function
    def _make_input_field(self):
        x = tf.cast(tf.range(-self.nx_size // 2, self.nx_size // 2), tf.float32) * self.px_delta
        x1, y1 = tf.meshgrid(x, x)
        return tf_rect(x1, self.nx_size * self.px_delta) * tf_rect(
            y1, self.nx_size * self.px_delta
        )

    @tf.function
    def _phase_screen_stack(self):
        return tf.stack([layer.tf_scrn for layer in self.tf_layer_object_arr], axis=0)

    @tf.function
    def _rng_state_stack(self):
        return tf.stack([layer.tf_rng.state.read_value() for layer in self.tf_layer_object_arr], axis=0)

    @tf.function
    def _r0_stack(self):
        return tf.stack([layer.tf_r0.read_value() for layer in self.tf_layer_object_arr], axis=0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def set_b_scale_profile(self, b_scales):
        b_scales = tf.cast(b_scales, tf.float32)
        tf.debugging.assert_equal(tf.shape(b_scales)[0], self.tf_number_of_layers)
        for layer_idx in range(self.num_layers):
            self.tf_layer_object_arr[layer_idx].set_b_scale(b_scales[layer_idx])
        self.tf_current_b_scales.assign(b_scales)
        return self.tf_current_b_scales

    @tf.function(input_signature=[])
    def reset_to_base(self):
        for layer in self.tf_layer_object_arr:
            layer.reset_to_base()
        self.tf_current_b_scales.assign(tf.ones_like(self.tf_current_b_scales))
        self.global_step.assign(0)
        return self.get_state()

    @tf.function(input_signature=[])
    def get_state(self):
        return {
            "phase_screens": self._phase_screen_stack(),
            "rng_states": self._rng_state_stack(),
            "global_step": self.global_step.read_value(),
            "b_scales": self.tf_current_b_scales.read_value(),
            "r0": self._r0_stack(),
            "base_r0": self.tf_base_r0_array,
        }

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="phase_screens"),
            tf.TensorSpec(shape=[None, None], dtype=tf.int64, name="rng_states"),
            tf.TensorSpec(shape=(), dtype=tf.int64, name="global_step"),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name="b_scales"),
        ]
    )
    def set_state(self, phase_screens, rng_states, global_step, b_scales):
        tf.debugging.assert_equal(tf.shape(phase_screens)[0], self.tf_number_of_layers)
        tf.debugging.assert_equal(tf.shape(rng_states)[0], self.tf_number_of_layers)
        tf.debugging.assert_equal(tf.shape(b_scales)[0], self.tf_number_of_layers)
        for layer_idx in range(self.num_layers):
            self.tf_layer_object_arr[layer_idx].set_state(
                phase_screens[layer_idx],
                rng_states[layer_idx],
                b_scales[layer_idx],
            )
        self.tf_current_b_scales.assign(tf.cast(b_scales, tf.float32))
        self.global_step.assign(tf.cast(global_step, tf.int64))
        return self.get_state()

    @tf.function
    def _schedule_profile_for_step(self, b_scale_schedule, step):
        rows = tf.shape(b_scale_schedule)[0]
        tf.debugging.assert_greater(rows, 0)
        schedule_idx = tf.cond(
            tf.equal(rows, 1),
            lambda: tf.constant(0, dtype=tf.int32),
            lambda: tf.minimum(step, rows - 1),
        )
        return b_scale_schedule[schedule_idx]

    @tf.function
    def _tf_move_one_tick_all_layers(self):
        ps_arr = []
        for layer_idx in range(self.num_layers):
            ext_lvl = tf.cast(self.tf_extr_per_tick[layer_idx], tf.int32)
            for _ in tf.range(ext_lvl):
                self.tf_layer_object_arr[layer_idx].add_row()
            ps_arr.append(self.tf_layer_object_arr[layer_idx].tf_scrn)
        return ps_arr

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
        ]
    )
    def simulate_steps(self, num_steps, b_scale_schedule):
        """Run a GPU-resident simulation and return complex Uout frames."""
        tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
        Uin = self._make_input_field()
        ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]
        frames = tf.TensorArray(dtype=tf.complex64, size=num_steps)

        for step in tf.range(num_steps):
            profile = self._schedule_profile_for_step(b_scale_schedule, step)
            self.set_b_scale_profile(profile)
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
            frames = frames.write(step, Uout)
            ps_arr = self._tf_move_one_tick_all_layers()
            self.global_step.assign_add(1)

        return frames.stack()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="capture_indices"),
        ]
    )
    def simulate_steps_capture(self, num_steps, b_scale_schedule, capture_indices):
        """Run simulation and capture layer state at requested step indices."""
        tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
        Uin = self._make_input_field()
        ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]
        frames = tf.TensorArray(dtype=tf.complex64, size=num_steps)

        num_captures = tf.shape(capture_indices)[0]
        captured_screens = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers, self.nx_size, self.nx_size]),
        )
        captured_b_scales = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers]),
        )
        captured_r0 = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers]),
        )
        capture_pos = tf.constant(0, dtype=tf.int32)

        for step in tf.range(num_steps):
            profile = self._schedule_profile_for_step(b_scale_schedule, step)
            self.set_b_scale_profile(profile)
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
            frames = frames.write(step, Uout)

            next_capture = tf.cond(
                capture_pos < num_captures,
                lambda: capture_indices[capture_pos],
                lambda: tf.constant(-1, dtype=tf.int32),
            )
            if tf.equal(step, next_capture):
                captured_screens = captured_screens.write(capture_pos, self._phase_screen_stack())
                captured_b_scales = captured_b_scales.write(
                    capture_pos, self.tf_current_b_scales.read_value()
                )
                captured_r0 = captured_r0.write(capture_pos, self._r0_stack())
                capture_pos += 1

            ps_arr = self._tf_move_one_tick_all_layers()
            self.global_step.assign_add(1)

        return {
            "frames": frames.stack(),
            "captured_screens": captured_screens.stack(),
            "captured_b_scales": captured_b_scales.stack(),
            "captured_r0": captured_r0.stack(),
            "capture_indices": capture_indices,
            "final_state_phase_screens": self._phase_screen_stack(),
            "final_b_scales": self.tf_current_b_scales.read_value(),
            "global_step": self.global_step.read_value(),
        }

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="capture_indices"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="wavefront_indices"),
        ]
    )
    def simulate_steps_sparse_capture(
        self,
        num_steps,
        b_scale_schedule,
        capture_indices,
        wavefront_indices,
    ):
        """Run simulation while storing only requested wavefronts and layer screens."""
        tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
        Uin = self._make_input_field()
        ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]

        num_captures = tf.shape(capture_indices)[0]
        captured_screens = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers, self.nx_size, self.nx_size]),
        )
        captured_b_scales = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers]),
        )
        captured_r0 = tf.TensorArray(
            dtype=tf.float32,
            size=num_captures,
            element_shape=tf.TensorShape([self.num_layers]),
        )

        num_wavefronts = tf.shape(wavefront_indices)[0]
        captured_wavefronts = tf.TensorArray(
            dtype=tf.complex64,
            size=num_wavefronts,
            element_shape=tf.TensorShape([self.nx_size, self.nx_size]),
        )

        capture_pos = tf.constant(0, dtype=tf.int32)
        wavefront_pos = tf.constant(0, dtype=tf.int32)
        Uout = tf.zeros([self.nx_size, self.nx_size], dtype=tf.complex64)

        for step in tf.range(num_steps):
            profile = self._schedule_profile_for_step(b_scale_schedule, step)
            self.set_b_scale_profile(profile)
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)

            next_wavefront_capture = tf.cond(
                wavefront_pos < num_wavefronts,
                lambda: wavefront_indices[wavefront_pos],
                lambda: tf.constant(-1, dtype=tf.int32),
            )
            if tf.equal(step, next_wavefront_capture):
                captured_wavefronts = captured_wavefronts.write(wavefront_pos, Uout)
                wavefront_pos += 1

            next_screen_capture = tf.cond(
                capture_pos < num_captures,
                lambda: capture_indices[capture_pos],
                lambda: tf.constant(-1, dtype=tf.int32),
            )
            if tf.equal(step, next_screen_capture):
                captured_screens = captured_screens.write(capture_pos, self._phase_screen_stack())
                captured_b_scales = captured_b_scales.write(
                    capture_pos, self.tf_current_b_scales.read_value()
                )
                captured_r0 = captured_r0.write(capture_pos, self._r0_stack())
                capture_pos += 1

            ps_arr = self._tf_move_one_tick_all_layers()
            self.global_step.assign_add(1)

        return {
            "final_frame": Uout,
            "captured_wavefronts": captured_wavefronts.stack(),
            "wavefront_indices": wavefront_indices,
            "captured_screens": captured_screens.stack(),
            "captured_b_scales": captured_b_scales.stack(),
            "captured_r0": captured_r0.stack(),
            "capture_indices": capture_indices,
            "final_state_phase_screens": self._phase_screen_stack(),
            "final_b_scales": self.tf_current_b_scales.read_value(),
            "global_step": self.global_step.read_value(),
        }

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
        ]
    )
    def simulate_steps_no_save(self, num_steps, b_scale_schedule):
        tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
        Uin = self._make_input_field()
        ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]
        Uout = tf.zeros([self.nx_size, self.nx_size], dtype=tf.complex64)

        for step in tf.range(num_steps):
            profile = self._schedule_profile_for_step(b_scale_schedule, step)
            self.set_b_scale_profile(profile)
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
            ps_arr = self._tf_move_one_tick_all_layers()
            self.global_step.assign_add(1)

        return Uout

    # ------------------------------------------------------------------
    # Host-offloaded frame streaming
    # ------------------------------------------------------------------
    # simulate_steps/simulate_steps_capture accumulate every Uout frame in a
    # TensorArray and .stack() it, so the GPU holds ~2 * num_steps frames at
    # peak. The tf.py_function sink below copies each frame off the GPU as it
    # is produced AND acts as a per-step host sync point; without that sync
    # the executor enqueues all iterations ahead of the GPU stream and the
    # Uout buffers cannot be recycled. py_function is not serializable, so
    # this graph lives outside the tracked object graph and only exists on
    # live GpuSimulator instances, never in SavedModel exports.
    def _build_offload_graph(self):
        runtime = self._offload_runtime

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
                tf.TensorSpec(shape=[None], dtype=tf.int32, name="capture_indices"),
            ]
        )
        def offload_steps(num_steps, b_scale_schedule, capture_indices):
            tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
            Uin = self._make_input_field()
            ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]

            num_captures = tf.shape(capture_indices)[0]
            captured_screens = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers, self.nx_size, self.nx_size]),
            )
            captured_b_scales = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers]),
            )
            captured_r0 = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers]),
            )
            capture_pos = tf.constant(0, dtype=tf.int32)

            for step in tf.range(num_steps):
                profile = self._schedule_profile_for_step(b_scale_schedule, step)
                self.set_b_scale_profile(profile)
                Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
                tf.py_function(func=runtime.dispatch, inp=[step, Uout], Tout=tf.int32)

                next_capture = tf.cond(
                    capture_pos < num_captures,
                    lambda: capture_indices[capture_pos],
                    lambda: tf.constant(-1, dtype=tf.int32),
                )
                if tf.equal(step, next_capture):
                    captured_screens = captured_screens.write(
                        capture_pos, self._phase_screen_stack()
                    )
                    captured_b_scales = captured_b_scales.write(
                        capture_pos, self.tf_current_b_scales.read_value()
                    )
                    captured_r0 = captured_r0.write(capture_pos, self._r0_stack())
                    capture_pos += 1

                ps_arr = self._tf_move_one_tick_all_layers()
                self.global_step.assign_add(1)

            return {
                "captured_screens": captured_screens.stack(),
                "captured_b_scales": captured_b_scales.stack(),
                "captured_r0": captured_r0.stack(),
                "capture_indices": capture_indices,
                "final_state_phase_screens": self._phase_screen_stack(),
                "final_b_scales": self.tf_current_b_scales.read_value(),
                "global_step": self.global_step.read_value(),
            }

        return offload_steps

    def simulate_steps_offload(
        self,
        num_steps,
        b_scale_schedule,
        capture_indices=None,
        frame_callback=None,
    ):
        """Run the b_scale-scheduled simulation with frames streamed to host RAM.

        Same physics and state evolution as `simulate_steps_capture`, but each
        Uout frame leaves the GPU as soon as it is produced, so GPU memory
        stays flat at a few frames regardless of num_steps. Frames go to
        `frame_callback(step, frame)` (numpy complex64 [nx, nx]; copy it if
        you keep it) or, without a callback, into one [num_steps, nx, nx]
        array returned under "frames" (None when a callback is used).
        """
        num_steps = int(num_steps)
        schedule = tf.convert_to_tensor(
            np.asarray(b_scale_schedule, dtype=np.float32)
        )
        captures = np.asarray(
            [] if capture_indices is None else capture_indices, dtype=np.int32
        ).reshape(-1)

        if self._offload_runtime.graph_fn is None:
            self._offload_runtime.graph_fn = self._build_offload_graph()

        frames = None
        if frame_callback is None:
            frames = np.empty(
                (num_steps, self.nx_size, self.nx_size), dtype=np.complex64
            )

            def frame_callback(step, frame):
                frames[step] = frame

        self._offload_runtime.sink = frame_callback
        try:
            result = self._offload_runtime.graph_fn(
                tf.constant(num_steps, dtype=tf.int32),
                schedule,
                tf.convert_to_tensor(captures, dtype=tf.int32),
            )
        finally:
            self._offload_runtime.sink = None

        out = {key: value.numpy() for key, value in result.items()}
        out["frames"] = frames
        return out

    # ------------------------------------------------------------------
    # Host-captured sparse simulation (CPU tf.Variable offload)
    # ------------------------------------------------------------------
    # simulate_steps_sparse_capture returns captured wavefronts through
    # TensorArray.stack(), which materializes a contiguous [n, nx, nx]
    # complex64 tensor on the GPU *on top of* the TensorArray elements --
    # the dominant allocation for dense captures and the OOM site for
    # segment_steps >= 500 at res 1024. The graph below
    # scatters each captured frame straight into a CPU-resident
    # tf.Variable instead, so GPU memory stays flat regardless of how
    # many ticks are captured. Like the offload graph, buffer and graph
    # live outside the tracked object graph and are never exported in
    # saved_model_signatures.
    def ensure_host_capture_capacity(self, min_capacity):
        """(Re)create the CPU capture buffer if smaller than min_capacity.

        Grow-only. Rebuilding replaces the Variable, so the capture graph
        is re-traced; callers that know their per-call maximum (e.g.
        simulate_pass_replacement) should pre-size once before looping.
        """
        min_capacity = max(1, int(min_capacity))
        runtime = self._host_capture_runtime
        if runtime.wavefront_buffer is not None and runtime.capacity >= min_capacity:
            return
        with tf.device("/CPU:0"):
            runtime.wavefront_buffer = tf.Variable(
                tf.zeros(
                    [min_capacity, self.nx_size, self.nx_size], dtype=tf.complex64
                ),
                trainable=False,
                name="host_captured_wavefronts",
            )
        runtime.capacity = min_capacity
        runtime.graph_fn = self._build_sparse_capture_host_graph()

    def release_host_capture_buffer(self):
        """Drop the CPU capture buffer and its graph (frees host RAM)."""
        runtime = self._host_capture_runtime
        runtime.wavefront_buffer = None
        runtime.graph_fn = None
        runtime.capacity = 0

    def _build_sparse_capture_host_graph(self):
        runtime = self._host_capture_runtime
        capacity_const = tf.constant(runtime.capacity, dtype=tf.int32)

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32, name="num_steps"),
                tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="b_scale_schedule"),
                tf.TensorSpec(shape=[None], dtype=tf.int32, name="capture_indices"),
                tf.TensorSpec(shape=[None], dtype=tf.int32, name="wavefront_indices"),
            ]
        )
        def sparse_capture_host_steps(
            num_steps,
            b_scale_schedule,
            capture_indices,
            wavefront_indices,
        ):
            """simulate_steps_sparse_capture with wavefronts written to host."""
            tf.debugging.assert_equal(tf.shape(b_scale_schedule)[1], self.tf_number_of_layers)
            Uin = self._make_input_field()
            ps_arr = [tf_layer.tf_scrn for tf_layer in self.tf_layer_object_arr]

            num_captures = tf.shape(capture_indices)[0]
            captured_screens = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers, self.nx_size, self.nx_size]),
            )
            captured_b_scales = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers]),
            )
            captured_r0 = tf.TensorArray(
                dtype=tf.float32,
                size=num_captures,
                element_shape=tf.TensorShape([self.num_layers]),
            )

            num_wavefronts = tf.shape(wavefront_indices)[0]
            tf.debugging.assert_less_equal(
                num_wavefronts,
                capacity_const,
                message="host capture buffer too small; call ensure_host_capture_capacity",
            )

            capture_pos = tf.constant(0, dtype=tf.int32)
            wavefront_pos = tf.constant(0, dtype=tf.int32)
            Uout = tf.zeros([self.nx_size, self.nx_size], dtype=tf.complex64)

            for step in tf.range(num_steps):
                profile = self._schedule_profile_for_step(b_scale_schedule, step)
                self.set_b_scale_profile(profile)
                Uout, _, _ = self.propagator.propagate(Uin, ps_arr)

                next_wavefront_capture = tf.cond(
                    wavefront_pos < num_wavefronts,
                    lambda: wavefront_indices[wavefront_pos],
                    lambda: tf.constant(-1, dtype=tf.int32),
                )
                if tf.equal(step, next_wavefront_capture):
                    # Pin the scatter to the buffer's device. Without this
                    # the placer runs ResourceScatterNdUpdate on the GPU,
                    # which materializes the WHOLE CPU buffer on the GPU
                    # each call (read-modify-writeback) and defeats the
                    # offload entirely.
                    with tf.device("/CPU:0"):
                        runtime.wavefront_buffer.scatter_nd_update(
                            tf.reshape(wavefront_pos, [1, 1]),
                            Uout[tf.newaxis, ...],
                        )
                    wavefront_pos += 1

                next_screen_capture = tf.cond(
                    capture_pos < num_captures,
                    lambda: capture_indices[capture_pos],
                    lambda: tf.constant(-1, dtype=tf.int32),
                )
                if tf.equal(step, next_screen_capture):
                    captured_screens = captured_screens.write(capture_pos, self._phase_screen_stack())
                    captured_b_scales = captured_b_scales.write(
                        capture_pos, self.tf_current_b_scales.read_value()
                    )
                    captured_r0 = captured_r0.write(capture_pos, self._r0_stack())
                    capture_pos += 1

                ps_arr = self._tf_move_one_tick_all_layers()
                self.global_step.assign_add(1)

            return {
                "final_frame": Uout,
                "num_wavefronts": wavefront_pos,
                "wavefront_indices": wavefront_indices,
                "captured_screens": captured_screens.stack(),
                "captured_b_scales": captured_b_scales.stack(),
                "captured_r0": captured_r0.stack(),
                "capture_indices": capture_indices,
                "final_state_phase_screens": self._phase_screen_stack(),
                "final_b_scales": self.tf_current_b_scales.read_value(),
                "global_step": self.global_step.read_value(),
            }

        return sparse_capture_host_steps

    def simulate_steps_sparse_capture_host(
        self,
        num_steps,
        b_scale_schedule,
        capture_indices,
        wavefront_indices,
    ):
        """Sparse-capture run with wavefronts streamed to a CPU buffer.

        Same physics, state evolution and return contract as
        simulate_steps_sparse_capture, except "captured_wavefronts" comes
        back as a numpy complex64 array copied out of the host buffer, so
        the GPU never holds more than the in-flight frame. Screens and the
        small per-capture vectors still travel through TensorArrays (they
        are sparse by convention; the same buffer pattern extends to them
        if that ever changes).
        """
        num_steps_t = tf.convert_to_tensor(num_steps, dtype=tf.int32)
        schedule_t = tf.convert_to_tensor(b_scale_schedule, dtype=tf.float32)
        capture_t = tf.convert_to_tensor(capture_indices, dtype=tf.int32)
        wavefront_t = tf.convert_to_tensor(wavefront_indices, dtype=tf.int32)

        self.ensure_host_capture_capacity(int(wavefront_t.shape[0]))
        runtime = self._host_capture_runtime
        result = runtime.graph_fn(num_steps_t, schedule_t, capture_t, wavefront_t)

        # .numpy() on the count synchronizes with the graph; the buffer
        # slice is additionally ordered after the in-graph writes because
        # it reads the same variable resource. The device pin matters:
        # eager StridedSlice has no resource input, so without it TF's
        # default placement copies the ENTIRE buffer to the GPU just to
        # slice it there.
        n_written = int(result["num_wavefronts"].numpy())
        out = dict(result)
        with tf.device("/CPU:0"):
            out["captured_wavefronts"] = (
                runtime.wavefront_buffer[:n_written].numpy()
            )
        return out

    # ------------------------------------------------------------------
    # Sudden matrix-replacement schedule (Python-side driver)
    # ------------------------------------------------------------------
    # The weighted/b_scale path bakes the entire run into a single tf.function.
    # The replacement path can't, because rebuilding A_mat/B_base requires
    # NumPy-side SVDs in PhaseScreenVonKarman. So we drive segments in Python
    # and call the existing graph (`simulate_steps_sparse_capture`) once per
    # segment. All segments share the same shapes -> exactly one retrace.
    def simulate_pass_replacement(
        self,
        per_layer_r0_schedule,        # np.ndarray [num_segments, num_layers]
        segment_steps,                # int, constant length per segment
        wavefront_capture_steps=None, # 1D global step indices to dump wavefronts
        screen_capture_steps=None,    # 1D global step indices to dump screens
        regenerate_on_switch=False,
        verbose=True,
        per_segment_callback=None,    # Callable(seg_idx, dict) -> dict|None
        capture_to_host=False,        # stream wavefront captures to a CPU buffer
        match_initial_r0=True,        # start from screens at the first segment's r0
    ):
        """Run a multi-segment pass where layer r0 is hard-switched between segments.

        Returns a dict with concatenated captures, plus per-segment r0 metadata.
        When `per_segment_callback` is given, the callback is invoked with
        each segment's captures immediately after the GPU graph returns and
        BEFORE they are accumulated in memory; the captures are NOT appended
        to the run-wide buffers, so the function returns empty wavefront and
        screen arrays. The list of callback return values is returned under
        the key 'per_segment_summaries'. This keeps peak RAM at one segment's
        capture batch rather than the full run (avoids 80 GB blow-up when
        every tick is captured at res 1024).

        With `capture_to_host=True` wavefront captures bypass the GPU-side
        TensorArray/.stack() entirely and stream into a CPU tf.Variable
        buffer, so GPU peak memory no longer grows with segment_steps (the
        stack is the OOM site for dense captures at res 1024). Results are
        bit-identical to the default path. Host RAM then holds the captures
        instead; call release_host_capture_buffer() afterwards to reclaim it.

        With `match_initial_r0=True` the base screens are rescaled to the
        first segment's r0 before the pass starts (exact: a von Karman screen
        drawn from the same random numbers scales as r0^(-5/6)), so the first
        captures already have the right statistics instead of relaxing from
        the configured r0. With `regenerate_on_switch=True` each switch draws
        a fresh, seeded screen instead of keeping the current one.
        """
        per_layer_r0_schedule = np.asarray(per_layer_r0_schedule, dtype=np.float32)
        if per_layer_r0_schedule.ndim != 2:
            raise ValueError(
                f"per_layer_r0_schedule must be 2D [num_segments, num_layers], "
                f"got shape {per_layer_r0_schedule.shape}"
            )
        num_segments, ps_layers = per_layer_r0_schedule.shape
        if ps_layers != self.num_layers:
            raise ValueError(
                f"per_layer_r0_schedule has {ps_layers} layers, simulator has {self.num_layers}"
            )

        segment_steps = int(segment_steps)
        if segment_steps <= 0:
            raise ValueError("segment_steps must be > 0")
        total_steps = num_segments * segment_steps

        # Resolve capture step lists -> per-segment local indices.
        global_wavefront_steps = (
            np.asarray(wavefront_capture_steps, dtype=np.int64)
            if wavefront_capture_steps is not None
            else np.asarray([], dtype=np.int64)
        )
        global_screen_steps = (
            np.asarray(screen_capture_steps, dtype=np.int64)
            if screen_capture_steps is not None
            else np.asarray([], dtype=np.int64)
        )
        # Filter to valid range and sort/unique.
        global_wavefront_steps = np.unique(
            global_wavefront_steps[
                (global_wavefront_steps >= 0) & (global_wavefront_steps < total_steps)
            ]
        )
        global_screen_steps = np.unique(
            global_screen_steps[
                (global_screen_steps >= 0) & (global_screen_steps < total_steps)
            ]
        )

        # Reset state so every call to this method is deterministic w.r.t. the
        # b_scale knob (replacement mode owns it now).
        self.reset_to_base()
        if match_initial_r0 and num_segments and not regenerate_on_switch:
            for layer, first_r0 in zip(self.tf_layer_object_arr, per_layer_r0_schedule[0]):
                scale = (float(layer.base_scrn_r0) / float(first_r0)) ** (5.0 / 6.0)
                layer.tf_scrn.assign(layer.tf_base_scrn * np.float32(scale))

        if capture_to_host:
            # Size the CPU buffer to the busiest segment ONCE so the host
            # graph traces exactly once per pass (capacity growth would
            # otherwise re-trace mid-run).
            if global_wavefront_steps.size:
                _, seg_counts = np.unique(
                    global_wavefront_steps // segment_steps, return_counts=True
                )
                max_wf_per_segment = int(seg_counts.max())
            else:
                max_wf_per_segment = 1  # sentinel capture only
            self.ensure_host_capture_capacity(max_wf_per_segment)

        # Per-segment outputs.
        all_wavefronts = []
        all_wavefront_steps = []
        all_screens = []
        all_screen_steps = []
        all_measured_r0 = []           # realized r0 from each layer at capture time
        all_capture_b_scales = []
        per_segment_metadata = []
        per_segment_summaries = []     # populated only when per_segment_callback is set

        # Constant b_scale schedule for the graph call.
        ones_b_scale = tf.ones([segment_steps, self.num_layers], dtype=tf.float32)
        segment_steps_const = tf.constant(segment_steps, dtype=tf.int32)

        wall_start = time.time()
        for seg_idx in range(num_segments):
            global_step_offset = seg_idx * segment_steps
            new_r0_row = per_layer_r0_schedule[seg_idx]

            # ---- Hard-switch A/B matrices per layer for this segment.
            seg_switch_start = time.time()
            for layer_idx in range(self.num_layers):
                screen_seed = None
                if regenerate_on_switch:
                    screen_seed = int(np.random.SeedSequence([self.seed, layer_idx, seg_idx])
                                      .generate_state(1)[0] & 0x7FFFFFFF)
                self.tf_layer_object_arr[layer_idx].replace_matrices_for_r0(
                    float(new_r0_row[layer_idx]),
                    regenerate_scrn=regenerate_on_switch,
                    screen_seed=screen_seed,
                )
            # Sync the simulator-level b_scale cache (replacement resets it).
            self.tf_current_b_scales.assign(
                tf.ones([self.num_layers], dtype=tf.float32)
            )
            switch_time = time.time() - seg_switch_start

            # ---- Build per-segment local capture indices.
            seg_lo = global_step_offset
            seg_hi = global_step_offset + segment_steps  # exclusive
            wf_mask = (global_wavefront_steps >= seg_lo) & (global_wavefront_steps < seg_hi)
            local_wf = (global_wavefront_steps[wf_mask] - seg_lo).astype(np.int32)
            sc_mask = (global_screen_steps >= seg_lo) & (global_screen_steps < seg_hi)
            local_sc = (global_screen_steps[sc_mask] - seg_lo).astype(np.int32)

            # The graph needs at least one element to keep TensorArray shapes
            # happy across segments. Pad with sentinel `segment_steps - 1` if
            # empty and discard afterwards.
            wf_was_empty = local_wf.size == 0
            sc_was_empty = local_sc.size == 0
            if wf_was_empty:
                local_wf = np.asarray([segment_steps - 1], dtype=np.int32)
            if sc_was_empty:
                local_sc = np.asarray([segment_steps - 1], dtype=np.int32)

            # ---- Reset global_step so per-segment capture indices line up.
            self.global_step.assign(0)

            seg_run_start = time.time()
            if capture_to_host:
                result = self.simulate_steps_sparse_capture_host(
                    segment_steps_const,
                    ones_b_scale,
                    tf.convert_to_tensor(local_sc, dtype=tf.int32),
                    tf.convert_to_tensor(local_wf, dtype=tf.int32),
                )
                wf_np = result["captured_wavefronts"]  # already numpy
            else:
                result = self.simulate_steps_sparse_capture(
                    segment_steps_const,
                    ones_b_scale,
                    tf.convert_to_tensor(local_sc, dtype=tf.int32),
                    tf.convert_to_tensor(local_wf, dtype=tf.int32),
                )
                wf_np = result["captured_wavefronts"].numpy()
            run_time = time.time() - seg_run_start

            sc_np = result["captured_screens"].numpy()
            r0_np = result["captured_r0"].numpy()
            bs_np = result["captured_b_scales"].numpy()
            # Drop the graph output dict now: in GPU mode it pins the
            # stacked capture tensor, which otherwise stays alive through
            # the whole *next* segment's graph call.
            del result

            if per_segment_callback is not None:
                # Streaming mode: hand the segment's captures to the callback
                # and discard them so we never accumulate all wavefronts in RAM.
                callback_payload = {
                    "segment_index": seg_idx,
                    "global_step_offset": int(global_step_offset),
                    "wavefronts": wf_np if not wf_was_empty else np.zeros(
                        (0, self.nx_size, self.nx_size), dtype=np.complex64
                    ),
                    "wavefront_steps": (
                        (local_wf.astype(np.int64) + global_step_offset)
                        if not wf_was_empty
                        else np.zeros((0,), dtype=np.int64)
                    ),
                    "screens": sc_np if not sc_was_empty else np.zeros(
                        (0, self.num_layers, self.nx_size, self.nx_size), dtype=np.float32
                    ),
                    "screen_steps": (
                        (local_sc.astype(np.int64) + global_step_offset)
                        if not sc_was_empty
                        else np.zeros((0,), dtype=np.int64)
                    ),
                    "captured_r0": r0_np if not sc_was_empty else np.zeros(
                        (0, self.num_layers), dtype=np.float32
                    ),
                    "captured_b_scales": bs_np if not sc_was_empty else np.zeros(
                        (0, self.num_layers), dtype=np.float32
                    ),
                    "target_per_layer_r0": new_r0_row,
                }
                summary = per_segment_callback(seg_idx, callback_payload)
                per_segment_summaries.append(summary if summary is not None else {})
                # Drop the heavy arrays explicitly so the next segment starts clean.
                del wf_np, sc_np, r0_np, bs_np, callback_payload
            else:
                if not wf_was_empty:
                    all_wavefronts.append(wf_np)
                    all_wavefront_steps.append(
                        (local_wf.astype(np.int64) + global_step_offset)
                    )
                if not sc_was_empty:
                    all_screens.append(sc_np)
                    all_screen_steps.append(
                        (local_sc.astype(np.int64) + global_step_offset)
                    )
                    all_measured_r0.append(r0_np)
                    all_capture_b_scales.append(bs_np)

            per_segment_metadata.append(
                {
                    "segment_index": seg_idx,
                    "global_step_start": int(seg_lo),
                    "global_step_end_exclusive": int(seg_hi),
                    "target_per_layer_r0": new_r0_row.tolist(),
                    "switch_time_s": switch_time,
                    "run_time_s": run_time,
                }
            )

            if verbose and (seg_idx < 3 or seg_idx % max(1, num_segments // 10) == 0
                            or seg_idx == num_segments - 1):
                print(
                    f"[replacement] segment {seg_idx + 1}/{num_segments} "
                    f"steps={seg_lo}-{seg_hi - 1} "
                    f"switch={switch_time*1000:.1f}ms run={run_time:.2f}s",
                    flush=True,
                )

        total_time = time.time() - wall_start

        def _cat_or_empty(parts, shape_tail, dtype):
            if len(parts) == 1:
                # Skip the concat copy: a dense single-segment capture at
                # res 1024 would otherwise briefly double host RAM.
                return parts[0]
            if parts:
                return np.concatenate(parts, axis=0)
            return np.zeros((0, *shape_tail), dtype=dtype)

        captured_wavefronts = _cat_or_empty(
            all_wavefronts, (self.nx_size, self.nx_size), np.complex64
        )
        captured_screens = _cat_or_empty(
            all_screens,
            (self.num_layers, self.nx_size, self.nx_size),
            np.float32,
        )
        captured_r0 = _cat_or_empty(all_measured_r0, (self.num_layers,), np.float32)
        captured_b_scales = _cat_or_empty(
            all_capture_b_scales, (self.num_layers,), np.float32
        )
        wavefront_indices = (
            np.concatenate(all_wavefront_steps).astype(np.int64)
            if all_wavefront_steps
            else np.zeros((0,), dtype=np.int64)
        )
        screen_indices = (
            np.concatenate(all_screen_steps).astype(np.int64)
            if all_screen_steps
            else np.zeros((0,), dtype=np.int64)
        )

        return {
            "captured_wavefronts": captured_wavefronts,
            "wavefront_indices": wavefront_indices,
            "captured_screens": captured_screens,
            "capture_indices": screen_indices,
            "captured_r0": captured_r0,
            "captured_b_scales": captured_b_scales,
            "per_layer_r0_schedule": per_layer_r0_schedule,
            "segment_steps": segment_steps,
            "num_segments": num_segments,
            "total_steps": total_steps,
            "per_segment_metadata": per_segment_metadata,
            "per_segment_summaries": per_segment_summaries,
            "total_time_s": total_time,
            "mode": "replacement",
        }


def saved_model_signatures(simulator: GpuSimulator):
    """Return explicit signatures used for SavedModel export."""
    return {
        "serving_default": simulator.simulate_steps.get_concrete_function(),
        "simulate_steps": simulator.simulate_steps.get_concrete_function(),
        "simulate_steps_capture": simulator.simulate_steps_capture.get_concrete_function(),
        "simulate_steps_sparse_capture": simulator.simulate_steps_sparse_capture.get_concrete_function(),
        "simulate_steps_no_save": simulator.simulate_steps_no_save.get_concrete_function(),
        "reset_to_base": simulator.reset_to_base.get_concrete_function(),
        "get_state": simulator.get_state.get_concrete_function(),
        "set_state": simulator.set_state.get_concrete_function(),
        "set_b_scale_profile": simulator.set_b_scale_profile.get_concrete_function(),
    }
