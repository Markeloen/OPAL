"""Layer stack geometry: per-layer r0, heights, winds and extrusion rates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .layer import GpuLayer
from .profiles import calc_r0_profile, calculate_number_of_extrusions

__all__ = ["GPUGeometry", "derive_layer_seeds"]


def derive_layer_seeds(seed, number_of_layers):
    """Distinct deterministic per-layer seeds from a master seed.

    Returns ``[None] * number_of_layers`` when ``seed`` is ``None`` (every
    layer then draws its own random seed).
    """
    if seed is None:
        return [None] * number_of_layers
    rng = np.random.default_rng(seed)
    return [int(rng.integers(1, 2**31 - 1)) for _ in range(number_of_layers)]


@dataclass
class GPUGeometry:
    """Multilayer channel geometry and the GPU layers that realise it.

    On construction the HV profile is sliced into ``number_of_layers`` layers
    (see :func:`opal.profiles.calc_r0_profile`) and one :class:`GpuLayer` is
    built per layer with its own r0 and seed.

    Attributes
    ----------
    r0_array, layer_height_array, wind_profile_array : list of float
        Per-layer Fried parameter (m), height (m) and wind speed (m/s).
    number_extrusions_array_per_tick : list of float
        Rows each layer advances per tick (``v * dt / dx``).
    extrusion_muliplication_factor : int
        Integer factor applied to the per-tick extrusion rates so the slowest
        (ground) layer advances at least one whole row per tick.
    layer_seeds : list of int or None
        Seeds of the individual layers.
    """

    nx_size: int
    pixel_scale: float
    wvl: float
    r0: float = None
    number_of_layers: int = 10
    whole_simulation_time: float = 1
    per_tick_simulation: float = 1e-3
    satellite_orbit: float = 6e5
    ground_wind_speed: float = 5
    satellite_zenith_angle_starting: float = 90
    L0: float = 50
    seed: int | None = None
    on_layer: object = field(default=None, repr=False, compare=False)
    r0_array: list = field(default_factory=list, init=False)
    layer_height_array: list = field(default_factory=list, init=False)
    wind_profile_array: list = field(default_factory=list, init=False)
    layer_object_array: list = field(default_factory=list, init=False)
    number_extrusions_array_per_tick: list = field(default_factory=list, init=False)
    number_extrusion_array_whole: list = field(default_factory=list, init=False)
    layer_seeds: list = field(default_factory=list, init=False)

    def __post_init__(self):
        results = calc_r0_profile(
            self.ground_wind_speed,
            self.satellite_orbit,
            False,
            self.r0,
            self.number_of_layers,
            self.wvl,
            zenith_angle=self.satellite_zenith_angle_starting,
        )
        self.r0_array, self.layer_height_array, self.wind_profile_array, self.cn2 = results

        self.Cn2_profile_arr = [self.cn2(h) for h in self.layer_height_array]

        self.layer_seeds = derive_layer_seeds(self.seed, self.number_of_layers)

        for i in range(self.number_of_layers):
            if self.on_layer is not None:
                self.on_layer(i, self.number_of_layers, float(self.r0_array[i]))
            self.layer_object_array.append(
                GpuLayer(
                    self.nx_size,
                    self.pixel_scale,
                    self.r0_array[i],
                    self.L0,
                    random_seed=self.layer_seeds[i],
                )
            )
            self.number_extrusions_array_per_tick.append(
                calculate_number_of_extrusions(
                    self.pixel_scale, self.wind_profile_array[i], self.per_tick_simulation
                )
            )

        self.number_extrusion_array_whole = np.ceil(
            np.array(self.number_extrusions_array_per_tick)
            * self.whole_simulation_time
            / self.per_tick_simulation
        )

        first = self.number_extrusions_array_per_tick[0]
        self.extrusion_muliplication_factor = math.ceil(1 / first) if first < 1 else math.ceil(first)

    def describe(self):
        """Per-layer summary as a list of dicts (height, r0, wind, rows/tick, seed)."""
        rows = []
        for i in range(self.number_of_layers):
            rows.append(
                {
                    "layer": i,
                    "height_m": float(self.layer_height_array[i]),
                    "r0_m": float(np.asarray(self.r0_array[i]).item()),
                    "wind_mps": float(self.wind_profile_array[i]),
                    "cn2": float(np.asarray(self.Cn2_profile_arr[i]).item()),
                    "rows_per_tick": float(np.asarray(self.number_extrusions_array_per_tick[i]).item()),
                    "seed": self.layer_seeds[i],
                }
            )
        return rows

    def show_object_info(self, logger):
        """Log the geometry through ``logger``."""
        logger.info(f"nx_size: {self.nx_size}")
        logger.info(f"pixel_scale: {self.pixel_scale}")
        logger.info(f"per_tick_simulation: {self.per_tick_simulation}")
        logger.info(f"number_of_layers: {self.number_of_layers}")
        logger.info(f"satellite_orbit: {self.satellite_orbit}")
        logger.info(f"ground_wind_speed: {self.ground_wind_speed}")
        logger.info(f"r0_array: {self.r0_array}")
        logger.info(f"layer_height_array: {self.layer_height_array}")
        logger.info(f"Cn2 profile: {self.Cn2_profile_arr}")
        logger.info(f"full r0: {self.r0}")
        logger.info(f"wind_profile_array: {self.wind_profile_array}")
        logger.info(f"number_extrusions_array_per_tick: {self.number_extrusions_array_per_tick}")
        logger.info(f"number_extrusion_array_whole: {self.number_extrusion_array_whole}")
        logger.info(f"extrusion_muliplication_factor: {self.extrusion_muliplication_factor}")
        logger.info(f"D / r0: {self.nx_size * self.pixel_scale / self.r0}")
