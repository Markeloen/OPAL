"""Satellite-pass geometry and per-layer r0 schedules for exact pass simulation.

A circular overhead pass is sampled horizon to horizon (or in a window around
zenith). The zenith HV profile is sliced into OPAL's layers, each layer's r0
is scaled along the pass with ``r0(z) = r0_zenith cos(z)^(3/5)``, and the
result is downsampled into a segment schedule for
:meth:`opal.GpuSimulator.simulate_pass_replacement`.

All angles are in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..profiles import calc_r0_profile

__all__ = [
    "CircularOrbitPass",
    "circular_orbit_pass",
    "truncate_pass_to_window",
    "ZenithCn2Profile",
    "build_zenith_cn2_profile",
    "per_layer_r0_along_pass",
    "effective_path_r0",
    "build_replacement_schedule",
]

_G = 6.67408e-11
_M_EARTH = 5.972e24
_R_EARTH = 6.378e6


@dataclass
class CircularOrbitPass:
    """Time-resolved geometry of an overhead circular-orbit pass (t = 0 at zenith)."""

    t: np.ndarray
    elevation_deg: np.ndarray
    zenith_angle_deg: np.ndarray
    range_m: np.ndarray
    v_radial: np.ndarray
    v_tangential: np.ndarray
    slew_rate_rad_s: np.ndarray
    h_orbit: float
    min_elevation_deg: float

    @property
    def duration_s(self):
        return float(self.t[-1] - self.t[0])


def circular_orbit_pass(h_orbit=700_000.0, min_elevation_deg=20.0, num_samples=1024):
    """Overhead pass of a circular orbit between two ``min_elevation_deg`` horizons."""
    R = _R_EARTH + float(h_orbit)
    v_orbit = np.sqrt(_G * _M_EARTH / R)
    w_orbit = 2.0 * np.pi / (2.0 * np.pi * R / v_orbit)

    g_min = np.deg2rad(float(min_elevation_deg))
    coeffs = [1.0, -2.0 * _R_EARTH * np.cos(g_min + np.pi / 2.0), _R_EARTH**2 - R**2]
    L_max = max(np.roots(coeffs).real)
    gamma_earth_max = np.arccos((L_max**2 - R**2 - _R_EARTH**2) / (-2.0 * R * _R_EARTH))

    gamma_earth = np.linspace(-gamma_earth_max, gamma_earth_max, int(num_samples))
    horiz_period = 2.0 * gamma_earth_max / w_orbit
    t = np.linspace(-horiz_period / 2.0, horiz_period / 2.0, int(num_samples))

    L = np.sqrt(R**2 + _R_EARTH**2 - 2.0 * R * _R_EARTH * np.cos(gamma_earth))
    gamma_horiz = np.arccos((R**2 - L**2 - _R_EARTH**2) / (-2.0 * L * _R_EARTH)) - np.pi / 2.0
    elevation = np.rad2deg(gamma_horiz)

    psi = np.pi - np.abs(gamma_earth) - (np.pi / 2.0 + gamma_horiz)
    v_radial = v_orbit * np.cos(np.pi / 2.0 - psi)
    v_tangential = v_orbit * np.sin(np.pi / 2.0 - psi)
    v_radial = v_radial.copy()
    v_radial[int(num_samples) // 2:] *= -1

    return CircularOrbitPass(
        t=t.astype(np.float64),
        elevation_deg=elevation.astype(np.float64),
        zenith_angle_deg=(90.0 - elevation).astype(np.float64),
        range_m=L.astype(np.float64),
        v_radial=v_radial.astype(np.float64),
        v_tangential=v_tangential.astype(np.float64),
        slew_rate_rad_s=(v_tangential / np.maximum(L, 1.0)).astype(np.float64),
        h_orbit=float(h_orbit),
        min_elevation_deg=float(min_elevation_deg),
    )


def truncate_pass_to_window(orbit_pass, window_s):
    """Keep ``[-window_s/2, +window_s/2]`` around zenith; ``None``/``<=0`` keeps all."""
    if window_s is None or window_s <= 0:
        return orbit_pass
    half = float(window_s) / 2.0
    mask = (orbit_pass.t >= -half) & (orbit_pass.t <= half)
    if not np.any(mask):
        return orbit_pass
    return replace(
        orbit_pass,
        t=orbit_pass.t[mask],
        elevation_deg=orbit_pass.elevation_deg[mask],
        zenith_angle_deg=orbit_pass.zenith_angle_deg[mask],
        range_m=orbit_pass.range_m[mask],
        v_radial=orbit_pass.v_radial[mask],
        v_tangential=orbit_pass.v_tangential[mask],
        slew_rate_rad_s=orbit_pass.slew_rate_rad_s[mask],
    )


@dataclass
class ZenithCn2Profile:
    """Per-layer r0 at zenith plus the profile it came from."""

    r0_layers_zenith: np.ndarray
    layer_heights: np.ndarray
    wind_profile: np.ndarray
    cn2_callable: object
    effective_r0_zenith: float


def build_zenith_cn2_profile(num_layers, wavelength, ground_wind_speed=5.0,
                             satellite_orbit=700_000.0, total_r0_zenith=None,
                             max_altitude=20_000.0):
    """Slice the zenith HV profile exactly as the simulator does."""
    r0_array, heights, wind, cn2 = calc_r0_profile(
        ground_wind_speed,
        satellite_orbit,
        False,
        total_r0_zenith,
        num_layers,
        wavelength,
        max_altitude=max_altitude,
        zenith_angle=0.0,
    )
    r0_layers = np.asarray(r0_array, dtype=np.float64)
    return ZenithCn2Profile(
        r0_layers_zenith=r0_layers,
        layer_heights=np.asarray(heights, dtype=np.float64),
        wind_profile=np.asarray(wind, dtype=np.float64),
        cn2_callable=cn2,
        effective_r0_zenith=float(np.sum(r0_layers ** (-5.0 / 3.0)) ** (-3.0 / 5.0)),
    )


def per_layer_r0_along_pass(zenith_profile, zenith_angle_deg_t, max_zenith_angle_deg=85.0):
    """Slant-path scaling ``r0 = r0_zenith cos(z)^(3/5)``, shape ``[T, layers]``.

    Zenith angles are clamped at ``max_zenith_angle_deg``.
    """
    z = np.clip(np.asarray(zenith_angle_deg_t, dtype=np.float64),
                -max_zenith_angle_deg, max_zenith_angle_deg)
    scale = np.power(np.maximum(np.cos(np.deg2rad(z)), 1e-6), 3.0 / 5.0)
    return (zenith_profile.r0_layers_zenith[None, :] * scale[:, None]).astype(np.float64)


def effective_path_r0(per_layer_r0):
    """Path r0 over the last axis: ``(sum r0_i^(-5/3))^(-3/5)``."""
    arr = np.asarray(per_layer_r0, dtype=np.float64)
    finite = np.isfinite(arr) & (arr > 0)
    out = np.full(arr.shape[:-1], np.nan, dtype=np.float64)
    mask = np.any(finite, axis=-1)
    safe = np.where(finite, arr, np.inf)
    full = np.power(np.sum(np.power(safe, -5.0 / 3.0), axis=-1), -3.0 / 5.0)
    out[mask] = full[mask]
    return out


def build_replacement_schedule(per_layer_r0_t, num_steps, num_segments):
    """Sample ``r0(t)`` at segment midpoints.

    ``num_steps`` must be a multiple of ``num_segments``.

    Returns
    -------
    (schedule [segments, layers] float32, segment_steps, segment_starts)
    """
    per_layer_r0_t = np.asarray(per_layer_r0_t, dtype=np.float64)
    if per_layer_r0_t.ndim != 2:
        raise ValueError("per_layer_r0_t must be 2D [T_pass, num_layers]")
    T_pass = per_layer_r0_t.shape[0]
    num_segments = int(num_segments)
    if num_segments <= 0 or num_steps <= 0:
        raise ValueError("num_steps and num_segments must be positive")
    if num_steps % num_segments != 0:
        raise ValueError(f"num_steps ({num_steps}) must be a multiple of num_segments ({num_segments})")
    segment_steps = num_steps // num_segments
    mid = (np.arange(num_segments, dtype=np.float64) + 0.5) / num_segments
    idx = np.clip((mid * (T_pass - 1)).round().astype(int), 0, T_pass - 1)
    starts = np.arange(num_segments, dtype=np.int64) * segment_steps
    return per_layer_r0_t[idx, :].astype(np.float32), segment_steps, starts
