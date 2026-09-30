"""Single-mode fiber coupling of received fields and its theory.

Measured coupling uses the normalised overlap of the aperture-masked field
with a Gaussian fiber mode (Dikmelik & Davidson 2005, Eq. 1), optionally
after centre-of-mass tip-tilt removal. Theory is available for Kolmogorov
(exact Eq. 15 and the Gaussian Eq. 19 approximation), von Karman (Eq. 15 with
the finite-L0 structure function) and tip-tilt-corrected von Karman
(Monte Carlo on OPAL's own FFT screens).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import i0e

from ..phase_screens import ft_sh_phase_screen
from .r0 import von_karman_sf_coefficient

__all__ = [
    "circular_mask",
    "gaussian_fiber_mode",
    "remove_tilt_com",
    "compute_fiber_coupling",
    "coupling_efficiency_theoretical",
    "coupling_efficiency_theoretical_vk",
    "coupling_efficiency_theoretical_tt",
    "effective_r0",
    "expected_layer_r0",
]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def circular_mask(shape, center=None, radius=None):
    """Boolean circular aperture mask (radius in pixels)."""
    h, w = shape
    cy, cx = center if center is not None else (h // 2, w // 2)
    if radius is None:
        radius = min(cy, cx, h - cy, w - cx)
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def gaussian_fiber_mode(shape, pixel_scale, a=1.12, aperture_diameter=None):
    """Unit-power Gaussian fiber mode back-propagated to the pupil.

    ``a = D / (2 w0)`` is the coupling-geometry parameter (1.12 is optimal
    for an unobstructed aperture). Returns ``(mode, aperture_diameter, w0)``.
    """
    n = int(shape[0])
    grid_size = n * float(pixel_scale)
    if aperture_diameter is None:
        aperture_diameter = grid_size
    x = np.linspace(-grid_size / 2.0, grid_size / 2.0, n)
    xx, yy = np.meshgrid(x, x)
    r2 = xx**2 + yy**2
    omega_0 = float(aperture_diameter) / (2.0 * float(a))
    mode = np.sqrt(2.0 / (np.pi * omega_0**2)) * np.exp(-(r2 / omega_0**2))
    return mode.astype(np.complex128), float(aperture_diameter), omega_0


def remove_tilt_com(wavefronts, pixel_scale, aperture_diameter=None, pad_factor=2):
    """Centre-of-mass tip-tilt removal.

    Per frame: Fraunhofer-propagate the masked field, take the PSF intensity
    centre of mass ``<f>``, and multiply the pupil field by the conjugate ramp
    ``exp(-i 2 pi (x <fx> + y <fy>))``.

    Returns ``(corrected, tilts)`` with tilts in cycles/m ``(fx, fy)`` per frame.
    """
    wavefronts = np.asarray(wavefronts, dtype=np.complex128)
    if wavefronts.ndim == 2:
        wavefronts = wavefronts[None]
    n = wavefronts.shape[-1]
    grid_size = n * float(pixel_scale)
    if aperture_diameter is None:
        aperture_diameter = grid_size
    radius_px = float(aperture_diameter) / (2.0 * float(pixel_scale))
    mask = circular_mask((n, n), radius=radius_px).astype(np.float64)

    npad = int(pad_factor) * n
    f_coords = np.fft.fftshift(np.fft.fftfreq(npad, d=pixel_scale))
    x_phys = (np.arange(n) - n / 2.0) * float(pixel_scale)
    xx, yy = np.meshgrid(x_phys, x_phys)

    corrected = np.empty_like(wavefronts)
    tilts = np.zeros((len(wavefronts), 2), dtype=np.float64)
    for idx, frame in enumerate(wavefronts):
        focal = np.fft.fftshift(np.fft.fft2(frame * mask, s=(npad, npad)))
        intensity = np.abs(focal) ** 2
        total = intensity.sum()
        if total <= 0:
            corrected[idx] = frame
            continue
        fy_mean = float(np.sum(intensity.sum(axis=1) * f_coords) / total)
        fx_mean = float(np.sum(intensity.sum(axis=0) * f_coords) / total)
        tilts[idx] = (fx_mean, fy_mean)
        corrected[idx] = frame * np.exp(-1j * 2.0 * np.pi * (fx_mean * xx + fy_mean * yy))
    return corrected, tilts


def compute_fiber_coupling(wavefronts, pixel_scale, a=1.12, aperture_diameter=None):
    """Single-mode coupling efficiency of complex fields ``[frames, N, N]``.

    The overlap runs over the receiver aperture while the fiber mode is
    normalised to unit power over the full plane (Dikmelik & Davidson Eqs. 1, 5).

    Returns a dict with ``eta`` per frame, the aperture ``mask``, the masked
    ``mode``, ``aperture_diameter`` and ``omega_0``.
    """
    wavefronts = np.asarray(wavefronts, dtype=np.complex128)
    if wavefronts.ndim == 2:
        wavefronts = wavefronts[None]
    if wavefronts.ndim != 3:
        raise ValueError(f"Expected wavefronts with shape [frames, N, N], got {wavefronts.shape}")

    grid_size = wavefronts.shape[-1] * float(pixel_scale)
    if aperture_diameter is None:
        aperture_diameter = grid_size
    if aperture_diameter > grid_size:
        raise ValueError(f"Aperture diameter {aperture_diameter} m exceeds grid size {grid_size} m")

    mode, aperture_diameter, omega_0 = gaussian_fiber_mode(
        wavefronts.shape[-2:], pixel_scale, a, aperture_diameter=aperture_diameter
    )
    radius_px = float(aperture_diameter) / (2.0 * float(pixel_scale))
    mask = circular_mask(wavefronts.shape[-2:], radius=radius_px).astype(np.complex128)
    dA = float(pixel_scale) ** 2
    mode_masked = mode * mask
    mode_power = float(np.sum(np.abs(mode) ** 2) * dA)
    masked = wavefronts * mask[None, :, :]
    wavefront_power = np.sum(np.abs(masked) ** 2, axis=(-2, -1)) * dA
    overlaps = np.sum(masked * np.conj(mode_masked), axis=(-2, -1)) * dA
    denominator = wavefront_power * mode_power
    eta = np.divide(
        np.abs(overlaps) ** 2,
        denominator,
        out=np.full_like(wavefront_power, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "eta": np.clip(eta, 0.0, 1.0),
        "mask": mask,
        "mode": mode_masked,
        "aperture_diameter": float(aperture_diameter),
        "omega_0": float(omega_0),
        "dA": dA,
    }


# Speckle count A_R/A_c = (D / (2 rho_c))^2 with rho_c = 3.44^(-3/5) r0.
_SPECKLE_COEFF = 3.44 ** (6.0 / 5.0) / 4.0


@lru_cache(maxsize=2048)
def _eta_gaussian(d_over_r0, a):
    if not np.isfinite(d_over_r0) or d_over_r0 <= 0:
        return math.nan
    ar_over_ac = _SPECKLE_COEFF * float(d_over_r0) ** 2
    x = np.linspace(0.0, 1.0, 181, dtype=np.float64)
    x1, x2 = np.meshgrid(x, x, indexing="ij")
    r2 = x1**2 + x2**2
    z = 2.0 * ar_over_ac * x1 * x2
    integrand = np.exp(z - (a * a + ar_over_ac) * r2) * i0e(z) * x1 * x2
    result = _trapz(_trapz(integrand, x, axis=0), x, axis=0)
    return float(np.clip(8.0 * a * a * result, 0.0, 1.0))


@lru_cache(maxsize=2048)
def _eta_kolmogorov(d_over_r0, a):
    if not np.isfinite(d_over_r0) or d_over_r0 <= 0:
        return math.nan
    ar_over_ac = _SPECKLE_COEFF * float(d_over_r0) ** 2
    x = np.linspace(0.0, 1.0, 121, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, 96, dtype=np.float64)
    x1, x2 = np.meshgrid(x, x, indexing="ij")
    r2 = x1**2 + x2**2
    u = np.divide(2.0 * x1 * x2, r2, out=np.zeros_like(r2), where=r2 > 0)
    v56 = (ar_over_ac * r2) ** (5.0 / 6.0)
    inner = np.clip(1.0 - u[:, :, None] * np.cos(theta)[None, None, :], 0.0, None) ** (5.0 / 6.0)
    F = _trapz(np.exp(-v56[:, :, None] * inner), theta, axis=2)
    integrand = np.exp(-(a * a) * r2) * F * x1 * x2
    result = _trapz(_trapz(integrand, x, axis=0), x, axis=0)
    return float(np.clip(8.0 * a * a / np.pi * result, 0.0, 1.0))


def coupling_efficiency_theoretical(d_over_r0, a=1.12, method="kolmogorov"):
    """Plane-wave coupling efficiency vs ``D/r0`` for Kolmogorov turbulence.

    ``method="kolmogorov"`` evaluates the exact Eq. (15); ``"gaussian"`` the
    faster Eq. (19) approximation.
    """
    key = (round(float(d_over_r0), 5), round(float(a), 5))
    if method == "gaussian":
        return _eta_gaussian(*key)
    return _eta_kolmogorov(*key)


@lru_cache(maxsize=2048)
def _eta_von_karman(aperture_diameter, r0, L0, a):
    if not (np.isfinite(r0) and r0 > 0 and np.isfinite(aperture_diameter) and aperture_diameter > 0):
        return math.nan
    x = np.linspace(0.0, 1.0, 121, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, 96, dtype=np.float64)
    x1, x2 = np.meshgrid(x, x, indexing="ij")
    r2 = x1**2 + x2**2
    cos_t = np.cos(theta)[None, None, :]
    rho = (aperture_diameter / 2.0) * np.sqrt(
        np.clip(r2[:, :, None] - 2.0 * (x1 * x2)[:, :, None] * cos_t, 0.0, None)
    )
    half_sf = np.zeros_like(rho)
    pos = rho > 0
    half_sf[pos] = 0.5 * von_karman_sf_coefficient(rho[pos], L0) * (L0 / r0) ** (5.0 / 3.0)
    F = _trapz(np.exp(-half_sf), theta, axis=2)
    integrand = np.exp(-(a * a) * r2) * F * x1 * x2
    result = _trapz(_trapz(integrand, x, axis=0), x, axis=0)
    return float(np.clip(8.0 * a * a / np.pi * result, 0.0, 1.0))


def coupling_efficiency_theoretical_vk(aperture_diameter, r0, L0, a=1.12):
    """Coupling efficiency for a von Karman MCF (depends on D and L0 separately)."""
    return _eta_von_karman(round(float(aperture_diameter), 5), round(float(r0), 6),
                           round(float(L0), 3), round(float(a), 5))


_TT_MC_N = 256
_TT_MC_N_SCREENS = 100
_TT_MC_INNER_SCALE = 0.001
_TT_MC_APERTURE_FRACTION = 0.3125
_TT_MC_SEED_BASE = 7777


@lru_cache(maxsize=2048)
def _eta_tt(aperture_diameter, r0, L0, a):
    if not (np.isfinite(r0) and r0 > 0 and np.isfinite(aperture_diameter) and aperture_diameter > 0):
        return math.nan
    pixel_scale = aperture_diameter / (_TT_MC_N * _TT_MC_APERTURE_FRACTION)
    rng = np.random.default_rng(_TT_MC_SEED_BASE)
    seeds = rng.integers(1, 2**31 - 1, size=_TT_MC_N_SCREENS)
    screens = np.stack([
        ft_sh_phase_screen(r0, _TT_MC_N, pixel_scale, L0, _TT_MC_INNER_SCALE, seed=int(s))
        for s in seeds
    ])
    corrected, _ = remove_tilt_com(np.exp(1j * screens), pixel_scale, aperture_diameter=aperture_diameter)
    eta = compute_fiber_coupling(corrected, pixel_scale, a=a, aperture_diameter=aperture_diameter)["eta"]
    return float(np.mean(eta))


def coupling_efficiency_theoretical_tt(aperture_diameter, r0, L0, a=1.12):
    """Tip-tilt-corrected coupling efficiency (Monte Carlo over 100 von Karman screens).

    Matches the Marechal estimate ``0.8145 exp(-0.134 (D/r0)^(5/3))`` for
    ``D/r0 <~ 2`` and remains valid in the speckle regime.
    """
    return _eta_tt(round(float(aperture_diameter), 5), round(float(r0), 6),
                   round(float(L0), 3), round(float(a), 5))


def effective_r0(layer_r0):
    """Path r0 of a layer stack: ``(sum r0_i^(-5/3))^(-3/5)``."""
    layer_r0 = np.asarray(layer_r0, dtype=np.float64)
    finite = np.isfinite(layer_r0) & (layer_r0 > 0)
    if not np.any(finite):
        return math.nan
    return float(np.sum(layer_r0[finite] ** (-5.0 / 3.0)) ** (-3.0 / 5.0))


def expected_layer_r0(base_r0_array, b_scale_row):
    """Per-layer r0 implied by a b-scale row: ``r0_base / b^(6/5)``."""
    return np.asarray(base_r0_array, dtype=np.float32) / np.power(
        np.maximum(np.asarray(b_scale_row, dtype=np.float32), 1e-12), 6.0 / 5.0
    )
