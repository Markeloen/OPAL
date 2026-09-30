"""Fried-parameter estimators for phase screens and received wavefronts.

* MCF 1/e crossing (Kolmogorov and von Karman inversions), applied to a
  phase screen (``exp(i phi)``) or to a complex received field.
* Phase structure-function fit in the inertial range, which is insensitive
  to finite-screen and outer-scale bias.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma, kv

from .._fft import circ, corr2_ft

__all__ = [
    "radial_average",
    "structure_function_2d",
    "von_karman_sf_coefficient",
    "estimate_r0_from_mcf_1e",
    "estimate_r0_from_mcf_1e_vk",
    "estimate_r0_from_structure_function",
    "estimate_r0_from_phase_screen",
    "estimate_r0_from_wavefront",
]


def radial_average(data):
    """Azimuthal average of ``|data|`` about the array centre (1-pixel bins)."""
    data = np.abs(np.asarray(data))
    center = (data.shape[0] // 2, data.shape[1] // 2)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
    counts = np.bincount(r.ravel())
    counts = np.where(counts == 0, 1, counts)
    return np.bincount(r.ravel(), data.ravel()) / counts


def _one_over_e_crossing(radial_mcf, pixel_scale):
    target = 1 / np.e
    radial_mcf = np.asarray(radial_mcf, dtype=float)
    cross = np.where(np.diff(np.signbit(radial_mcf - target)))[0]
    if len(cross) == 0:
        return None
    idx = int(cross[0])
    r1, r2 = idx * pixel_scale, (idx + 1) * pixel_scale
    m1, m2 = radial_mcf[idx], radial_mcf[idx + 1]
    if m2 == m1:
        return None
    return r1 + (target - m1) * (r2 - r1) / (m2 - m1)


def estimate_r0_from_mcf_1e(radial_mcf, pixel_scale):
    """Kolmogorov inversion of the MCF 1/e width: ``r0 = r_1e / (1/3.44)^(3/5)``.

    Returns ``(r0, r_1e)`` or ``(None, None)`` if the MCF never reaches 1/e.
    """
    r_1e = _one_over_e_crossing(radial_mcf, pixel_scale)
    if r_1e is None:
        return None, None
    return float(r_1e / ((1 / 3.44) ** (3 / 5))), float(r_1e)


def structure_function_2d(phase):
    """Isotropic phase structure function ``D(r)`` of a screen (pixel-separation bins).

    Uses zero-padded FFT correlations, so the estimate is unbiased by the
    finite, non-periodic window.
    """
    phase = np.asarray(phase, dtype=np.float64)
    phase = phase - phase.mean()
    n = phase.shape[0]
    pad = 2 * n
    mask = np.ones_like(phase)

    f_phase = np.fft.rfft2(phase, s=(pad, pad))
    f_phase2 = np.fft.rfft2(phase * phase, s=(pad, pad))
    f_mask = np.fft.rfft2(mask, s=(pad, pad))

    corr_pp = np.fft.irfft2(f_phase * np.conj(f_phase), s=(pad, pad))
    corr_p2m = np.fft.irfft2(f_phase2 * np.conj(f_mask), s=(pad, pad))
    corr_mp2 = np.fft.irfft2(f_mask * np.conj(f_phase2), s=(pad, pad))
    counts = np.fft.irfft2(f_mask * np.conj(f_mask), s=(pad, pad))

    d2 = (corr_p2m + corr_mp2 - 2.0 * corr_pp) / np.maximum(counts, 1e-9)
    d2 = np.fft.fftshift(d2)[pad // 2 - n // 2: pad // 2 + n // 2,
                            pad // 2 - n // 2: pad // 2 + n // 2]
    return radial_average(d2)


def von_karman_sf_coefficient(r, L0):
    """``C(r, L0)`` with ``D(r) = C (L0 / r0)^(5/3)`` for von Karman turbulence (Conan 2000).

    Reduces to ``6.88 (r / r0)^(5/3)`` for ``r << L0``.
    """
    x = 2.0 * np.pi * np.asarray(r, dtype=np.float64) / float(L0)
    lead = (2.0 ** (1.0 / 6.0) * gamma(11.0 / 6.0) / np.pi ** (8.0 / 3.0)) * (
        (24.0 / 5.0) * gamma(6.0 / 5.0)
    ) ** (5.0 / 6.0)
    bracket = gamma(5.0 / 6.0) / 2.0 ** (1.0 / 6.0) - x ** (5.0 / 6.0) * kv(5.0 / 6.0, x)
    return lead * bracket


def estimate_r0_from_mcf_1e_vk(radial_mcf, pixel_scale, L0):
    """Von Karman inversion of the MCF 1/e width: ``r0 = L0 (C(r_1e, L0) / 2)^(3/5)``."""
    r_1e = _one_over_e_crossing(radial_mcf, pixel_scale)
    if r_1e is None:
        return None, None
    c = float(von_karman_sf_coefficient(np.asarray([r_1e]), L0)[0])
    if not np.isfinite(c) or c <= 0:
        return None, None
    return float(L0 * (0.5 * c) ** (3.0 / 5.0)), float(r_1e)


def estimate_r0_from_structure_function(screen, pixel_scale, L0=None, fit_range_pixels=None):
    """Estimate r0 from the phase structure function at small separations.

    Parameters
    ----------
    L0 : float, optional
        Invert the von Karman structure function; otherwise Kolmogorov.
    fit_range_pixels : (lo, hi), optional
        Inclusive separation range (pixels). Default ``(2, n // 8)``.

    Returns
    -------
    dict with ``measured_r0`` (median over bins), ``per_bin_r0``,
    ``separations_m`` and ``structure_function``.
    """
    screen = np.asarray(screen, dtype=np.float64)
    n = screen.shape[0]
    lo, hi = fit_range_pixels or (2, max(4, n // 8))

    d_r = structure_function_2d(screen)
    hi = min(hi, len(d_r) - 1)
    r = np.arange(lo, hi + 1) * pixel_scale
    d = d_r[lo: hi + 1]
    valid = d > 0
    if not np.any(valid):
        return {"measured_r0": math.nan, "per_bin_r0": np.array([])}
    r, d = r[valid], d[valid]

    if L0 is not None:
        per_bin = L0 * (von_karman_sf_coefficient(r, L0) / d) ** (3.0 / 5.0)
    else:
        per_bin = r * (6.88 / d) ** (3.0 / 5.0)
    return {
        "measured_r0": float(np.median(per_bin)),
        "per_bin_r0": per_bin,
        "separations_m": r,
        "structure_function": d,
    }


def _mcf_estimate(field, pixel_scale, aperture_diameter, L0):
    n = field.shape[0]
    if aperture_diameter is None:
        aperture_diameter = n * pixel_scale
    x = np.arange(-n / 2.0, n / 2.0)
    nx, ny = np.meshgrid(x, x)
    mask = circ(nx * pixel_scale, ny * pixel_scale, aperture_diameter)
    mcf = corr2_ft(field, field, mask, pixel_scale)
    center = mcf[n // 2, n // 2]
    if center == 0:
        return {"measured_r0": math.nan, "r_1e": math.nan,
                "radial_mcf": np.array([], dtype=float), "mcf": mcf}
    normalized = np.abs(mcf / center)
    radial = radial_average(normalized)
    if L0 is not None and float(L0) > 0:
        r0, r_1e = estimate_r0_from_mcf_1e_vk(radial, pixel_scale, float(L0))
    else:
        r0, r_1e = estimate_r0_from_mcf_1e(radial, pixel_scale)
    return {
        "measured_r0": math.nan if r0 is None else r0,
        "r_1e": math.nan if r_1e is None else r_1e,
        "radial_mcf": radial,
        "mcf": normalized,
    }


def estimate_r0_from_phase_screen(screen, pixel_scale, aperture_diameter=None, L0=None):
    """r0 of a single phase screen from the MCF of ``exp(i screen)``."""
    return _mcf_estimate(np.exp(1j * np.asarray(screen, dtype=np.float32)),
                         pixel_scale, aperture_diameter, L0)


def estimate_r0_from_wavefront(U, pixel_scale, aperture_diameter=None, L0=None):
    """Effective path r0 of a complex received field from its MCF 1/e width.

    With ``L0 > 0`` the von Karman inversion is used, otherwise Kolmogorov.
    The aperture must be comfortably inside the grid (the absorbing boundary
    starts at ~0.47 N).
    """
    return _mcf_estimate(np.asarray(U, dtype=np.complex64), pixel_scale, aperture_diameter, L0)
