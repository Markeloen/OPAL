"""Per-frame field statistics used for dataset labels and quick analysis."""

from __future__ import annotations

import numpy as np

__all__ = ["aperture_mask", "scintillation_index", "tilt_centroid"]


def aperture_mask(n, pixel_scale, diameter=None):
    """Boolean circular aperture of ``diameter`` metres centred on an ``n x n`` grid."""
    if diameter is None:
        diameter = n * pixel_scale
    x = (np.arange(n) - n / 2.0) * pixel_scale
    xx, yy = np.meshgrid(x, x)
    return (xx**2 + yy**2) <= (diameter / 2.0) ** 2


def scintillation_index(field, mask=None):
    """Spatial scintillation index ``<I^2>/<I>^2 - 1`` of a complex field over ``mask``."""
    intensity = np.abs(np.asarray(field)) ** 2
    if mask is not None:
        intensity = intensity[mask]
    mean = intensity.mean()
    if mean <= 0:
        return float("nan")
    return float((intensity**2).mean() / mean**2 - 1.0)


def tilt_centroid(field, pixel_scale, wavelength, mask=None, pad_factor=2):
    """Angle of arrival (rad) of a field from its far-field centre of mass.

    Returns ``(theta_x, theta_y)``.
    """
    field = np.asarray(field, dtype=np.complex64)
    n = field.shape[-1]
    if mask is not None:
        field = field * mask
    npad = int(pad_factor) * n
    f = np.fft.fftshift(np.fft.fftfreq(npad, d=pixel_scale))
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(field, s=(npad, npad)))) ** 2
    total = intensity.sum()
    if total <= 0:
        return float("nan"), float("nan")
    fx = float(np.sum(intensity.sum(axis=0) * f) / total)
    fy = float(np.sum(intensity.sum(axis=1) * f) / total)
    return wavelength * fx, wavelength * fy
