"""NumPy FFT helpers shared by the phase-screen generator and the analysis tools.

Conventions follow Schmidt, *Numerical Simulation of Optical Wave Propagation*
(SPIE, 2010): centred, physically scaled transforms.
"""

import numpy as np

__all__ = ["ft2", "ift2", "circ", "corr2_ft"]


def ft2(data, delta):
    """Centred, scaled 2-D FFT: ``fftshift(fft2(fftshift(g))) * delta**2``."""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data))) * delta**2


def ift2(G, delta_f, FFT=None):
    """Centred, scaled inverse 2-D FFT.

    Parameters
    ----------
    G : ndarray
        Spectrum to transform.
    delta_f : float
        Frequency-grid spacing.
    FFT : callable, optional
        Accelerated FFT object (e.g. pyFFTW); NumPy is used otherwise.
    """
    N = G.shape[0]
    if FFT:
        return np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(G))) * (N * delta_f) ** 2


def circ(x, y, D):
    """Circular aperture of diameter ``D`` evaluated on coordinate grids."""
    r = np.sqrt(x**2 + y**2)
    return (r < D / 2.0).astype(np.float32)


def corr2_ft(u1, u2, mask, delta):
    """Masked 2-D cross-correlation via FFTs, normalised by the mask overlap.

    With ``u1 == u2 == U`` this is the (unnormalised) mutual coherence
    function of the field over the aperture ``mask``.
    """
    N = u1.shape[0]
    c = np.zeros((N, N), dtype=np.complex64)
    delta_f = 1 / (N * delta)

    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    U12corr = ift2(np.conj(U1) * U2, delta_f)

    maskcorr = ift2(abs(ft2(mask, delta)) ** 2, delta_f) * delta**2
    idx = maskcorr != 0
    c[idx] = U12corr[idx] / maskcorr[idx] * mask[idx]
    return c
