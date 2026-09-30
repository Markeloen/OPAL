"""Angular-spectrum split-step propagation through the layer stack.

Implements the multi-step angular-spectrum propagator of Schmidt (2010,
Listing 9.1) for a downlink: the field starts above the top layer and each
partial propagation is followed by the next layer's phase screen and a
super-Gaussian absorbing boundary. All operators are precomputed once and kept
on the device as constants.
"""

from math import cos

import numpy as np
import tensorflow as tf

from . import _tfsetup  # noqa: F401
from ._fft import circ, corr2_ft, ft2, ift2

__all__ = [
    "Propagator",
    "tf_ft2",
    "tf_ift2",
    "tf_rect",
    "ft2",
    "ift2",
    "circ",
    "corr2_ft",
]


def tf_ft2(g, delta):
    """Centred, scaled 2-D FFT on the device."""
    g = tf.cast(g, tf.complex64)
    delta = tf.complex(tf.cast(delta, dtype=tf.float32), 0.0)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(g))) * delta**2


def tf_ift2(g, delta_f):
    """Centred, scaled inverse 2-D FFT on the device (square grids)."""
    N = tf.cast(tf.shape(g)[0], tf.float32)
    g = tf.cast(g, tf.complex64)
    delta_f = tf.complex(tf.cast(delta_f, dtype=tf.float32), 0.0)
    N_complex = tf.complex(N, 0.0)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N_complex * delta_f) ** 2


def tf_rect(x, width):
    """Rectangle function ``|x| <= width / 2``."""
    return tf.cast(tf.abs(x) <= (width / 2), tf.float32)


class Propagator:
    """Precomputed angular-spectrum propagator for a fixed layer geometry.

    Parameters
    ----------
    N : int
        Grid size (pixels).
    wvl : float
        Wavelength (m).
    delta1, deltan : float
        Source- and observation-plane grid spacing (m).
    z : sequence of float
        Layer heights (m), ground to top.
    zenith : float
        Zenith angle (deg) applied to the partial propagation distances.
    """

    def __init__(self, N, wvl, delta1, deltan, z, zenith=0):
        self.N = N
        x = np.arange(-N / 2.0, N / 2.0)
        [nx, ny] = np.meshgrid(x, x)
        k = 2 * np.pi / wvl
        nsq = nx**2 + ny**2

        # Super-Gaussian absorbing boundary.
        w = 0.47 * N
        self.sg = np.exp(-nsq**8 / w**16)

        # Downlink: measure distances from just above the top layer.
        reversed_z = z[::-1]
        new_z = [((z[-1] + 1) - x) * (1 / cos(np.deg2rad(zenith))) for x in reversed_z]
        z = np.array([0, *new_z])
        self.n = len(z)
        Delta_z = z[1:] - z[: self.n - 1]
        alpha = z / z[-1]
        self.delta = (1 - alpha) * delta1 + alpha * deltan
        m = self.delta[1:] / self.delta[: self.n - 1]

        x1 = nx * self.delta[0]
        y1 = ny * self.delta[0]
        r1sq = x1**2 + y1**2

        self.Q1 = np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * r1sq)
        self.Q2 = []
        self.m = m
        self.deltaf = []

        for idx in range(self.n - 1):
            deltaf = 1 / (N * self.delta[idx])
            self.deltaf.append(deltaf)

            fX = nx * deltaf
            fY = ny * deltaf
            fsq = fX**2 + fY**2
            Z = Delta_z[idx]

            Q2 = np.exp(-1j * (np.pi**2) * 2 * Z / (m[idx] * k) * fsq)
            self.Q2.append(Q2)

        self.Q2 = np.asarray(self.Q2)
        self.deltaf = np.asarray(self.deltaf)
        self.xn = nx * self.delta[-1]
        self.yn = ny * self.delta[-1]
        rnsq = self.xn**2 + self.yn**2
        self.Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * Z) * rnsq)

        self.tf_Q1 = tf.constant(self.Q1, dtype=tf.complex64)
        self.tf_Q2 = tf.constant(self.Q2, dtype=tf.complex64)
        self.tf_Q3 = tf.constant(self.Q3, dtype=tf.complex64)
        self.tf_m = tf.constant(self.m, dtype=tf.float32)
        self.tf_deltaf = tf.constant(self.deltaf, dtype=tf.float32)
        self.tf_N = tf.constant(self.N, dtype=tf.float32)
        self.tf_sg = tf.constant(self.sg, dtype=tf.float32)
        self.tf_delta = tf.constant(self.delta, dtype=tf.float32)

    @tf.function
    def propagate(self, Uin, PS_arr):
        """Propagate ``Uin`` through the phase screens ``PS_arr`` (ground to top).

        Returns
        -------
        (Uout, xn, yn)
            Received field and observation-plane coordinates.
        """
        PS_arr = PS_arr[::-1]
        PS_arr_complex = tf.complex(0.0, tf.convert_to_tensor(PS_arr, dtype=tf.float32))

        Uin = self.tf_Q1 * tf.cast(Uin, tf.complex64)

        for idx in range(self.n - 1):
            ff = tf_ift2(
                self.tf_Q2[idx] * tf_ft2(Uin / tf.complex(self.tf_m[idx], 0.0), self.tf_delta[idx]),
                self.tf_deltaf[idx],
            )
            Uin = tf.math.multiply(tf.complex(self.tf_sg, 0.0), tf.exp(PS_arr_complex[idx])) * ff

        Uout = self.tf_Q3 * Uin
        return Uout, self.xn, self.yn
