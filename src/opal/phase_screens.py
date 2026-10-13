"""Von Karman phase screens.

Two generators are provided:

* :func:`ft_sh_phase_screen` - a finite FFT screen with five levels of
  sub-harmonics (Schmidt 2010, Lane et al. 1992) used to seed every layer.
* :class:`PhaseScreenVonKarman` - the infinite, non-wrapping screen of
  Assemat et al. (2006) / Fried & Clark (2008). New rows are drawn from the
  conditional Gaussian ``x = A z + B b`` where ``z`` are the two most recent
  rows (the stencil) and ``b ~ N(0, 1)``. The GPU layer
  (:class:`opal.layer.GpuLayer`) runs the same extrusion on the device; this
  class supplies the covariance-derived ``A`` and ``B`` operators and the
  initial screen.
"""

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy
from scipy import linalg
from scipy.special import gamma, kv

from ._fft import ift2

__all__ = [
    "PhaseScreenVonKarman",
    "phase_covariance",
    "clear_kernel_cache",
    "ft_phase_screen",
    "ft_sh_phase_screen",
]


def _covariance_prefactor(r0, L0):
    """r0-dependent scalar of the von Karman covariance."""
    A = (float(L0) / float(r0)) ** (5.0 / 3)
    B1 = (2 ** (-5.0 / 6)) * gamma(11.0 / 6) / (numpy.pi ** (8.0 / 3))
    B2 = ((24.0 / 5) * gamma(6.0 / 5)) ** (5.0 / 6)
    return A * B1 * B2


def _covariance_shape(r, L0):
    """r0-independent part of the von Karman covariance (the Bessel term)."""
    r = numpy.float32(r)
    L0 = float(L0)

    # Avoid the singularity of K_{5/6} at r = 0.
    r += 1e-40

    return (((2 * numpy.pi * r) / L0) ** (5.0 / 6)) * kv(5.0 / 6, (2 * numpy.pi * r) / L0)


def phase_covariance(r, r0, L0):
    """Von Karman phase covariance at separation ``r`` (Assemat et al. 2006, Eq. 5)."""
    return _covariance_prefactor(r0, L0) * _covariance_shape(r, L0)


# The Bessel term over the stencil geometry depends only on (N, dx, L0), not
# on r0, and dominates the operator build time. Every layer of a simulator
# (and every r0 switch of a pass) shares it, so the last few are kept.
_KERNEL_CACHE = {}
_KERNEL_CACHE_SIZE = 2


def _stencil_covariance_shape(key, positions, L0):
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        separations = numpy.zeros((len(positions), len(positions)))
        _calc_separations_fast(positions, separations)
        kernel = _covariance_shape(separations, L0)
        del separations
        kernel.setflags(write=False)
        while len(_KERNEL_CACHE) >= _KERNEL_CACHE_SIZE:
            _KERNEL_CACHE.pop(next(iter(_KERNEL_CACHE)))
        _KERNEL_CACHE[key] = kernel
    return kernel


def clear_kernel_cache():
    """Release the cached covariance kernels (a few tens of MB per grid size)."""
    _KERNEL_CACHE.clear()


# Largest growth per extruded row accepted from the deterministic noise-free
# extrusion test. Valid operators sit just below 1 (0.998-0.99999 depending on
# L0/pixel); float32 round-off at too large an L0/pixel gives 1.01-4.5.
_MAX_STABLE_GROWTH = 1.0002
_STABILITY_ITERATIONS = 2000
# Measured float32 limit of the outer scale in pixels: N = 128-2048 all give
# stable operators up to L0/pixel = 2500; failures start at 3000 (N >= 512),
# 4000 (N = 256) and 5000 (N = 128).
_SAFE_L0_OVER_PIXEL = 2500


def extrusion_growth_rate(A_mat, iterations=_STABILITY_ITERATIONS):
    """Asymptotic growth per extruded row of the noise-free extrusion ``A``.

    Power iteration on the companion map ``(row_0, ..., row_k) ->
    (A @ stencil, row_0, ..., row_{k-1})`` from a fixed random start, in the
    float32 precision the GPU uses. Values above 1 mean the screen diverges
    (the extrusion is unstable). Costs about 0.25 s at N=1024.
    """
    A = numpy.asarray(A_mat, dtype=numpy.float32)
    n = A.shape[0]
    v = numpy.random.default_rng(12345).standard_normal(A.shape[1]).astype(numpy.float32)
    v /= numpy.linalg.norm(v)
    logs = numpy.empty(iterations)
    for k in range(iterations):
        v = numpy.concatenate([A @ v, v[:-n]])
        norm = numpy.linalg.norm(v)
        if not numpy.isfinite(norm) or norm == 0.0:
            return float("inf") if not numpy.isfinite(norm) else 0.0
        v /= norm
        logs[k] = numpy.log(norm)
    return float(numpy.exp(logs[iterations // 2:].mean()))


@numba.jit(nopython=True, parallel=True)
def _calc_separations_fast(positions, separations):
    for i in numba.prange(len(positions)):
        x1, y1 = positions[i]
        for j in range(len(positions)):
            x2, y2 = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1
            separations[i, j] = numpy.sqrt(delta_x**2 + delta_y**2)


class PhaseScreenVonKarman:
    """Infinite von Karman phase screen (NumPy reference implementation).

    Parameters
    ----------
    nx_size : int
        Screen size in pixels (square).
    pixel_scale : float
        Pixel pitch in metres.
    r0 : float
        Fried parameter of this screen in metres.
    L0 : float
        Outer scale in metres.
    random_seed : int or numpy.random.Generator, optional
        Seed of the initial screen. ``None`` draws fresh OS entropy.
    n_columns : int
        Number of stencil rows conditioning each new row (default 2).
    initial_screen : bool
        Generate the initial FFT screen. Set ``False`` when only the ``A``/``B``
        operators are needed (e.g. hard r0 switches during a pass).
    """

    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2,
                 initial_screen=True):
        self.n_columns = n_columns

        self.requested_nx_size = nx_size
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed

        self.set_X_coords()
        self.set_stencil_coords()

        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self._release_scratch()
        if initial_screen:
            self.make_initial_screen()

    def set_X_coords(self):
        self.X_coords = numpy.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = numpy.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        self.stencil = numpy.zeros((self.stencil_length, self.nx_size))
        self.stencil[: self.n_columns] = 1

        self.stencil_coords = numpy.array(numpy.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_separations(self):
        """Pairwise distances between the stencil points and the new row."""
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        self.separations = numpy.zeros((len(positions), len(positions)))
        _calc_separations_fast(positions, self.separations)

    def make_covmats(self):
        """Covariance blocks between the stencil (z) and the new row (x).

        The r0-independent Bessel term is cached per geometry, so building
        several layers (or switching r0 during a pass) computes it once.
        """
        positions = numpy.append(self.stencil_positions, self.X_positions, axis=0)
        key = (
            int(self.nx_size), int(self.stencil_length), int(self.n_columns),
            float(self.pixel_scale), float(self.L0),
        )
        shape = _stencil_covariance_shape(key, positions, self.L0)
        self.cov_mat = _covariance_prefactor(self.r0, self.L0) * shape

        self.cov_mat_zz = self.cov_mat[: self.n_stencils, : self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[: self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, : self.n_stencils]

    def makeAMatrix(self):
        """A = Cov_xz Cov_zz^-1 (Cholesky solve), checked for stability.

        The covariance is evaluated in float32, as in the reference
        implementation. When L0 is large compared with the pixel scale the
        stencil covariance becomes too ill-conditioned for that precision:
        the factorization either fails, or it "succeeds" with an operator whose
        extrusion grows without bound (screens overflow to NaN after a few
        hundred rows). Both cases raise a ``LinAlgError`` that says which
        numbers to change, instead of producing invalid screens.
        """
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
        except linalg.LinAlgError:
            raise self._operator_error("the stencil covariance cannot be factored") from None
        inv_cov_zz = linalg.cho_solve(cf, numpy.identity(self.cov_mat_zz.shape[0]))

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

        growth = extrusion_growth_rate(self.A_mat)
        if not growth <= _MAX_STABLE_GROWTH:
            raise self._operator_error(
                f"the extrusion operator is unstable (rows grow by x{growth:.4g} per extruded row)"
            )

    def _operator_error(self, reason):
        ratio = float(self.L0) / float(self.pixel_scale)
        suggested = float(f"{_SAFE_L0_OVER_PIXEL * float(self.pixel_scale):.2g}")
        return linalg.LinAlgError(
            f"Cannot build the extrusion operators for a layer with r0={self.r0:.4g} m: {reason}. "
            f"The outer scale is too large for the pixel scale (L0={self.L0:g} m, pixel scale "
            f"{self.pixel_scale:g} m, L0/pixel={ratio:.0f}, N={self.nx_size}); the extrusion covariance "
            f"is then numerically singular. Change the numbers: lower L0 (channel_params.L0) to about "
            f"{suggested:g} m or less (L0/pixel <= {_SAFE_L0_OVER_PIXEL}), or use a coarser pixel scale "
            "(channel_params.pixel_size)."
        )

    def makeBMatrix(self):
        """B B^T = Cov_xx - A Cov_zx, factored by SVD."""
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        u, W, ut = numpy.linalg.svd(BBt)

        L_mat = numpy.zeros((self.nx_size, self.nx_size))
        numpy.fill_diagonal(L_mat, numpy.sqrt(W))

        self.B_mat = u.dot(L_mat)

    def _release_scratch(self):
        """Drop the covariance blocks once A and B exist (~110 MB at N=1024)."""
        self.cov_mat = self.cov_mat_zz = self.cov_mat_xx = None
        self.cov_mat_zx = self.cov_mat_xz = None
        self.separations = None
        self.stencil = None

    def release_operators(self):
        """Drop the host copies of ``A``, ``B`` and the screen.

        Used by :class:`opal.layer.GpuLayer` once they live on the GPU. The
        CPU reference :meth:`add_row` needs them, so call this only when the
        object is no longer used for CPU extrusion.
        """
        self.A_mat = None
        self.B_mat = None
        self._scrn = None

    def make_initial_screen(self):
        """Draw the initial screen from an FFT screen with sub-harmonics.

        The same seed always reproduces the same screen.
        """
        self._R = numpy.random.default_rng(self.random_seed)

        self._scrn = ft_sh_phase_screen(
            self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        )

        self._scrn = self._scrn[:, : self.nx_size]

    def get_new_row(self):
        random_data = self._R.normal(0, 1, size=self.nx_size)
        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)
        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """Extrude one row on the CPU (reference path; the GPU layer is used for simulation)."""
        new_row = self.get_new_row()
        self._scrn = numpy.append(new_row, self._scrn, axis=0)[: self.stencil_length, : self.nx_size]
        return self.scrn

    @property
    def scrn(self):
        """Current phase map in radians."""
        return self._scrn[: self.requested_nx_size, : self.requested_nx_size]


def _subharmonic_workers(N, budget_bytes=512 * 2**20):
    """Threads for the sub-harmonic terms, bounded by a transient-memory budget."""
    if N < 256:
        return 1
    per_term = N * N * 16 * 3  # complex128 argument, exponential and product
    return int(max(1, min(9, os.cpu_count() or 1, budget_bytes // per_term)))


def _ordered_map(pool, fn, items, window):
    """``map`` over a thread pool with at most ``window`` results in flight."""
    pending = deque()
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def ft_sh_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """Von Karman FFT phase screen with sub-harmonics (Schmidt 2010, Listing 9.3).

    Parameters
    ----------
    r0 : float
        Fried parameter in metres.
    N : int
        Screen size in pixels.
    delta : float
        Pixel pitch in metres.
    L0, l0 : float
        Outer and inner scale in metres.
    seed : int or numpy.random.Generator, optional
        Seed for a deterministic screen. A ``Generator`` is used in place.

    Returns
    -------
    ndarray
        Phase in radians, shape ``(N, N)``.
    """
    R = numpy.random.default_rng(seed)

    D = N * delta
    phs_hi = ft_phase_screen(r0, N, delta, L0, l0, FFT, seed=seed)

    coords = numpy.arange(-N / 2, N / 2) * delta
    x, y = numpy.meshgrid(coords, coords)

    phs_lo = numpy.zeros(phs_hi.shape)

    # The 9 sub-harmonic terms per level are independent full-size complex
    # exponentials; NumPy releases the GIL, so they are evaluated in threads
    # and summed in the original order (bit-identical to a serial loop).
    workers = _subharmonic_workers(N)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        # Three-by-three frequency grids with spacing 1/(3^p D).
        for p in range(1, 6):
            del_f = 1 / (3**p * D)
            fx = numpy.arange(-1, 2) * del_f

            fx, fy = numpy.meshgrid(fx, fx)
            f = numpy.sqrt(fx**2 + fy**2)

            fm = 5.92 / l0 / (2 * numpy.pi)
            f0 = 1.0 / L0

            PSD_phi = 0.023 * r0 ** (-5.0 / 3) * numpy.exp(-1 * (f / fm) ** 2) / ((f**2 + f0**2) ** (11.0 / 6))
            PSD_phi[1, 1] = 0

            cn = (R.normal(size=(3, 3)) + 1j * R.normal(size=(3, 3))) * numpy.sqrt(PSD_phi) * del_f

            def term(ij, cn=cn, fx=fx, fy=fy):
                i, j = ij
                return cn[i, j] * numpy.exp(1j * 2 * numpy.pi * (fx[i, j] * x + fy[i, j] * y))

            ij = [(i, j) for i in range(0, 3) for j in range(0, 3)]
            SH = numpy.zeros((N, N), dtype="complex")
            terms = _ordered_map(pool, term, ij, workers) if pool is not None else map(term, ij)
            for t in terms:
                SH += t

            phs_lo = phs_lo + SH
    finally:
        if pool is not None:
            pool.shutdown()

    phs_lo = phs_lo.real - phs_lo.real.mean()

    return phs_lo + phs_hi


def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
    """Von Karman FFT phase screen without sub-harmonics (Schmidt 2010, Listing 9.2)."""
    delta = float(delta)
    r0 = float(r0)
    L0 = float(L0)
    l0 = float(l0)

    R = numpy.random.default_rng(seed)

    del_f = 1.0 / (N * delta)

    fx = numpy.arange(-N / 2.0, N / 2.0) * del_f

    (fx, fy) = numpy.meshgrid(fx, fx)
    f = numpy.sqrt(fx**2.0 + fy**2.0)

    fm = 5.92 / l0 / (2 * numpy.pi)
    f0 = 1.0 / L0

    PSD_phi = 0.023 * r0 ** (-5.0 / 3.0) * numpy.exp(-1 * ((f / fm) ** 2)) / (((f**2) + (f0**2)) ** (11.0 / 6))

    PSD_phi[int(N / 2), int(N / 2)] = 0

    cn = (R.normal(size=(N, N)) + 1j * R.normal(size=(N, N))) * numpy.sqrt(PSD_phi) * del_f

    return ift2(cn, 1, FFT).real
