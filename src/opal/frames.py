"""Lazy, memory-safe access to the frames of OPAL output directories.

Every OPAL writer stores complex64 fields ``[T, N, N]`` in ``.npy`` files that
can be memory-mapped. :func:`open_frames` wraps any of them in a
:class:`FrameSeries`, which reads frames from disk only when they are indexed,
so a 100 GiB run can be inspected, analysed or animated with a few hundred MiB
of RAM. Supported sources:

========================  =============================================
``frames.npy``            ``opal run`` (device / host mode)
``frames_chunk_*.npy``    ``opal run`` (chunked mode, via its manifest)
``wavefronts.npy``        ``opal pass`` (captured wavefronts)
``final_frame.npy``       ``opal run`` (no_save mode, one frame)
any ``.npy`` file         ``[T, N, N]`` or ``[N, N]`` complex array
========================  =============================================

This module does not import TensorFlow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "FrameSeries",
    "open_frames",
    "read_manifest",
    "select_indices",
    "is_dataset_dir",
    "load_frames",
    "iter_frames",
    "DEFAULT_BATCH_BYTES",
]

MIB = 1024**2
GIB = 1024**3
DEFAULT_BATCH_BYTES = 256 * MIB


class FrameSeries:
    """Read-only ``[T, N, N]`` frame sequence backed by memory-mapped files.

    Indexing never loads more than it returns:

    * ``fs[k]``            one frame ``[N, N]``
    * ``fs[a:b:s]``        a stacked copy of those frames ``[n, N, N]``
    * ``fs[[i, j, ...]]``  a stacked copy of the listed frames

    Attributes
    ----------
    shape, dtype, nbytes : the logical ``[T, N, N]`` array
    steps : ``int64 [T]`` simulation tick of every frame
    kind : ``"run"``, ``"chunked"``, ``"pass"``, ``"final_frame"`` or ``"npy"``
    manifest : dict, the run's ``manifest.json`` (empty if absent)
    """

    def __init__(self, parts, *, steps=None, kind="npy", source=None, manifest=None):
        if not parts:
            raise ValueError("FrameSeries needs at least one part")
        self._parts = []
        self._files = []
        start = 0
        for arr in parts:
            self._files.append(_file_backing(arr))
            if arr.ndim == 2:
                arr = arr[None]
            if arr.ndim != 3:
                raise ValueError(f"Frames must be [T, N, N] or [N, N], got shape {arr.shape}")
            self._parts.append((start, arr))
            start += int(arr.shape[0])
        first = self._parts[0][1]
        self.shape = (start, int(first.shape[1]), int(first.shape[2]))
        self.dtype = first.dtype
        self.ndim = 3
        self.kind = kind
        self.source = Path(source) if source is not None else None
        self.manifest = manifest or {}
        if steps is None:
            steps = np.arange(start, dtype=np.int64)
        steps = np.asarray(steps, dtype=np.int64)
        if steps.shape[0] != start:
            raise ValueError(f"steps has {steps.shape[0]} entries for {start} frames")
        self.steps = steps

    # ------------------------------------------------------------------
    def __len__(self):
        return self.shape[0]

    @property
    def frame_nbytes(self):
        return int(self.shape[1] * self.shape[2] * self.dtype.itemsize)

    @property
    def nbytes(self):
        return self.frame_nbytes * self.shape[0]

    def __repr__(self):
        src = f" from {self.source}" if self.source else ""
        return (f"FrameSeries({self.kind}, {self.shape[0]} x {self.shape[1]} x {self.shape[2]} "
                f"{self.dtype}, {self.nbytes / GIB:.2f} GiB on disk{src})")

    def _locate(self, t):
        t = int(t)
        if t < 0:
            t += self.shape[0]
        if not 0 <= t < self.shape[0]:
            raise IndexError(f"frame {t} out of range [0, {self.shape[0]})")
        for k in range(len(self._parts) - 1, -1, -1):
            start = self._parts[k][0]
            if t >= start:
                return k, t - start
        raise IndexError(t)

    def _read_into(self, out, t, handles):
        # Memory-mapped files are read with plain positioned reads: mapped pages
        # would otherwise pile up in the process working set (up to the whole
        # file) even though only one batch is used at a time.
        k, local = self._locate(t)
        backing = self._files[k]
        if backing is None:
            out[...] = self._parts[k][1][local]
            return
        filename, offset = backing
        f = handles.get(filename)
        if f is None:
            f = handles[filename] = open(filename, "rb")
        f.seek(offset + local * self.frame_nbytes)
        if f.readinto(out.reshape(-1).view(np.uint8)) != self.frame_nbytes:
            raise OSError(f"{filename} is truncated (frame {t})")

    def _one(self, t):
        out = np.empty(self.shape[1:], dtype=self.dtype)
        handles = {}
        try:
            self._read_into(out, t, handles)
        finally:
            for f in handles.values():
                f.close()
        return out

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._one(idx)
        if isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        if idx.ndim != 1:
            raise IndexError("FrameSeries supports int, slice, or 1-D index arrays")
        out = np.empty((len(idx), self.shape[1], self.shape[2]), dtype=self.dtype)
        handles = {}
        try:
            for k, t in enumerate(idx):
                self._read_into(out[k], t, handles)
        finally:
            for f in handles.values():
                f.close()
        return out

    def __iter__(self):
        for t in range(self.shape[0]):
            yield self._one(t)

    def __array__(self, dtype=None, copy=None):
        # Materialises everything: only for small series. Prefer iter_batches.
        out = self[:]
        return out.astype(dtype) if dtype is not None else out

    def batch_size_for(self, max_bytes=DEFAULT_BATCH_BYTES, bytes_per_frame=None):
        """Frames per batch so one batch stays below ``max_bytes``."""
        per = int(bytes_per_frame or self.frame_nbytes)
        return max(1, int(max_bytes // max(1, per)))

    def iter_batches(self, indices=None, batch_size=None, max_bytes=DEFAULT_BATCH_BYTES):
        """Yield ``(indices, frames)`` batches; only one batch is in RAM at a time."""
        if indices is None:
            indices = np.arange(self.shape[0])
        indices = np.asarray(indices, dtype=np.int64)
        b = int(batch_size) if batch_size else self.batch_size_for(max_bytes)
        for i in range(0, len(indices), b):
            sel = indices[i:i + b]
            yield sel, self[sel]


# ----------------------------------------------------------------------
def _file_backing(arr):
    """``(filename, data_offset)`` when ``arr`` is a whole, plain ``.npy`` memmap."""
    if (type(arr) is np.memmap and getattr(arr, "filename", None) and arr.flags.c_contiguous
            and arr.dtype.isnative and not arr.dtype.hasobject):
        base = arr
        while isinstance(base.base, np.memmap):
            base = base.base
        if base is not arr and base.size != arr.size:
            return None
        return str(arr.filename), int(arr.offset)
    return None


def read_manifest(path):
    """``manifest.json`` of a run / pass / dataset directory, or ``{}``."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    m = path / "manifest.json"
    if m.exists():
        return json.loads(m.read_text(encoding="utf-8"))
    return {}


def is_dataset_dir(path):
    path = Path(path)
    return path.is_dir() and (path / "samples").is_dir() and (path / "labels.csv").exists()


def open_frames(path):
    """Open the frames of a run directory, pass directory or ``.npy`` file lazily.

    Raises ``FileNotFoundError`` with a hint when the directory holds no frames
    (e.g. a ``no_save`` run without ``final_frame.npy`` or a dataset).
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"{path} is not a .npy file (dataset samples are opened with opal.viewer)")
        arr = np.load(path, mmap_mode="r")
        return FrameSeries([arr], kind="npy", source=path, manifest=read_manifest(path))
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist")

    manifest = read_manifest(path)
    start_step = int(manifest.get("start_step", 0) or 0)

    single = path / "frames.npy"
    if single.exists():
        arr = np.load(single, mmap_mode="r")
        return FrameSeries([arr], steps=start_step + np.arange(arr.shape[0]), kind="run",
                           source=path, manifest=manifest)

    wf = path / "wavefronts.npy"
    if wf.exists():
        arr = np.load(wf, mmap_mode="r")
        steps_file = path / "wavefront_steps.npy"
        steps = np.load(steps_file) if steps_file.exists() else None
        return FrameSeries([arr], steps=steps, kind="pass", source=path, manifest=manifest)

    chunk_manifest = path / "frames_chunk_manifest.json"
    if chunk_manifest.exists():
        chunks = json.loads(chunk_manifest.read_text(encoding="utf-8"))["chunks"]
        if chunks and all(c.get("file") for c in chunks):
            chunks = sorted(chunks, key=lambda c: c["start_step"])
            parts = [np.load(path / Path(c["file"]).name, mmap_mode="r") for c in chunks]
            steps = np.concatenate([start_step + np.arange(c["start_step"], c["end_step_exclusive"])
                                    for c in chunks])
            return FrameSeries(parts, steps=steps, kind="chunked", source=path, manifest=manifest)

    final = path / "final_frame.npy"
    if final.exists():
        arr = np.load(final, mmap_mode="r")
        end = manifest.get("end_step")
        steps = [int(end) - 1] if end is not None else None
        return FrameSeries([arr], steps=steps, kind="final_frame", source=path, manifest=manifest)

    if is_dataset_dir(path):
        raise FileNotFoundError(
            f"{path} is a dataset, not a run: its per-frame fiber coupling, scintillation, tilt and r0 "
            f"are already stored as labels ({Path(path) / 'labels.csv'} and each sample's .npz). "
            "Browse it with `opal view DIR` or load it with opal.load_dataset / opal.load_sample.")
    raise FileNotFoundError(
        f"No frames found in {path}. Expected frames.npy, frames_chunk_*.npy, wavefronts.npy or "
        "final_frame.npy (runs started with --no-save keep nothing on disk)."
    )


def select_indices(total, frames=None, *, stride=1, start=None, stop=None):
    """Frame indices to process.

    Takes ``range(start, stop, stride)`` and, when ``frames`` is an integer
    smaller than that range, picks ``frames`` evenly spaced indices from it.
    ``frames=None`` or ``"all"`` keeps the whole range.
    """
    stride = max(1, int(stride or 1))
    rng = np.arange(total)[slice(start, stop, stride)]
    if frames is None or (isinstance(frames, str) and frames.lower() == "all"):
        return rng
    frames = int(frames)
    if frames < 1:
        raise ValueError("frames must be >= 1 or 'all'")
    if frames >= len(rng):
        return rng
    pick = np.unique(np.linspace(0, len(rng) - 1, frames).round().astype(int))
    return rng[pick]


# ----------------------------------------------------------------------
def load_frames(run_dir, mmap=True):
    """Frames of a run directory as a ``[T, N, N]`` array-like.

    With ``mmap=True`` (default) nothing is read up front: single-file runs
    return a read-only ``np.memmap`` and chunked runs a lazy
    :class:`FrameSeries`. ``mmap=False`` loads everything into RAM.
    """
    fs = open_frames(run_dir)
    if not mmap:
        return fs[:]
    if len(fs._parts) == 1 and fs.kind in ("run", "pass", "npy"):
        return fs._parts[0][1]
    return fs


def iter_frames(run_dir, stride=1):
    """Yield ``(index, frame)`` from any run directory without loading it all.

    ``index`` counts frames from 0 within the directory; the matching
    simulation tick is ``open_frames(run_dir).steps[index]``.
    """
    fs = open_frames(run_dir)
    for i in range(0, len(fs), max(1, int(stride))):
        yield i, fs[i]
