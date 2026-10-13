"""Labelled, reproducible turbulence datasets for machine learning.

:func:`generate_dataset` sweeps channel parameters (Cartesian product of the
given values), draws several independent realizations per parameter point and
streams every simulated field from the GPU to host memory. For each sample it
stores a time series of fields (optionally strided and centre-cropped) as the
requested targets, together with per-frame physical labels:

* ``eta``   single-mode fiber coupling efficiency,
* ``scint`` aperture scintillation index,
* ``tilt``  angle of arrival (far-field centroid, rad),

and per-sample summary labels in ``labels.csv`` (parameters, seed, D/r0,
measured path r0, mean/std of the per-frame labels). Every sample is
regenerable from ``manifest.json`` alone: point ``p``, realization ``r`` uses
seed ``SeedSequence([base_seed, p, r])``.

Layout::

    out_dir/
      manifest.json         config, sweep, seeds, targets, timing; "status" is
                            "incomplete" until every sample is written
      labels.csv            one row per sample
      samples/sample_00000.npz
        field | intensity | phase   [F, n, n]
        steps                       [F]       tick index of each frame
        eta, scint                  [F]
        tilt                        [F, 2]    (theta_x, theta_y) rad
"""

from __future__ import annotations

import csv
import datetime as _dt
import gc
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from . import _timing
from ._version import __version__
from .analysis.fiber import compute_fiber_coupling
from .analysis.r0 import estimate_r0_from_wavefront
from .analysis.stats import aperture_mask, scintillation_index, tilt_centroid
from .config import ALIASES, get_value, load_config, rename_legacy_keys, set_value, to_plain

__all__ = [
    "parse_sweep",
    "sweep_points",
    "sample_seed",
    "generate_dataset",
    "load_sample",
    "iter_samples",
    "load_labels",
    "load_dataset",
    "as_tf_dataset",
]

TARGETS = ("field", "intensity", "phase")
_LABEL_PARAMS = ("r0", "wind", "zenith", "layers", "N", "dx", "wavelength", "L0", "dt", "orbit")


def parse_sweep(spec):
    """Parse ``"key=v1,v2,..."`` or ``"key=lo:hi:n"`` (linear) into ``(key, values)``.

    ``key`` is any config alias (``r0``, ``wind``, ``zenith``, ``L0``, ...) or
    ``elevation`` (deg), which is converted to the zenith angle.
    """
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return spec[0], [float(v) for v in spec[1]]
    if "=" not in spec:
        raise ValueError(f"Sweep {spec!r} must look like key=v1,v2 or key=lo:hi:n")
    key, rhs = (s.strip() for s in spec.split("=", 1))
    if key != "elevation" and key not in ALIASES and "." not in key:
        raise ValueError(f"Unknown sweep parameter {key!r}. Known: elevation, {', '.join(sorted(ALIASES))}")
    if ":" in rhs:
        parts = rhs.split(":")
        if len(parts) != 3:
            raise ValueError(f"Range sweep {spec!r} must be lo:hi:n")
        values = np.linspace(float(parts[0]), float(parts[1]), int(parts[2])).tolist()
    else:
        values = [float(v) for v in rhs.split(",") if v.strip()]
    if not values:
        raise ValueError(f"Sweep {spec!r} has no values")
    return key, values


def sweep_points(sweeps):
    """Cartesian product of parsed sweeps -> list of ``{key: value}`` dicts."""
    parsed = [parse_sweep(s) for s in (sweeps or [])]
    if not parsed:
        return [{}]
    keys = [k for k, _ in parsed]
    return [dict(zip(keys, combo)) for combo in itertools.product(*[v for _, v in parsed])]


def sample_seed(base_seed, point_index, realization):
    """Seed of realization ``realization`` at sweep point ``point_index``."""
    state = np.random.SeedSequence([int(base_seed), int(point_index), int(realization)]).generate_state(1)
    return int(state[0] & 0x7FFFFFFF)


def _apply_point(config, point):
    cfg = json.loads(json.dumps(config))
    for key, value in point.items():
        if key == "elevation":
            set_value(cfg, "zenith", 90.0 - float(value))
        else:
            set_value(cfg, key, value)
    return load_config(cfg)


def _center_crop(frame, crop):
    if not crop:
        return frame
    n = frame.shape[-1]
    lo = (n - crop) // 2
    return frame[..., lo:lo + crop, lo:lo + crop]


def _point_params(cfg):
    out = {k: get_value(cfg, k) for k in _LABEL_PARAMS}
    out["elevation"] = 90.0 - out["zenith"]
    return out


class _SampleWriter:
    """Collects strided frames of one realization and derives its labels."""

    def __init__(self, T, stride, crop, targets, N, dx, wvl, label_aperture, r0_frames):
        self.stride, self.crop, self.targets = stride, crop, targets
        self.keep = np.arange(0, T, stride)
        self.slot = {int(s): i for i, s in enumerate(self.keep)}
        F = len(self.keep)
        n = crop or N
        self.out = {}
        if "field" in targets:
            self.out["field"] = np.empty((F, n, n), np.complex64)
        if "intensity" in targets:
            self.out["intensity"] = np.empty((F, n, n), np.float32)
        if "phase" in targets:
            self.out["phase"] = np.empty((F, n, n), np.float32)
        self.eta = np.empty(F, np.float64)
        self.scint = np.empty(F, np.float64)
        self.tilt = np.empty((F, 2), np.float64)
        self.dx, self.wvl, self.aperture = dx, wvl, label_aperture
        self.mask = aperture_mask(N, dx, label_aperture)
        r0_idx = np.unique(np.linspace(0, F - 1, max(1, min(r0_frames, F))).round().astype(int))
        self.r0_slots = set(int(i) for i in r0_idx)
        self.r0_frames = []

    def __call__(self, step, frame):
        i = self.slot.get(int(step))
        if i is None:
            return
        full = np.asarray(frame)
        view = _center_crop(full, self.crop)
        if "field" in self.out:
            self.out["field"][i] = view
        if "intensity" in self.out:
            self.out["intensity"][i] = np.abs(view) ** 2
        if "phase" in self.out:
            self.out["phase"][i] = np.angle(view)
        self.eta[i] = compute_fiber_coupling(full, self.dx, aperture_diameter=self.aperture)["eta"][0]
        self.scint[i] = scintillation_index(full, self.mask)
        self.tilt[i] = tilt_centroid(full, self.dx, self.wvl, mask=self.mask)
        if i in self.r0_slots:
            self.r0_frames.append(full.copy())

    def measured_r0(self, L0):
        vals = [estimate_r0_from_wavefront(f, self.dx, None, L0=L0)["measured_r0"] for f in self.r0_frames]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else math.nan


def generate_dataset(
    config,
    out_dir,
    *,
    sweep=None,
    realizations=1,
    num_steps=None,
    stride=1,
    crop=None,
    targets=("intensity", "phase"),
    seed=None,
    label_aperture=None,
    r0_frames=4,
    compress=False,
    resume=True,
    log=print,
):
    """Generate a labelled dataset of time-correlated turbulent fields.

    Parameters
    ----------
    config : dict or path
        Base configuration; ``sweep`` values override it per point.
    out_dir : path
        Dataset directory (created).
    sweep : list of str, optional
        e.g. ``["r0=0.05:0.2:4", "wind=5,10"]`` (Cartesian product).
    realizations : int
        Independent realizations per sweep point.
    num_steps : int, optional
        Ticks per sample (default: the config's ``num_steps``).
    stride : int
        Keep every ``stride``-th tick.
    crop : int, optional
        Centre-crop stored frames to ``crop x crop`` (labels always use the full grid).
    targets : sequence of {"field", "intensity", "phase"}
    seed : int, optional
        Base seed (default: config seed, else random). Recorded in the manifest.
    label_aperture : float, optional
        Receiver aperture (m) for the eta/scint/tilt labels.
        Default ``0.4 * N * dx`` (inside the absorbing boundary).
    r0_frames : int
        Frames per sample used for the measured-r0 label (0 disables it).
    compress : bool
        Use ``np.savez_compressed`` for samples.
    resume : bool
        Skip samples whose file already exists (seeds are deterministic). Raises
        ``ValueError`` if ``out_dir`` holds a dataset generated with different
        settings. ``False`` deletes existing samples first.
    """
    from .runners import warn_if_no_gpu
    from .simulator import GpuSimulator

    t0 = _timing.clock()
    t_start = time.time()

    def say(msg):
        if log is not None:
            log(f"[opal {time.time() - t0:7.2f}s] {msg}")

    warn_if_no_gpu(say)

    base = load_config(config)
    targets = tuple(targets)
    bad = [t for t in targets if t not in TARGETS]
    if bad or not targets:
        raise ValueError(f"targets must be a non-empty subset of {TARGETS}, got {targets}")
    stride = max(1, int(stride))
    realizations = max(1, int(realizations))
    T = int(num_steps if num_steps is not None else get_value(base, "steps"))
    if seed is None:
        seed = get_value(base, "seed")
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    seed = int(seed)

    points = sweep_points(sweep)
    parsed = [parse_sweep(s) for s in (sweep or [])]
    total = len(points) * realizations
    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    saver = np.savez_compressed if compress else np.savez
    n_frames = len(range(0, T, stride))

    manifest = {
        "opal_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "status": "incomplete",
        "base_seed": seed,
        "seed_rule": "SeedSequence([base_seed, point, realization]).generate_state(1)[0] & 0x7FFFFFFF",
        "config": to_plain(base),
        "sweep": {k: v for k, v in parsed},
        "points": points,
        "realizations": realizations,
        "num_samples": total,
        "num_steps": T,
        "stride": stride,
        "crop": int(crop) if crop else None,
        "targets": list(targets),
        "per_frame_labels": {"eta": "fiber coupling efficiency", "scint": "scintillation index",
                             "tilt": "angle of arrival (rad), far-field centroid"},
        "label_aperture_m": label_aperture,
        "r0_frames": int(r0_frames),
    }
    manifest_path = out_dir / "manifest.json"
    existing = sorted(samples_dir.glob("sample_*.npz")) if samples_dir.is_dir() else []
    if resume and manifest_path.exists():
        diff = _settings_diff(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
        if diff:
            raise ValueError(
                f"{out_dir} already holds a dataset generated with different settings ({'; '.join(diff)}). "
                "Resuming would mix incompatible samples: re-run with the original settings, pass "
                "--no-resume (resume=False) to replace it, or choose another output directory.")
    if not resume and existing:
        for path in existing:
            path.unlink()
        say(f"--no-resume: removed {len(existing)} existing samples from {samples_dir}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    say(f"dataset: {len(points)} sweep points x {realizations} realizations = {total} samples, "
        f"{T} ticks (stride {stride}, {n_frames} frames), targets {', '.join(targets)}")

    rows = []
    sim_time = 0.0
    resumed = 0
    idx = 0
    for p, point in enumerate(points):
        cfg = _apply_point(base, point)
        N = int(get_value(cfg, "N"))
        dx = float(get_value(cfg, "dx"))
        if crop and (int(crop) > N or int(crop) % 2):
            raise ValueError(f"crop must be an even integer <= N ({N}), got {crop}")
        aperture = float(label_aperture) if label_aperture else 0.4 * N * dx
        params = _point_params(cfg)
        sim = None
        for r in range(realizations):
            s = sample_seed(seed, p, r)
            path = samples_dir / f"sample_{idx:05d}.npz"
            if resume and path.exists():
                with np.load(path) as data:
                    row = json.loads(str(data["labels_json"]))
                    stored = set(data.files)
                expected = json.loads(json.dumps({
                    "seed": s, "frames": n_frames, "stride": stride, "crop": int(crop) if crop else N,
                    "label_aperture_m": aperture, **params, **{f"sweep_{k}": v for k, v in point.items()}}))
                bad = [f"{k} {row.get(k)!r} != {v!r}" for k, v in expected.items() if row.get(k) != v]
                bad += [f"target {t} missing" for t in targets if t not in stored]
                if bad:
                    raise ValueError(
                        f"{path} was generated with different settings ({'; '.join(bad[:4])}). Resuming "
                        "would mix incompatible samples: re-run with the original settings, pass "
                        "--no-resume (resume=False) to replace them, or choose another output directory.")
                rows.append(row)
                resumed += 1
                say(f"sample {idx + 1}/{total} exists, skipped")
                idx += 1
                continue
            if sim is None:
                say(f"building simulator: N={N}, {int(get_value(cfg, 'layers'))} layers")
                tb = time.time()
                sim = GpuSimulator(cfg, seed=s, progress=_timing.layer_progress(say, N))
                say(f"built simulator in {time.time() - tb:.1f} s (reused for every realization "
                    "of this sweep point via reseed)")
            else:
                sim.reseed(s)
            writer = _SampleWriter(T, stride, int(crop) if crop else None, targets, N, dx,
                                   float(params["wavelength"]), aperture, int(r0_frames))
            sched = np.ones((1, sim.num_layers), np.float32)
            start = time.time()
            sim.simulate_steps_offload(T, sched, frame_callback=writer)
            elapsed = time.time() - start
            sim_time += elapsed
            r0m = writer.measured_r0(params["L0"]) if r0_frames else math.nan
            tilt_mag = np.hypot(writer.tilt[:, 0], writer.tilt[:, 1])
            row = {
                "sample": idx,
                "file": f"samples/{path.name}",
                "point": p,
                "realization": r,
                "seed": s,
                **{f"sweep_{k}": v for k, v in point.items()},
                **params,
                "frames": int(len(writer.keep)),
                "stride": stride,
                "crop": int(crop) if crop else N,
                "label_aperture_m": aperture,
                "D_over_r0": aperture / params["r0"],
                "r0_measured": r0m,
                "eta_mean": float(np.nanmean(writer.eta)),
                "eta_std": float(np.nanstd(writer.eta)),
                "scint_mean": float(np.nanmean(writer.scint)),
                "tilt_rms_rad": float(np.sqrt(np.nanmean(tilt_mag**2))),
                "sim_time_s": elapsed,
            }
            saver(path, steps=writer.keep, eta=writer.eta, scint=writer.scint, tilt=writer.tilt,
                  labels_json=np.array(json.dumps(row)), **writer.out)
            rows.append(row)
            desc = ", ".join(f"{k}={v:g}" for k, v in point.items()) or "base"
            say(f"sample {idx + 1}/{total} [{desc}] seed {s}: {elapsed:.2f} s | "
                f"r0 {r0m:.3f} m | eta {row['eta_mean']:.3f} | SI {row['scint_mean']:.3f}")
            idx += 1
        del sim
        gc.collect()

    fields = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in fields:
                fields.append(k)
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    manifest.update(
        status="complete",
        samples_resumed=resumed,
        simulation_time_s=sim_time,
        wall_time_s=time.time() - t_start,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    say(f"done: {total} samples in {time.time() - t_start:.1f} s (GPU simulation {sim_time:.1f} s"
        + (f", {resumed} existing samples reused" if resumed else "") + f") -> {out_dir}")
    return manifest


_RESUME_KEYS = ("base_seed", "config", "sweep", "points", "realizations", "num_steps", "stride", "crop",
                "targets", "label_aperture_m", "r0_frames")


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): value}


def _settings_diff(old, new):
    """Human-readable differences of the settings that determine the samples."""
    norm = lambda v: json.loads(json.dumps(v))  # noqa: E731 - tuples -> lists, numpy scalars -> numbers
    diffs = []
    for key in _RESUME_KEYS:
        if key not in old:
            continue
        a, b = norm(old[key]), norm(new[key])
        if key == "config" and isinstance(a, dict):
            a = rename_legacy_keys(a)
        if a == b:
            continue
        if key == "config":
            fa, fb = _flatten(a), _flatten(b)
            keys = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
            diffs += [f"config {k}: {fa.get(k)!r} -> {fb.get(k)!r}" for k in keys[:4]]
        else:
            diffs.append(f"{key}: {a!r} -> {b!r}"[:120])
    return diffs


def _read_manifest(ds_dir):
    manifest = json.loads((Path(ds_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") == "incomplete":
        raise ValueError(
            f"{ds_dir} is an incomplete dataset (generation was interrupted or is still running); re-run "
            "the same `opal dataset` command to resume it, or load single files with load_sample().")
    return manifest


def load_labels(ds_dir):
    """Read ``labels.csv`` as a list of dicts with numeric values converted."""
    rows = []
    with (Path(ds_dir) / "labels.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            conv = {}
            for k, v in row.items():
                try:
                    f = float(v)
                    conv[k] = int(f) if f.is_integer() and k in ("sample", "point", "realization", "seed",
                                                                  "frames", "stride", "crop", "layers", "N") else f
                except (TypeError, ValueError):
                    conv[k] = v
            rows.append(conv)
    return rows


def load_sample(ds_dir, index):
    """Load one sample as a dict of arrays (plus its ``labels`` dict)."""
    with np.load(Path(ds_dir) / "samples" / f"sample_{int(index):05d}.npz") as data:
        out = {k: data[k] for k in data.files if k != "labels_json"}
        out["labels"] = json.loads(str(data["labels_json"]))
    return out


def iter_samples(ds_dir):
    """Yield every sample in index order."""
    manifest = _read_manifest(ds_dir)
    for i in range(int(manifest["num_samples"])):
        yield load_sample(ds_dir, i)


def _check_request(ds_dir, manifest, target, label_keys, per_frame_keys=()):
    stored = manifest.get("targets") or list(TARGETS)
    if target not in stored:
        raise ValueError(f"target {target!r} is not stored in {ds_dir} (it has: {', '.join(stored)}); "
                         "generate the dataset with --targets including it")
    labels = set(load_sample(ds_dir, 0)["labels"]) | set(per_frame_keys)
    missing = [k for k in label_keys if k not in labels]
    if missing:
        raise ValueError(f"unknown label key(s) {missing}; available: {', '.join(sorted(labels))}")


def load_dataset(ds_dir, target="intensity", label_keys=("r0", "eta_mean")):
    """Stack a whole dataset in memory: ``X [S, F, n, n]`` and ``y [S, len(label_keys)]``."""
    _check_request(ds_dir, _read_manifest(ds_dir), target, label_keys)
    xs, ys = [], []
    for s in iter_samples(ds_dir):
        xs.append(s[target])
        ys.append([float(s["labels"][k]) for k in label_keys])
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def as_tf_dataset(ds_dir, target="intensity", label_keys=("r0",), per_frame=False):
    """Stream a dataset as ``tf.data.Dataset`` of ``(x, y)``.

    ``per_frame=False`` yields whole sequences ``[F, n, n]``; ``True`` yields
    single frames with the per-frame labels (``eta``, ``scint``, ``tilt_x``,
    ``tilt_y``) or sample labels.
    """
    import tensorflow as tf

    manifest = _read_manifest(ds_dir)
    per_frame_keys = {"eta", "scint", "tilt_x", "tilt_y"}
    _check_request(ds_dir, manifest, target, label_keys, per_frame_keys if per_frame else ())
    first = load_sample(ds_dir, 0)
    x_dtype = tf.complex64 if target == "field" else tf.float32
    shape = first[target].shape

    def label_row(sample, f=None):
        vals = []
        for k in label_keys:
            if f is not None and k in per_frame_keys:
                if k.startswith("tilt_"):
                    vals.append(sample["tilt"][f, 0 if k == "tilt_x" else 1])
                else:
                    vals.append(sample[k][f])
            else:
                vals.append(sample["labels"][k])
        return np.asarray(vals, np.float32)

    def gen():
        for i in range(int(manifest["num_samples"])):
            s = load_sample(ds_dir, i)
            if per_frame:
                for f in range(s[target].shape[0]):
                    yield s[target][f], label_row(s, f)
            else:
                yield s[target], label_row(s)

    x_shape = shape[1:] if per_frame else shape
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec(shape=x_shape, dtype=x_dtype),
                          tf.TensorSpec(shape=(len(label_keys),), dtype=tf.float32)),
    )
