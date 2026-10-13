"""Analysis of saved runs: the engine behind ``opal analyze``.

:func:`analyze_run` reads frames lazily (one bounded batch at a time, see
:mod:`opal.frames`), so it works on runs of any size, and writes its results
next to the run::

    <run>/analysis/
      fiber.csv            frame, step, eta[, target_r0_m, eta_theory]
      fiber_summary.json   settings, statistics, theory, timing
      fiber.png            time series + histogram (with plot=True)
      r0.csv               frame, step, r0_m[, target_r0_m]
      r0_summary.json
      r0.png

This module does not import TensorFlow.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import time
from pathlib import Path

import numpy as np

from .._version import __version__
from ..frames import DEFAULT_BATCH_BYTES, GIB, MIB, open_frames, select_indices
from .fiber import compute_fiber_coupling, coupling_efficiency_theoretical_vk
from .r0 import estimate_r0_from_wavefront

__all__ = ["analyze_run", "load_analysis", "ANALYSES"]

ANALYSES = ("fiber", "r0")
NOTE = ("frames of one run are correlated in time; for ensemble statistics use independent "
        "realizations (opal dataset -k K or GpuSimulator.reseed)")


def _pass_targets(run_dir):
    """``{step: theoretical_r0}`` from a pass directory's ``r0_tracking.csv``."""
    path = Path(run_dir) / "r0_tracking.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return {int(r["step"]): float(r["theoretical_r0"]) for r in csv.DictReader(fh)}


def _schedule_targets(fs, steps):
    """Target path r0 per frame for a run made with a turbulence schedule.

    Uses the applied ``schedule.npy`` (b per tick and layer) and the per-layer
    r0 in the manifest: ``r0_i(t) = r0_i * b_i(t)^(-6/5)``, combined over
    layers as ``(sum r0_i^(-5/3))^(-3/5)``. ``None`` when the run had no schedule.
    """
    src = fs.source if fs.source is not None and Path(fs.source).is_dir() else None
    layers = ((fs.manifest or {}).get("channel") or {}).get("layers")
    if src is None or not layers or not (src / "schedule.npy").exists():
        return None
    sched = np.load(src / "schedule.npy").astype(np.float64)
    base = np.array([float(l["r0_m"]) for l in layers], dtype=np.float64)
    if sched.ndim != 2 or sched.shape[1] != base.size:
        return None
    rel = np.clip(np.asarray(steps) - int(fs.manifest.get("start_step", 0) or 0), 0, sched.shape[0] - 1)
    with np.errstate(divide="ignore", over="ignore"):
        inv = np.where(np.isfinite(base) & (base > 0), base ** (-5.0 / 3.0), 0.0)[None, :] * sched[rel] ** 2
        total = inv.sum(axis=1)
        return np.where(total > 0, total ** (-3.0 / 5.0), math.inf)


def _write_csv(path, header, columns):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in zip(*columns):
            w.writerow([f"{v:.8g}" if isinstance(v, (float, np.floating)) else int(v) for v in row])


def _stats(values):
    ok = values[np.isfinite(values)]
    if ok.size == 0:
        return {"count": 0}
    return {"count": int(ok.size), "mean": float(ok.mean()), "std": float(ok.std()),
            "median": float(np.median(ok)), "min": float(ok.min()), "max": float(ok.max()),
            "p05": float(np.percentile(ok, 5)), "p95": float(np.percentile(ok, 95))}


def _plot(path, what, steps, values, reference, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {"fiber": "coupling efficiency η", "r0": "measured r₀ (m)"}[what]
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1]},
                                  constrained_layout=True)
    ax.plot(steps, values, ".-", lw=0.8, ms=3, color="tab:blue", label="measured")
    if reference is not None:
        ref = np.broadcast_to(np.asarray(reference, dtype=float), np.shape(values))
        ref_label = "von Kármán theory" if what == "fiber" else "target r₀"
        ax.plot(steps, ref, "--", color="tab:red", lw=1.2, label=ref_label)
    ax.set_xlabel("simulation tick")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ok = values[np.isfinite(values)]
    if ok.size:
        axh.hist(ok, bins=min(40, max(5, ok.size // 5)), orientation="horizontal", color="tab:blue",
                 alpha=0.75)
        axh.axhline(np.mean(ok), color="k", lw=1, label=f"mean {np.mean(ok):.4g}")
        axh.legend(fontsize=8)
    axh.set_xlabel("frames")
    axh.set_ylim(ax.get_ylim())
    fig.suptitle(f"opal analyze {what}: {summary['source']}", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyze_run(path, what="fiber", *, frames=20, stride=1, start=None, stop=None, aperture=None,
                a=1.12, out_dir=None, save=True, plot=False, max_batch_bytes=DEFAULT_BATCH_BYTES,
                log=print):
    """Measure fiber coupling (``what="fiber"``) or r0 (``what="r0"``) over saved frames.

    Parameters
    ----------
    path : run directory, pass directory or ``.npy`` file (with a manifest next to it)
    frames : int, "all" or None
        How many frames of the selected range to analyse (evenly spaced).
        ``"all"``/``None`` takes every frame in ``range(start, stop, stride)``.
    aperture : float, optional
        Receiver aperture diameter (m). Fiber default: ``0.4 * N * dx``; r0
        default: the full grid.
    a : float
        Fiber coupling geometry ``D / (2 w0)`` (1.12 is optimal).
    out_dir : path, optional
        Where results go (default ``<run>/analysis``).
    save, plot : bool
        Write the CSV + summary JSON, and the PNG figure.
    max_batch_bytes : int
        RAM budget for one batch of frames (default 256 MiB).

    Returns
    -------
    dict: the summary (also written as ``<what>_summary.json``) plus the
    per-frame ``values`` and ``steps`` arrays.
    """
    if what not in ANALYSES:
        raise ValueError(f"what must be one of {ANALYSES}, got {what!r}")
    say = log if log is not None else (lambda _m: None)
    t0 = time.time()
    fs = open_frames(path)
    manifest = fs.manifest
    if not manifest.get("config"):
        raise ValueError(f"{path} has no manifest.json with the run config (needed for dx, r0, L0)")
    ch = manifest["config"]["channel_params"]
    ch = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in ch.items()}
    dx, L0, r0_cfg = float(ch["pixel_size"]), float(ch["L0"]), float(ch["total_r0"])
    N = int(fs.shape[-1])

    idx = select_indices(len(fs), frames, stride=stride, start=start, stop=stop)
    if idx.size == 0:
        raise ValueError(f"no frames selected (run has {len(fs)} frames; check --start/--stop/--stride)")
    steps = fs.steps[idx]
    targets_map = _pass_targets(fs.source) if fs.kind == "pass" and fs.source else None
    target_r0 = (np.array([targets_map.get(int(s), math.nan) for s in steps]) if targets_map else None)
    target_model = "pass target r0 (r0_tracking.csv)" if target_r0 is not None else None
    if target_r0 is None:
        target_r0 = _schedule_targets(fs, steps)
        if target_r0 is not None:
            target_model = ("turbulence schedule (schedule.npy): instantaneous target path r0; the screens "
                            "follow it as new rows scroll in")

    if what == "fiber":
        D = float(aperture) if aperture else 0.4 * N * dx
        if D >= 0.47 * N * dx:
            raise ValueError(f"aperture must be < {0.47 * N * dx:.3f} m (inside the absorbing boundary)")
        per_frame = fs.frame_nbytes * 4
    else:
        D = float(aperture) if aperture else None
        per_frame = fs.frame_nbytes * 2
    batch = max(1, int(max_batch_bytes // per_frame))
    read_gib = idx.size * fs.frame_nbytes / GIB
    say(f"{what}: {idx.size} of {len(fs)} frames from {fs.kind} output {fs.source} "
        f"({read_gib:.2f} GiB to read, {batch} frames per batch, <= {batch * per_frame / MIB:.0f} MiB RAM)")

    values = np.full(idx.size, np.nan)
    done, n_batches = 0, math.ceil(idx.size / batch)
    report = max(1, n_batches // 10)
    for b, (sel, block) in enumerate(fs.iter_batches(idx, batch_size=batch)):
        if what == "fiber":
            values[done:done + len(sel)] = compute_fiber_coupling(
                block, dx, a=a, aperture_diameter=D, max_batch_bytes=max_batch_bytes)["eta"]
        else:
            for k, frame in enumerate(block):
                values[done + k] = estimate_r0_from_wavefront(frame, dx, D, L0=L0)["measured_r0"]
        done += len(sel)
        del block
        if n_batches > 1 and ((b + 1) % report == 0 or b + 1 == n_batches):
            say(f"{what}: {done}/{idx.size} frames ({100 * done / idx.size:5.1f}%) {time.time() - t0:6.1f} s")

    stats = _stats(values)
    summary = {
        "analysis": what,
        "opal_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(fs.source) if fs.source else None,
        "source_kind": fs.kind,
        "frames_available": len(fs),
        "frames_analysed": int(idx.size),
        "selection": {"frames": frames if frames is None or isinstance(frames, str) else int(frames),
                      "stride": int(stride or 1), "start": start, "stop": stop},
        "grid": N,
        "pixel_size_m": dx,
        "configured_total_r0_m": r0_cfg,
        "L0_m": L0,
        "statistics": stats,
        "elapsed_s": None,
        "note": NOTE,
    }
    reference = None
    if what == "fiber":
        r0_ref = float(np.nanmedian(target_r0)) if target_r0 is not None else r0_cfg
        summary.update({"aperture_m": D, "a": float(a), "D_over_r0": D / r0_ref})
        if target_r0 is not None:
            cache = {}
            theory = np.array([cache.setdefault(round(r, 5), coupling_efficiency_theoretical_vk(D, r, L0, a=a))
                               if np.isfinite(r) else math.nan for r in target_r0])
            reference = theory
            summary["theory"] = {"model": "von Karman, per-frame target r0 from the " +
                                 ("pass" if fs.kind == "pass" else "turbulence schedule"),
                                 "mean": float(np.nanmean(theory))}
        else:
            theory_val = coupling_efficiency_theoretical_vk(D, r0_cfg, L0, a=a)
            reference = theory_val
            summary["theory"] = {"model": "von Karman, configured r0", "eta": theory_val}
            if stats.get("count"):
                summary["theory"]["deviation_percent"] = 100 * (stats["mean"] / theory_val - 1)
    else:
        summary["aperture_m"] = D if D else N * dx
        if target_r0 is not None:
            reference = target_r0
            ratio = values / target_r0
            ok = ratio[np.isfinite(ratio)]
            summary["target"] = {"model": target_model,
                                 "r0_m_median": float(np.nanmedian(target_r0)),
                                 "ratio_median": float(np.median(ok)) if ok.size else None}
        else:
            reference = r0_cfg
            if stats.get("count"):
                summary["target"] = {"model": "configured total r0", "r0_m": r0_cfg,
                                     "ratio_median": stats["median"] / r0_cfg}
    summary["elapsed_s"] = time.time() - t0

    files = {}
    if save or plot:
        base = Path(out_dir) if out_dir else (
            (fs.source if fs.source and fs.source.is_dir() else Path(path).parent) / "analysis")
        base.mkdir(parents=True, exist_ok=True)
        if save:
            if what == "fiber":
                header, cols = ["frame", "step", "eta"], [idx, steps, values]
                if target_r0 is not None:
                    header += ["target_r0_m", "eta_theory"]
                    cols += [target_r0, reference]
            else:
                header, cols = ["frame", "step", "r0_m"], [idx, steps, values]
                if target_r0 is not None:
                    header.append("target_r0_m")
                    cols.append(target_r0)
            _write_csv(base / f"{what}.csv", header, cols)
            files["csv"] = str(base / f"{what}.csv")
        if plot:
            _plot(base / f"{what}.png", what, steps, values, reference, summary)
            files["plot"] = str(base / f"{what}.png")
        if save:
            files["summary"] = str(base / f"{what}_summary.json")
            summary["files"] = files
            (base / f"{what}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        else:
            summary["files"] = files
    summary["values"] = values
    summary["steps"] = steps
    return summary


def load_analysis(run_dir, what):
    """``(steps, values, extra_columns)`` from ``<run>/analysis/<what>.csv``, or ``None``."""
    path = Path(run_dir) / "analysis" / f"{what}.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    key = "eta" if what == "fiber" else "r0_m"
    steps = np.array([int(r["step"]) for r in rows])
    values = np.array([float(r[key]) for r in rows])
    extra = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ("frame", "step", key)}
    return steps, values, extra
