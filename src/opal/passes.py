"""Exact dynamic-turbulence simulation of a satellite pass.

:func:`run_pass` builds the pass geometry, derives each layer's slant r0 along
the pass, and drives :meth:`GpuSimulator.simulate_pass_replacement`: the pass
is split into segments and, at every segment boundary, each layer's
extrusion operators are rebuilt for its new r0 while the current screens are
kept, so the field evolves continuously. Received wavefronts are captured at
evenly spaced ticks, streamed to a memory-mapped file, and (optionally) their
effective r0 is measured against the target.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _timing
from . import memory as mem
from ._version import __version__
from .analysis.pass_geometry import (
    build_replacement_schedule,
    build_zenith_cn2_profile,
    circular_orbit_pass,
    effective_path_r0,
    per_layer_r0_along_pass,
    truncate_pass_to_window,
)
from .analysis.r0 import estimate_r0_from_wavefront
from .config import PASS_DEFAULTS, load_config, to_plain

__all__ = ["PassPlan", "plan_pass", "run_pass"]


@dataclass
class PassPlan:
    orbit_pass: object
    zenith_profile: object
    per_layer_r0_t: np.ndarray
    effective_r0_t: np.ndarray
    schedule: np.ndarray
    segment_steps: int
    segment_starts: np.ndarray
    wavefront_steps: np.ndarray
    steps_per_pass_sample: float
    num_steps: int

    def pass_index(self, steps):
        idx = np.round(np.asarray(steps) / self.steps_per_pass_sample).astype(int)
        return np.clip(idx, 0, len(self.effective_r0_t) - 1)


def plan_pass(config):
    """Compute the pass geometry, r0 schedule and capture ticks for ``config``."""
    config = load_config(config)
    plain = to_plain(config)
    ch, sim = plain["channel_params"], plain["simulation_params"]
    pp = dict(PASS_DEFAULTS)
    pp.update(plain.get("pass_params", {}) or {})

    num_steps = int(sim["num_steps"])
    num_segments = int(pp["num_segments"])
    if num_steps % num_segments:
        raise ValueError(f"num_steps ({num_steps}) must be a multiple of num_segments ({num_segments})")
    altitude = float(pp.get("altitude_m") or ch["satellite_orbit_altitude"])

    orbit = circular_orbit_pass(altitude, float(pp["min_elevation_deg"]), int(pp["pass_num_samples"]))
    orbit = truncate_pass_to_window(orbit, pp.get("pass_window_s"))
    zenith = build_zenith_cn2_profile(
        num_layers=int(ch["num_layers"]),
        wavelength=float(ch["wavefront_wavelength"]),
        ground_wind_speed=float(ch["ground_wind_speed"]),
        satellite_orbit=float(ch["satellite_orbit_altitude"]),
        total_r0_zenith=float(ch["total_r0"]),
    )
    per_layer = per_layer_r0_along_pass(zenith, orbit.zenith_angle_deg)
    eff = effective_path_r0(per_layer)
    schedule, seg_steps, starts = build_replacement_schedule(per_layer, num_steps, num_segments)
    count = max(1, int(pp["wavefront_captures"]))
    wf_steps = np.unique(np.linspace(0, num_steps - 1, count).round().astype(np.int64))
    return PassPlan(
        orbit_pass=orbit,
        zenith_profile=zenith,
        per_layer_r0_t=per_layer,
        effective_r0_t=eff,
        schedule=schedule,
        segment_steps=seg_steps,
        segment_starts=starts,
        wavefront_steps=wf_steps,
        steps_per_pass_sample=(num_steps - 1) / max(1, len(orbit.t) - 1),
        num_steps=num_steps,
    )


def run_pass(config, out_dir=None, *, seed=None, simulator=None, measure_r0=True, log=print):
    """Simulate a full satellite pass with exact per-segment r0 switching.

    Parameters
    ----------
    config : dict or path
        Needs ``pass_params`` (defaults are filled in).
    out_dir : path, optional
        Writes ``wavefronts.npy`` (memory-mapped captures), ``schedule.npz``,
        ``r0_tracking.csv`` and ``manifest.json``.
    simulator : GpuSimulator, optional
        Reuse an existing simulator built from the same config.
    measure_r0 : bool
        Measure the effective r0 of every captured wavefront (MCF 1/e, von Karman).

    Returns
    -------
    dict with ``plan``, ``records`` (per capture), ``segments``, ``wavefronts``
    (array or memmap) and ``manifest``.
    """
    from .runners import warn_if_no_gpu
    from .simulator import GpuSimulator

    t0 = _timing.clock()

    def say(msg):
        if log is not None:
            log(f"[opal {time.time() - t0:7.2f}s] {msg}")

    config = load_config(config)
    plain = to_plain(config)
    ch = plain["channel_params"]
    pp = dict(PASS_DEFAULTS)
    pp.update(plain.get("pass_params", {}) or {})
    plan = plan_pass(config)
    orbit = plan.orbit_pass
    N = int(ch["phase_screen_size"])
    dx = float(ch["pixel_size"])
    L0 = float(ch["L0"])
    guard = float(pp.get("validation_guard_frac", 0.0) or 0.0)
    aperture = max(0.0, (1.0 - guard) * N * dx) or N * dx

    say(f"pass: {orbit.h_orbit / 1e3:.0f} km orbit, elevation {orbit.elevation_deg.min():.1f}-"
        f"{orbit.elevation_deg.max():.1f} deg over {orbit.duration_s:.0f} s")
    say(f"schedule: {plan.schedule.shape[0]} segments x {plan.segment_steps} steps = {plan.num_steps}, "
        f"path r0 {np.nanmin(plan.effective_r0_t):.3f}-{np.nanmax(plan.effective_r0_t):.3f} m, "
        f"{len(plan.wavefront_steps)} wavefront captures"
        + (f" (requested {int(pp['wavefront_captures'])}; at most one per tick)"
           if int(pp["wavefront_captures"]) > len(plan.wavefront_steps) else ""))

    warn_if_no_gpu(say)
    build_s = None
    if simulator is None:
        say(f"building simulator: N={N}, {int(ch['num_layers'])} layers")
        tb = time.time()
        simulator = GpuSimulator(config, seed=seed, progress=_timing.layer_progress(say, N))
        build_s = time.time() - tb
        say(f"built simulator in {build_s:.1f} s")
    say(f"seed {simulator.seed}")

    wf_store = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "schedule.npz",
                 per_layer_r0_schedule=plan.schedule,
                 segment_steps=plan.segment_steps,
                 segment_starts=plan.segment_starts,
                 per_layer_r0_t=plan.per_layer_r0_t,
                 effective_r0_t=plan.effective_r0_t,
                 pass_t=orbit.t,
                 elevation_deg=orbit.elevation_deg,
                 zenith_per_layer_r0=plan.zenith_profile.r0_layers_zenith,
                 layer_heights=plan.zenith_profile.layer_heights)
        np.save(out_dir / "wavefront_steps.npy", plan.wavefront_steps)
        wf_store = np.lib.format.open_memmap(out_dir / "wavefronts.npy", mode="w+", dtype=np.complex64,
                                             shape=(len(plan.wavefront_steps), N, N))
    else:
        wf_store = np.empty((len(plan.wavefront_steps), N, N), dtype=np.complex64)
    step_to_slot = {int(s): i for i, s in enumerate(plan.wavefront_steps)}

    records, segments = [], []
    last = {"t": time.time()}
    num_segments = plan.schedule.shape[0]

    def on_segment(seg_idx, payload):
        wf, steps = payload["wavefronts"], payload["wavefront_steps"]
        target_eff = float(effective_path_r0(np.asarray(payload["target_per_layer_r0"])[None, :])[0])
        measured = []
        for i in range(wf.shape[0]):
            s = int(steps[i])
            wf_store[step_to_slot[s]] = wf[i]
            pidx = int(plan.pass_index(s))
            r0m = math.nan
            if measure_r0:
                r0m = estimate_r0_from_wavefront(wf[i], dx, aperture, L0=L0)["measured_r0"]
            measured.append(r0m)
            records.append({
                "step": s,
                "segment": seg_idx,
                "pass_time_s": float(orbit.t[pidx]),
                "elevation_deg": float(orbit.elevation_deg[pidx]),
                "theoretical_r0": float(plan.effective_r0_t[pidx]),
                "target_effective_r0": target_eff,
                "measured_r0": float(r0m),
            })
        now = time.time()
        dt, last["t"] = now - last["t"], now
        mid = int(plan.pass_index(payload["global_step_offset"] + plan.segment_steps // 2))
        med = float(np.nanmedian(measured)) if measured and np.isfinite(measured).any() else math.nan
        seg = {"segment": seg_idx, "start_step": int(payload["global_step_offset"]),
               "elevation_deg": float(orbit.elevation_deg[mid]), "target_effective_r0": target_eff,
               "measured_r0_median": med, "captures": int(wf.shape[0]), "wall_time_s": dt}
        segments.append(seg)
        say(f"segment {seg_idx + 1:3d}/{num_segments} elev {seg['elevation_deg']:5.1f} deg "
            f"target r0 {target_eff:.3f} m"
            + (f" measured {med:.3f} m" if measure_r0 and np.isfinite(med) else "")
            + f" ({wf.shape[0]} captures, {dt:.2f} s)")
        return seg

    mem.reset_memory_stats()
    run_start = time.time()
    result = simulator.simulate_pass_replacement(
        per_layer_r0_schedule=plan.schedule,
        segment_steps=plan.segment_steps,
        wavefront_capture_steps=plan.wavefront_steps,
        screen_capture_steps=None,
        regenerate_on_switch=bool(pp["regenerate_on_switch"]),
        verbose=False,
        per_segment_callback=on_segment,
        capture_to_host=bool(pp["capture_to_host"]),
        match_initial_r0=bool(pp["match_initial_r0"]),
    )
    run_time = time.time() - run_start
    peak = mem.memory_info()["peak"]
    simulator.release_host_capture_buffer()

    ratio = np.array([r["measured_r0"] / r["theoretical_r0"] for r in records
                      if np.isfinite(r["measured_r0"]) and r["theoretical_r0"] > 0])
    say(f"done: {plan.num_steps} steps in {run_time:.1f} s | peak GPU {mem.format_bytes(peak)}"
        + (f" | measured/theory r0 median {np.median(ratio):.3f}" if ratio.size else ""))

    manifest = {
        "opal_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": simulator.seed,
        "config": plain,
        "pass": {"altitude_m": orbit.h_orbit, "min_elevation_deg": orbit.min_elevation_deg,
                 "duration_s": orbit.duration_s, "samples": int(len(orbit.t))},
        "num_steps": plan.num_steps,
        "num_segments": int(num_segments),
        "segment_steps": int(plan.segment_steps),
        "wavefront_captures": int(len(plan.wavefront_steps)),
        "aperture_m": aperture,
        "run_time_s": run_time,
        "switch_time_s": float(sum(m["switch_time_s"] for m in result["per_segment_metadata"])),
        "gpu_peak_bytes": peak,
        "r0_ratio_median": float(np.median(ratio)) if ratio.size else None,
        "timing": {"build_s": None if build_s is None else round(build_s, 3),
                   "simulate_s": round(run_time, 3)},
        "files": {},
    }
    if out_dir is not None:
        t_write = time.time()
        wf_store.flush()
        with (out_dir / "r0_tracking.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        with (out_dir / "segments.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(segments[0].keys()))
            writer.writeheader()
            writer.writerows(segments)
        manifest["files"] = {"wavefronts": "wavefronts.npy", "wavefront_steps": "wavefront_steps.npy",
                             "schedule": "schedule.npz", "r0_tracking": "r0_tracking.csv",
                             "segments": "segments.csv"}
        manifest["timing"]["write_s"] = round(time.time() - t_write, 3)
    if _timing.started() is not None:
        manifest["timing"]["since_process_start_s"] = round(time.time() - _timing.started(), 3)
    if out_dir is not None:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        say(f"wrote {out_dir}")
    return {"plan": plan, "records": records, "segments": segments, "wavefronts": wf_store,
            "manifest": manifest, "result": result}
