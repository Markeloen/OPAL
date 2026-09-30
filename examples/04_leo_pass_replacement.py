"""LEO satellite pass with exact per-layer r0 switching.

    python examples/04_leo_pass_replacement.py

Builds a 700 km circular overhead pass (20 deg to 20 deg elevation), scales
every layer's r0 along the slant path and splits the pass into segments. At
each segment boundary the extrusion operators of every layer are rebuilt for
its new r0 while the current screens are kept, so the received field evolves
continuously while its statistics follow the pass. The whole pass (or the
``pass_window_s`` around zenith) is mapped onto the run's ticks, so a
several-minute pass becomes a short, statistically faithful sequence. Every
captured wavefront's r0 is measured and compared with the target. This is a
reduced version of ``configs/leo_pass_b2.json``;
``opal pass -c configs/leo_pass_b2.json -o runs/b2`` runs the full case.
"""

from pathlib import Path

import numpy as np

import opal

ROOT = Path(__file__).resolve().parents[1]

config = opal.load_config(ROOT / "configs" / "leo_pass_b2.json",
                          overrides={"N": 256, "steps": 1000})
config["pass_params"].update(num_segments=10, wavefront_captures=40)   # 100 ticks per segment, as in the paper

plan = opal.plan_pass(config)
print(f"pass: {plan.orbit_pass.duration_s:.0f} s, elevation "
      f"{plan.orbit_pass.elevation_deg.min():.0f}-{plan.orbit_pass.elevation_deg.max():.0f} deg, "
      f"path r0 {np.nanmin(plan.effective_r0_t):.3f}-{np.nanmax(plan.effective_r0_t):.3f} m")

out = opal.run_pass(config, ROOT / "runs" / "leo_pass")

ratio = [r["measured_r0"] / r["theoretical_r0"] for r in out["records"] if np.isfinite(r["measured_r0"])]
print(f"measured / target r0: median {np.median(ratio):.3f} over {len(ratio)} captures")
print("wavefronts:", out["wavefronts"].shape, "->", ROOT / "runs" / "leo_pass" / "wavefronts.npy")
