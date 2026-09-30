"""Quick start: build a channel, run it, and look at what came out.

    python examples/01_quickstart.py

Runs the 256x256, 10-layer quick-start channel for 200 ticks (1 ms each) and
writes ``runs/quickstart/frames.npy`` (complex64 [200, 256, 256]) plus a
``manifest.json`` describing exactly how it was made.
"""

from pathlib import Path

import numpy as np

import opal
from opal.analysis import aperture_mask, estimate_r0_from_wavefront, scintillation_index

ROOT = Path(__file__).resolve().parents[1]

config = opal.load_config(ROOT / "configs" / "quickstart_256.json")
sim = opal.GpuSimulator(config)

info = sim.describe()
print(f"grid {info['grid']}x{info['grid']} @ {info['pixel_size_m']} m, "
      f"r0 = {info['total_r0_m']} m, D/r0 = {info['D_over_r0']:.1f}, seed {info['seed']}")
for layer in info["layers"]:
    print(f"  layer {layer['layer']}: h = {layer['height_m']:8.0f} m  r0 = {layer['r0_m']:.3f} m  "
          f"wind = {layer['wind_mps']:6.1f} m/s  ({layer['rows_per_tick']:.2f} rows/tick)")

result = opal.run(sim, mode="auto", out_dir=ROOT / "runs" / "quickstart")

frames = result.frames                          # [T, N, N] complex64, memory-mapped
dx = info["pixel_size_m"]
pupil = aperture_mask(info["grid"], dx, 1.0)    # 1 m receiver, away from the absorbing boundary
intensity = np.abs(frames[-1]) ** 2
r0 = estimate_r0_from_wavefront(frames[-1], dx, L0=25.0)["measured_r0"]
print(f"\n{result.mode} mode: {result.num_steps} ticks in {result.run_time_s:.1f} s "
      f"({result.steps_per_second:.1f} ticks/s)")
print(f"last frame, 1 m aperture: mean irradiance {intensity[pupil].mean():.3f}, "
      f"scintillation index {scintillation_index(frames[-1], pupil):.3f}; measured r0 {r0:.3f} m")
print("view it with:  opal view runs/quickstart --save quickstart.gif")
