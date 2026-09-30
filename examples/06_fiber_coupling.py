"""Single-mode fiber coupling: Monte Carlo vs von Karman theory.

    python examples/06_fiber_coupling.py

Draws independent channel realizations with ``reseed`` (no graph rebuilds),
computes the coupling efficiency of each received field into a single-mode
fiber, with and without centroid tip/tilt removal, and compares the
ensemble mean with the von Karman prediction.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

import opal
from opal.analysis import (compute_fiber_coupling, coupling_efficiency_theoretical_vk,
                           remove_tilt_com)

ROOT = Path(__file__).resolve().parents[1]
REALIZATIONS = 120                   # eta is heavy-tailed at large D/r0: the mean needs many samples
APERTURES = (0.2, 0.3, 0.5)          # m, must stay < 0.47 * N * dx

config = opal.load_config(ROOT / "configs" / "quickstart_256.json")
sim = opal.GpuSimulator(config)
dx, r0, L0 = sim.px_delta, sim.total_r0, 25.0
ones = tf.ones([1, sim.num_layers], tf.float32)

fields = []
for k in range(REALIZATIONS):
    sim.reseed(10_000 + k)
    fields.append(sim.simulate_steps_no_save(tf.constant(1), ones).numpy())
fields = np.stack(fields)

print(f"{REALIZATIONS} independent realizations, r0 = {r0} m, L0 = {L0} m")
print(" D (m)  D/r0   eta (MC)          eta (VK theory)   eta TT-corrected")
for D in APERTURES:
    eta = compute_fiber_coupling(fields, dx, aperture_diameter=D)["eta"]
    corrected, _ = remove_tilt_com(fields, dx, aperture_diameter=D)
    eta_tt = compute_fiber_coupling(corrected, dx, aperture_diameter=D)["eta"]
    theory = coupling_efficiency_theoretical_vk(D, r0, L0)
    sem = eta.std() / np.sqrt(REALIZATIONS)
    print(f" {D:4.2f}  {D / r0:5.1f}  {eta.mean():.3f} +- {sem:.3f}   {theory:.3f}"
          f"             {eta_tt.mean():.3f}")
