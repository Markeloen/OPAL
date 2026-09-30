"""Checkpoint / resume: long runs in segments, bit-exact.

    python examples/03_checkpoint_resume.py

The full simulator state (all layer screens, per-layer RNG states, tick
counter and turbulence scales) fits in one small ``.npz``. Continuing from it
reproduces an uninterrupted run exactly, so arbitrarily long time series can
be produced in pieces, across processes, or after a crash.
The CLI equivalent is ``opal run -c cfg.json --steps 50 --checkpoint state.npz``
(repeat to continue).
"""

from pathlib import Path

import numpy as np

import opal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "checkpoint"
config = opal.load_config(ROOT / "configs" / "quickstart_256.json")
sim = opal.GpuSimulator(config)

reference = np.asarray(opal.run(sim, 120, mode="device", log=None).frames)

state = OUT / "state.npz"
if state.exists():
    state.unlink()
pieces = []
for k in range(4):
    res = opal.run(sim, 30, mode="host", state_in=state if k else None, state_out=state, log=None)
    pieces.append(np.asarray(res.frames))
    print(f"segment {k}: ticks {res.manifest['start_step']}-{res.manifest['end_step'] - 1}")

print("4 x 30 ticks == 1 x 120 ticks:", np.array_equal(np.concatenate(pieces), reference))

# The same state can seed a fresh process: build a simulator from the same
# config and load the checkpoint.
other = opal.GpuSimulator(config)
opal.load_state(other, state)
print("resumed at tick", int(other.get_state()["global_step"]))
