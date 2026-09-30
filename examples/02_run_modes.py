"""Run modes and the memory guard.

    python examples/02_run_modes.py

OPAL can keep frames on the GPU (``device``), stream them to host RAM or a
memory-mapped file (``host``), write bounded chunks to disk (``chunked``), or
keep only the final field (``no_save``). All four evolve the *same* channel:
for a given seed they produce bit-identical frames. ``auto`` asks the memory
guard to choose.
"""

from pathlib import Path

import numpy as np

import opal
from opal import memory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "modes"

sim = opal.GpuSimulator(opal.load_config(ROOT / "configs" / "quickstart_256.json", overrides={"steps": 100}))

# What would the guard do for a much bigger job? (dry run, nothing is simulated)
baseline = memory.measure_baseline_peak_bytes(sim)
for steps in (100, 5_000, 200_000):
    est = memory.estimate_memory(steps, sim.nx_size, baseline)
    mode, chunk, reason = memory.select_mode("auto", est, has_output_dir=True)
    print(f"{steps:>7d} ticks -> {mode:8s} {reason}")

frames = {}
for mode in ("device", "host", "chunked", "no_save"):
    res = opal.run(sim, mode=mode, out_dir=OUT / mode, chunk_size=32, log=None)
    peak = res.manifest["gpu"]["run_peak_bytes"] / 2**20
    print(f"{mode:8s}: {res.steps_per_second:6.1f} ticks/s, peak GPU {peak:7.1f} MiB")
    frames[mode] = opal.load_frames(OUT / mode)

ref = frames["device"]
print("host    == device:", np.array_equal(frames["host"], ref))
print("chunked == device:", np.array_equal(frames["chunked"], ref))
print("no_save == device[-1]:", np.array_equal(frames["no_save"][0], ref[-1]))

# Large runs: iterate without loading everything
n = sum(1 for _ in opal.iter_frames(OUT / "chunked", stride=10))
print(f"iterated {n} frames from the chunked run")
