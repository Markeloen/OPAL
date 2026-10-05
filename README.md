# OPAL

**Optical Propagation through Atmospheric Layers**: a GPU-accelerated multilayer
atmospheric turbulence simulator for space-to-ground optical links.

OPAL evolves the complex optical field received by a ground station from a
satellite, tick by tick, through a stack of moving von Kármán phase screens. It
keeps everything resident on the GPU as TensorFlow graphs:
- the screens,
- their extrusion operators,
- the angular-spectrum propagator,
- the received field.

This makes long, time-correlated channel sequences cheap: roughly 0.12–0.2 s per
tick from 256² to 2048², and up to ~310× faster than an equivalent HCIPy
pipeline.

- **Physically grounded channel.**
  - Hufnagel–Valley C<sub>n</sub>² profile sliced into *L* layers.
  - Bufton wind plus satellite-slew apparent wind.
  - Infinite von Kármán phase screens that extrude new rows every tick
    (Assémat / Fried–Clark autoregressive method).
  - Angular-spectrum propagation between layers.
- **Dynamic turbulence.** Per-tick turbulence schedules (b-scale, C<sub>n</sub>²,
  per-layer r₀ or path r₀), plus *exact* per-layer r₀ switching for full
  satellite passes. Operators are rebuilt for the new statistics while the
  screens persist.
- **Four run modes and a memory guard.** Keep frames on the GPU, stream them to
  host RAM or a memory-mapped file, write bounded chunks to disk, or keep only
  the final field. `auto` picks a mode that fits and falls back on
  out-of-memory errors.
- **Bit-exact segmentation.** The full simulator state fits in one small
  checkpoint. Splitting a run into pieces reproduces the uninterrupted run
  exactly.
- **ML dataset generation.** Seeded parameter sweeps × independent realizations
  produce labelled, time-correlated irradiance/phase sequences, with fiber
  coupling, scintillation, angle of arrival and measured r₀ labels. Interrupted
  datasets resume.
- **Memory-safe analysis and viewing.** r₀ and single-mode-fiber coupling per
  frame, saved as CSV/JSON/PNG, and an interactive viewer. Both read frames
  lazily, so any run size works on a laptop.
- **Portable models.** Export a configured simulator as a TensorFlow SavedModel
  and run it without the Python source.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Configuration](#3-configuration)
4. [Run modes and the memory guard](#4-run-modes-and-the-memory-guard)
5. [Dynamic turbulence: schedules and satellite passes](#5-dynamic-turbulence-schedules-and-satellite-passes)
6. [Checkpoints and bit-exact segmentation](#6-checkpoints-and-bit-exact-segmentation)
7. [ML dataset generation](#7-ml-dataset-generation)
8. [Analysis](#8-analysis)
9. [Visualization (`opal view`)](#9-visualization-opal-view)
10. [SavedModel export](#10-savedmodel-export)
11. [Python API](#11-python-api)
12. [Output formats](#12-output-formats)
13. [Architecture](#13-architecture)
14. [Modifying the engine](#14-modifying-the-engine)
15. [Performance model](#15-performance-model)
16. [Examples](#16-examples)
17. [Citation](#17-citation) · [License](#18-license)

---

## 1. Installation

OPAL needs a CUDA-capable NVIDIA GPU and TensorFlow ≥ 2.10. It was developed
and benchmarked on an RTX 4080 (16 GiB), and also runs on 8 GiB laptop GPUs (the
memory guard adapts; see [section 4](#4-run-modes-and-the-memory-guard)).

### 1.1 Windows (native GPU)

TensorFlow 2.10 is the last release with native Windows CUDA support. The conda
environment pins it, together with CUDA 11.2 and cuDNN 8.1:

```bash
git clone https://github.com/Markeloen/OPAL.git
cd OPAL
conda env create -f environment.yml      # creates "opal" and installs OPAL in editable mode
conda activate opal
```

### 1.2 Linux / WSL2

```bash
git clone https://github.com/Markeloen/OPAL.git
cd OPAL
python -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]"
pip install -e ".[viewer]"
```

### 1.3 Verify

```bash
opal info                       # OPAL/TensorFlow versions, CUDA build, GPU, host RAM
opal run -c configs/quickstart_256.json --steps 50
```

`python -m opal ...` is equivalent to `opal ...`.

TensorFlow is an optional extra (`pip install -e ".[tf]"`) rather than a hard
dependency, so pip never replaces a working GPU build with a CPU-only wheel.
`matplotlib` is needed only for `opal view` and `analyze --plot`, and `ffmpeg`
only for `.mp4` output (GIFs need nothing extra). The example notebooks also
need JupyterLab: `pip install -e ".[examples]"` ([section 16](#16-examples)).

---

## 2. Quick start

### 2.1 Command line

```bash
# 200 ticks of the 256x256, 10-layer quick-start channel
opal run -c configs/quickstart_256.json

# Look at it (the run directory is printed at the start and end of the run)
opal view runs/<run-dir>                       # interactive window
opal view runs/<run-dir> --save qs.gif         # or a GIF

# Measure it
opal analyze r0 runs/<run-dir> --frames all
opal analyze fiber runs/<run-dir> --frames all --plot

# Change anything from the command line
opal run -c configs/quickstart_256.json --grid 512 --r0 0.08 --set wind=12 --steps 1000 -o runs/strong
```

### 2.2 Where outputs go

Every command that writes something has a default destination, so nothing is
lost if you forget `-o`. Folders are created relative to the current directory
and named by date, time and the main parameters:

| Command | Default output (without `-o`) |
|---|---|
| `opal run` | `runs/2026-07-16_143012_N1024_L10_r0-0.15_T1000_seed112082364/` |
| `opal pass` | `runs/<date>_<time>_pass_N512_L10_r0-0.2_T3000_seed1/` |
| `opal dataset` | `datasets/<date>_<time>_N256_L10_T200_seed7_k8/` |
| `opal export` | `exports/<date>_<time>_N1024_L10_r0-0.2_T1000_seed1/` |
| `opal analyze` | `<run>/analysis/` |
| `opal view --save NAME` | `NAME` (snapshots taken in the window go to `<run>/snapshots/`) |

`-o DIR` picks the folder yourself. `opal run --no-save` and
`opal pass --no-save` write nothing, which is useful for benchmarks. If `-o`
points at a folder that already holds a run, OPAL says which files it is about
to replace.

Every run directory holds the frames and a `manifest.json`. The manifest records
the resolved config, the seed, the mode, a timing breakdown, GPU memory and the
OPAL version, so any run can be reproduced.

### 2.3 What the console shows (and why the stopwatch agrees)

```text
OPAL 1.1.0 | N=256 layers=10 r0=0.1 m D/r0=25.6 seed=1
[opal    0.08s] output: runs\2026-10-04_225050_N256_L10_r0-0.1_T50_seed1
[opal    2.22s] startup 2.2 s (Python, NumPy/SciPy and TensorFlow import)
[opal    2.41s] building simulator: N=256, 10 layers (per-layer covariance, Cholesky and SVD on the CPU; done once per run)
[opal    9.73s] built simulator in 7.3 s | extrusion 212 rows/tick over all layers (wind multiplier x2; per-layer table: opal plan)
[opal   13.88s] memory plan: N=256 steps=50 requested=host -> host
...
[opal   20.91s] host: step     50/50 (100.0%)     7.1 steps/s | GPU in use 0.02 GiB
[opal   21.05s] wrote frames.npy (0.02 GiB): final flush to disk took 0.0 s
[opal   21.05s] done: 50 steps simulated in 7.15 s | 7.0 steps/s | peak GPU 0.04 GiB
[opal   21.05s] total wall time 21.1 s = startup 2.2 + build 7.3 + plan (1 traced test step) 4.1 + simulate 7.1 + write 0.0 + other 0.2 s
[opal   21.05s] output: runs\2026-10-04_225050_N256_L10_r0-0.1_T50_seed1
```

The clock in `[opal  X s]` starts when the command starts, so the last line
matches a stopwatch. A command spends its time in these phases:

| Phase | What happens | Typical cost |
|---|---|---|
| startup | Python, NumPy/SciPy and TensorFlow import, CUDA initialisation | 2–6 s |
| build | per layer: covariance matrices, Cholesky, SVD and the stability check of the extrusion operators (CPU), initial screen (FFT) | 7 s at 256², ≈ 25 s at 1024², minutes at 2048² (10 layers; slower on laptop CPUs) |
| plan | one traced test step that measures the simulator's real GPU footprint for the memory guard (the first call also compiles the graph) | 2–5 s |
| simulate | the ticks themselves; this is the number that `steps/s` refers to | ≈ 0.07–0.2 s per tick (10 layers, 256²–1024², depends on the GPU) |
| write | flushing the memory-mapped `frames.npy` to disk (host mode) or saving `frames.npy` (device mode) | 0 s to minutes, see [section 4.4](#44-large-outputs-on-machines-with-little-ram) |

The breakdown is also saved under `timing` in `manifest.json`.

### 2.4 Python

```python
import opal

config = opal.load_config("configs/quickstart_256.json", overrides={"steps": 500})
sim = opal.GpuSimulator(config)            # builds screens + operators on the GPU

result = opal.run(sim, mode="auto", out_dir="runs/qs")
frames = result.frames                     # complex64 [500, 256, 256] (memory-mapped)
print(result.mode, f"{result.steps_per_second:.1f} ticks/s")

sim.reseed(7)                              # new independent realization, no rebuild
```

In Python, `out_dir=None` keeps the frames in memory (or on the GPU) only.

---

## 3. Configuration

### 3.1 File format

A config is a JSON file with two sections, `simulation_params` and
`channel_params`, plus an optional `pass_params` used by `opal pass`.
Unspecified keys take their defaults. A value can be written plainly or as
`{"value": ..., "description": "..."}`.

```json
{
  "simulation_params": { "num_steps": 1000, "time_per_tick": 0.001, "seed": 1 },
  "channel_params": {
    "phase_screen_size": 1024, "pixel_size": 0.01, "num_layers": 10,
    "total_r0": 0.2, "wavefront_wavelength": 1.55e-6, "L0": 25.0,
    "ground_wind_speed": 5.0,
    "satellite_orbit_altitude": 600000.0, "satellite_zenith_angle": 0.0
  }
}
```

`opal config -c my.json` prints the fully resolved config, and
`opal config -c my.json --set r0=0.1 -o new.json` saves a new one. Every config
is validated when it is loaded:
- an unknown key is an error with a "did you mean" suggestion;
- the keys renamed in 1.1 (`satellite_orbit`, `satellite_zenith`) are rejected
  with the new name;
- values are checked for type and range.

### 3.2 Simulation and channel parameters

| Key | Short alias | Unit | Default | Meaning |
|---|---|---|---|---|
| `simulation_params.num_steps` | `steps` | ticks | 1000 | Ticks to simulate. One tick = one received frame |
| `simulation_params.time_per_tick` | `dt` | s | 1e-3 | Simulated time between two frames |
| `simulation_params.seed` | `seed` | – | `null` | Master seed. Per-layer seeds are derived from it. `null` draws a random seed and records it in the manifest |
| `channel_params.phase_screen_size` | `N`, `grid` | pixels | 512 | Grid size *N* (even). The aperture is *N*·`pixel_size` |
| `channel_params.pixel_size` | `dx` | m | 0.01 | Grid spacing |
| `channel_params.num_layers` | `layers` | – | 10 | Number of phase-screen layers. Each carries an equal share of the path turbulence |
| `channel_params.total_r0` | `r0` | m | 0.2 | Fried parameter of the whole path, at the wavelength below |
| `channel_params.wavefront_wavelength` | `wavelength`, `wvl` | m | 1.55e-6 | Optical wavelength |
| `channel_params.L0` | `L0` | m | 25 | Von Kármán outer scale. Keep `L0 / pixel_size` ≤ 2500 ([section 3.6](#36-numerical-limits)) |
| `channel_params.ground_wind_speed` | `wind` | m/s | 5 | Ground wind speed in the Bufton wind model |
| `channel_params.satellite_orbit_altitude` | `orbit` | m | 6e5 | Circular-orbit altitude of the satellite. Sets the apparent wind caused by the satellite slewing across the sky |
| `channel_params.satellite_zenith_angle` | `zenith` | deg | 0 | Zenith angle (0 = overhead) at which that slew rate is evaluated. It changes the slew-induced wind only; `total_r0` is used as given and the propagation distances stay vertical. For a full pass with elevation-dependent r₀ use `opal pass` |

The layer heights, per-layer r₀, C<sub>n</sub>² and winds are derived from these
values. `opal plan` prints them per layer.

### 3.3 Pass parameters (`pass_params`)

Used only by `opal pass` ([section 5.2](#52-satellite-passes-opal-pass)). Set them
in the file or with `--set pass_params.KEY=VALUE`.

| Key | Unit | Default | Meaning |
|---|---|---|---|
| `altitude_m` | m | `null` | Orbit altitude of the pass. `null` uses `channel_params.satellite_orbit_altitude` |
| `min_elevation_deg` | deg | 20 | The pass runs from this elevation through zenith (90°) and back down |
| `num_segments` | – | 30 | Number of constant-r₀ pieces the pass is split into. Must divide `num_steps` |
| `pass_num_samples` | – | 1024 | Points used to sample the pass geometry before it is mapped onto ticks |
| `pass_window_s` | s | `null` | Simulate only ±window/2 around zenith (`--window`). `null` = the whole pass |
| `wavefront_captures` | frames | 200 | Received fields saved, evenly spaced over the run (at most one per tick) |
| `capture_to_host` | – | `true` | Stream captures to host memory/disk instead of keeping them on the GPU |
| `regenerate_on_switch` | – | `false` | Draw a fresh screen at every segment switch instead of keeping the current one |
| `match_initial_r0` | – | `true` | Rescale the starting screens to the first segment's r₀ (see [section 5.2](#52-satellite-passes-opal-pass)) |
| `validation_guard_frac` | – | 0 | Skip this fraction of each segment, after a switch, when comparing measured with target r₀ |

A complete pass config (`configs/leo_pass_b2.json`):

```json
{
  "simulation_params": { "num_steps": 3000, "time_per_tick": 0.001, "seed": 20260716 },
  "channel_params": {
    "phase_screen_size": 512, "pixel_size": 0.01, "num_layers": 10,
    "total_r0": 0.25, "wavefront_wavelength": 1.55e-6, "L0": 25.0,
    "ground_wind_speed": 5.0,
    "satellite_orbit_altitude": 700000.0, "satellite_zenith_angle": 0.0
  },
  "pass_params": {
    "altitude_m": 700000.0,
    "min_elevation_deg": 20.0,
    "num_segments": 30,
    "pass_num_samples": 1024,
    "pass_window_s": null,
    "wavefront_captures": 200,
    "capture_to_host": true,
    "regenerate_on_switch": false,
    "match_initial_r0": true,
    "validation_guard_frac": 0.0
  }
}
```

### 3.4 Overrides and aliases

Short aliases work everywhere overrides are accepted:
- `--set KEY=VALUE` on the CLI (repeatable; `--set pass_params.num_segments=20`
  for pass keys);
- `load_config(path, overrides={...})`;
- dataset sweeps (`--sweep r0=0.05:0.2:4`).

`--grid`, `--layers`, `--r0`, `--seed` and `--steps` are shortcuts for the most
common ones.

### 3.5 Ready-made configs

| File | Case |
|---|---|
| `configs/quickstart_256.json` | 256², 10 layers, r₀ = 10 cm, 200 ticks: seconds to build, for trying things out |
| `configs/default_1024.json` | 1024², 10 layers: the paper's reference resolution |
| `configs/leo_pass_b2.json` | 700 km LEO overhead pass, 20°–90°–20° elevation, 30 exact r₀ segments (`opal pass`) |
| `configs/fiber_100us_b3.json` | 1024², 100 µs ticks, fiber-coupling case (0.8 m aperture) |

### 3.6 Numerical limits

- **Grid size.** `N` must be even.
- **Outer scale versus pixel size: `L0 / pixel_size` ≤ 2500.**
  - The extrusion operators are built from the phase covariance between the new
    row and the last rows of the screen. The covariance is computed in float32,
    as in the AOtools reference implementation the method comes from.
  - When `L0` is very large compared with the pixel, neighbouring covariance
    values become almost identical. The matrix then becomes numerically
    singular in float32, and the operator that predicts new rows stops being
    stable: rows grow without bound and the field turns into `NaN` after a few
    hundred rows.
  - Measured on N = 128–2048: every `L0 / pixel_size` up to 2500 is stable.
    Failures start between 3000 and 5000, depending on N. The default
    (L0 = 25 m, 1 cm pixels) sits exactly at the safe limit.
  - OPAL checks every operator when it builds it (a short power iteration on
    the extrusion map, ≈ 0.25 s per layer at 1024²). If an operator would be
    unstable, the run stops at once with a message like the one below, instead
    of producing `NaN`s:

    ```text
    opal run: error: layer r0=0.0864 m, L0=50 m, pixel 0.01 m (L0/pixel=5000), N=128: the extrusion
    operator is unstable (rows grow by x4.51 per extruded row). The covariance of the extrusion
    stencil is too ill-conditioned in float32 when L0 is much larger than the pixel. Use
    L0 <= 2500 x pixel (25 m here) via channel_params.L0, or a coarser channel_params.pixel_size.
    ```

  - Remedies: lower `L0`, or use a coarser `pixel_size`. In the far field the
    outer scale matters little once it is ≫ the aperture.
- **Wind and tick length.** Screens move by whole rows. See the extrusion
  multiplier in [section 15](#15-performance-model).

---

## 4. Run modes and the memory guard

### 4.1 Modes

One complex64 frame is 8·N² bytes: 8 MiB at 1024², 32 MiB at 2048². A long,
densely captured run therefore outgrows GPU memory quickly. OPAL offers four
ways to run the *same* channel. For a given seed they produce **bit-identical
frames**.

| Mode | Where frames go | GPU memory | Use it for |
|---|---|---|---|
| `device` | Accumulate on the GPU, copied out at the end | grows ≈ 2 frames per tick | Short bursts that fit on the GPU; frames stay on the device for further in-graph processing |
| `host` | Streamed per tick to host RAM, or to a memory-mapped `frames.npy` on disk | **flat** | Long, dense captures; the default for most saved runs |
| `chunked` | Written to disk in bounded `frames_chunk_*.npy` files | bounded by the chunk size | Beyond RAM; exported SavedModels; machines with little RAM |
| `no_save` | Only the final field is kept | minimal | Benchmarks, warm-up, in-graph statistics, Monte-Carlo draws |
| `auto` | The guard picks `device` → `host` → `chunked` → `no_save` | | Default |

### 4.2 How the guard decides

The guard measures the simulator's real static footprint with one traced step,
and predicts each mode's peak:
- `device`: baseline + 2 × output + margin;
- `host`: baseline + 4 frames + margin;
- `chunked`: baseline + 2 × chunk + margin.

The margin is max(1.5 GiB, 20 % of the GPU). The guard also checks host RAM:
the output must fit in 60 % of the available RAM, unless it is written to a
memory-mapped file on disk. It then picks the first safe mode and prints why. If
a mode still hits an out-of-memory error, the run falls back along
`device → host → chunked (halving the chunk) → no_save` and records it in the
manifest.

```bash
opal plan -c configs/default_1024.json --steps 20000      # dry run: which mode, and why
opal run  -c configs/default_1024.json --steps 20000 --mode host -o runs/long
opal run  -c configs/default_1024.json --steps 20000 --mode chunked --chunk-size 100 -o runs/chunks
```

Decisions on an 8 GiB laptop GPU with ≈ 9 GiB of free RAM:

| Run | Output size | `auto` picks |
|---|---|---|
| 256², 50 ticks | 0.02 GiB | `device` |
| 512², 1000 ticks | 1.95 GiB | `device` |
| 1024², 1000 ticks | 7.8 GiB | `host` (memory-mapped `frames.npy`) |
| 2048², 1000 ticks | 31 GiB | `host` (memory-mapped `frames.npy`) |
| any of the above with `--no-save` | | `no_save` |

### 4.3 Measured

**RTX 4080, 10 layers, 512², seed 20260716** (peak GPU memory, including the
≈ 150 MiB resident simulator state):

| Captured ticks | `device` | `host` | `chunked` (100) | `no_save` |
|---|---|---|---|---|
| 100 | 501 MiB · 5.8 ticks/s | 151 MiB · 5.7 ticks/s | 503 MiB · 5.9 ticks/s | 161 MiB · 5.8 ticks/s |
| 300 | 1308 MiB · 5.9 ticks/s | 152 MiB · 6.1 ticks/s | 501 MiB · 5.6 ticks/s | 153 MiB · 5.7 ticks/s |
| 1000 | 4103 MiB · 5.8 ticks/s | 163 MiB · 6.1 ticks/s | 507 MiB · 6.1 ticks/s | 158 MiB · 6.0 ticks/s |

Throughput is the same in every mode, because the device-to-host copy overlaps
with compute. Only the memory differs: device capture grows by two frames
(4 MiB) per tick, while host streaming stays flat.

**Guard decisions from the paper** (1000 ticks, 15 layers, RTX 4080):

| Grid | Selected mode | Wall time | Peak GPU |
|---|---|---|---|
| 128² | device | 279.8 s | 0.25 GiB |
| 512² | device | 210.5 s | 4.07 GiB |
| 1024² | chunked (100) | 196.2 s | 2.90 GiB |
| 2048² | chunked (25) | 219.9 s | 4.96 GiB |

Host streaming keeps GPU memory flat at any length. At 1024² with 10 layers,
1000 dense ticks (7.8 GiB of frames) run in one 164 s call at a flat 638 MiB,
against 24 MiB per tick for device capture.

### 4.4 Large outputs on machines with little RAM

In `host` mode with an output directory, frames go into a memory-mapped
`frames.npy`. The operating system keeps recently written pages in RAM and
writes them back to disk in the background. When the output is larger than the
free RAM (7.8 GiB of frames with 8–9 GiB free, for example):
- the run slows down gradually as the OS starts writing pages back while the
  simulation runs;
- the final flush (`wrote frames.npy ... final flush to disk took X s`) can take
  tens of seconds;
- the rest of the machine becomes sluggish while it happens.

OPAL prints a note when this applies. Two ways around it:
- `--mode chunked --chunk-size 100` writes bounded files (100 frames each), so
  memory stays bounded;
- a fast SSD and a coarser `--steps`/`dt` reduce the volume.

Analysis and viewing never load a whole run, so the result can always be
inspected afterwards ([section 8.5](#85-memory-behaviour)).

---

## 5. Dynamic turbulence: schedules and satellite passes

The configured channel has a fixed strength. OPAL can also change the turbulence
strength while the simulation runs, in two ways:

| | Turbulence schedule (`opal run --schedule`) | Satellite pass (`opal pass`) |
|---|---|---|
| Resolution in time | every tick, every layer | per segment (e.g. 30 per pass) |
| What changes | the strength of newly extruded rows (approximate statistics) | the extrusion operators themselves (exact von Kármán statistics at the new r₀) |
| Cost | free (inside the compiled graph) | one CPU operator rebuild per layer per segment (≈ 5 s at 512², ≈ 15 s at 1024², 10 layers) |
| Use it for | fades, ramps, gusts, moderate changes in strength, quick what-ifs | full passes where r₀ changes by an order of magnitude |

### 5.1 Turbulence schedules

**What a schedule is.** A table of numbers with one row per tick and one column
per layer. Row *t* sets the turbulence strength of every layer at tick *t*, so
the strength can change every tick, and each layer can follow its own curve.
Schedules are independent of pass segments.

Accepted shapes:
- `[T, L]`: one value per tick and per layer;
- `[T, 1]` or `[T]`: one value per tick, applied to every layer;
- a single row: held constant for the whole run.

A schedule shorter than the run holds its last row; a longer one is truncated.
It can be an `.npy`, `.npz`, `.csv` or `.json` file, or an array in Python.

**How the simulator uses it.** Each layer's screen moves with the wind by
*extrusion*: every tick, new rows are added on one side of the screen and the
same number drop off the other side. Each new row is predicted from the last
rows of the screen plus fresh randomness:

```
new_row = A · (last rows of the screen) + b · B · ξ        ξ ~ N(0, 1)
```

*A* and *B* are fixed matrices computed once from the von Kármán covariance at
the layer's r₀, and the second term is the *innovation*, the new turbulence
entering the screen. The schedule sets the multiplier **b** on that term
(`GpuLayer.get_new_row` in `layer.py`; `set_b_scale` stores it):
- the phase variance of new rows scales with b², so a layer's C<sub>n</sub>²
  scales by b²;
- r₀ ∝ (C<sub>n</sub>²)<sup>−3/5</sup>, so r₀ scales by (b²)<sup>−3/5</sup> = b<sup>−6/5</sup>;
- inverting gives the b needed for a target r₀: **b = (r₀,base / r₀)<sup>5/6</sup>**.

b = 1 is the configured channel; b = 2 means 4× the C<sub>n</sub>² and a 2.3×
smaller r₀; b = 0 freezes the screen (no new turbulence enters).

`--schedule-kind` says what the numbers in the file mean. OPAL converts them to
b:

| Kind | The numbers are | Converted to b by |
|---|---|---|
| `b_scale` (default) | the multiplier b itself | b |
| `cn2` | a C<sub>n</sub>² multiplier *m* per layer | b = √*m* |
| `r0` | the absolute r₀ (m) of each layer | b = (r₀,layer base / r₀)<sup>5/6</sup> |
| `path_r0` | the absolute total-path r₀ (m), one value per tick | b = (r₀,path base / r₀,path)<sup>5/6</sup>, the same for every layer |

**What to expect.** A schedule acts on *new* rows only. Rows already in the
screen keep the strength they were made with until they scroll out. So after a
change, a layer reaches the new strength gradually, over about
`N / rows_extruded` ticks (`rows_extruded` per layer is printed by `opal plan`).
Fast, high layers refresh in a few ticks; the slowest, lowest layer can take
hundreds. This matches turbulence being advected across the aperture by the
wind. Changes that are slow compared with this refresh time are reproduced
accurately, and sudden steps appear smoothed. For exact statistics after large
jumps, use `opal pass`.

**Worked example: r₀ drops from 10 cm to 5 cm over 1 s.**

```python
import numpy as np
np.save("ramp.npy", np.linspace(0.10, 0.05, 1000, dtype=np.float32))   # path r0 per tick (m)
```

```bash
opal run -c configs/quickstart_256.json --steps 1000 --schedule ramp.npy --schedule-kind path_r0 -o runs/ramp
opal analyze r0 runs/ramp --frames 50 --plot      # measured r0 vs the scheduled target, per frame
opal view runs/ramp                               # frames with the r0 curve underneath
```

At the end of the ramp b = (0.10/0.05)<sup>5/6</sup> = 1.78, so the
C<sub>n</sub>² is 3.2× the configured one. The run directory gets
`schedule.npy`, the b-scale actually applied (`[T, L]`). `opal analyze` reads it
to add the scheduled target r₀ to every frame. The measured curve trails the
target by the refresh time described above.

### 5.2 Satellite passes (`opal pass`)

During a pass the r₀ of every layer changes by an order of magnitude between the
horizon and zenith. `opal pass`:

1. builds the circular-orbit pass geometry (elevation, range, slew rate);
2. derives each layer's slant-path r₀ along it;
3. splits the pass into `num_segments` segments of equal length.

At each segment boundary every layer's extrusion operators *A* and *B* are
**rebuilt for the new r₀**, so the new rows have exact von Kármán statistics at
that r₀. The current screens are kept, so the field evolves continuously while
its statistics follow the pass. The whole pass, or a `--window` in seconds
around zenith, is mapped onto the run's ticks. The captured wavefronts are
streamed to a memory-mapped file, and their r₀ is measured against the target.

Two `pass_params` control what happens to the screens themselves:

- **`match_initial_r0`** (default `true`). The screens are first created at the
  r₀ of the configured channel, but the pass starts at a low elevation where r₀
  is different. Without correction the first captures would show the
  configured r₀, and the screens would only reach the pass's r₀ once new rows
  had replaced the old ones.
  - With `true`, OPAL rescales each starting screen to the first segment's r₀
    before the pass begins.
  - The rescaling is exact: a von Kármán phase screen generated from the same
    random numbers at another r₀ is the same screen multiplied by
    (r₀,new / r₀,old)<sup>−5/6</sup>, since phase ∝ r₀<sup>−5/6</sup>. So the
    statistics are right from the first capture.
  - Set it to `false` to reproduce the paper's results bit for bit.
- **`regenerate_on_switch`** (default `false`). Draws a completely new screen at
  every switch instead of keeping the current one. The statistics change
  instantly, but the field jumps at each boundary. The new screens are seeded
  from `(seed, layer, segment)`, so runs stay reproducible.

With kept screens, a layer needs about `N / rows_extruded` ticks after a switch
to refresh fully. Use segments long enough for the slow, low layers, e.g. ≥ 100
ticks per segment at 256².

```bash
opal pass -c configs/leo_pass_b2.json -o runs/b2                     # 30 segments, 200 captures
opal pass -c configs/leo_pass_b2.json -o runs/b2_zen --window 120 --segments 10 --steps 1000
opal view runs/b2                                                    # wavefronts + target/measured r0
```

`runs/b2/r0_tracking.csv` lists target vs measured r₀ per capture. In the paper's
B2 case (512², 3000 ticks, 30 segments) the median measured/predicted ratio is
0.99. On small grids (N ≤ 128) the MCF estimator reads r₀ high
(ratio ≈ 1.2–1.7), because the aperture is only a few r₀ across.

---

## 6. Checkpoints and bit-exact segmentation

The complete simulator state fits in one `.npz`:
- every layer's screen;
- every per-layer RNG state;
- the tick counter;
- the current turbulence scales.

A run continued from a checkpoint is bit-identical to an uninterrupted one:
2 × 500 = 1 × 1000, across processes and run modes.

```bash
# Run in 50-tick pieces: resume from the file if it exists, then overwrite it
opal run -c configs/quickstart_256.json --steps 50 --checkpoint state.npz -o runs/part1
opal run -c configs/quickstart_256.json --steps 50 --checkpoint state.npz -o runs/part2

# Or explicitly
opal run -c cfg.json --steps 1000 --state-out s1.npz
opal run -c cfg.json --steps 1000 --state-in s1.npz --state-out s2.npz
```

```python
opal.run(sim, 1000, mode="host", state_out="s1.npz")
opal.run(sim, 1000, mode="host", state_in="s1.npz")       # continues at tick 1000
```

`--capture 0,500,999` also saves every layer's screen, r₀ and b-scale at those
ticks into `captures.npz`.

---

## 7. ML dataset generation

Measured atmospheric channel data is scarce, expensive and unlabelled. OPAL
turns a config into an arbitrarily large, **labelled, reproducible** dataset of
received fields, for training and testing:
- wavefront-sensing and adaptive-optics models;
- fading and outage predictors;
- pointing/tracking loops;
- channel estimators.

```bash
opal dataset -c configs/quickstart_256.json -o datasets/r0_wind \
    --sweep r0=0.05:0.20:4 --sweep wind=5,10,20 -k 8 \
    --steps 200 --stride 5 --crop 128 --targets intensity,phase
```

This generates 4 r₀ values × 3 wind speeds × 8 independent realizations = 96
samples. Each sample is a 40-frame, 128² irradiance and phase sequence (every
5th tick of 200).

- **Sweeps.** `--sweep KEY=lo:hi:n` (linear) or `KEY=v1,v2,...`. Repeat it for a
  Cartesian product. Any config alias works, plus `elevation` (degrees; sets
  `satellite_zenith_angle = 90 − elevation`, which changes the slew-induced
  wind only, see [section 3.2](#32-simulation-and-channel-parameters)).
- **Targets.** `field` (complex), `intensity`, `phase`.
- **Per-frame labels**, over a `--label-aperture` in m (default 40 % of the
  grid):
  - `eta`: single-mode fiber coupling efficiency;
  - `scint`: scintillation index;
  - `tilt`: angle of arrival in rad, from the far-field centroid.
- **Per-sample labels:**
  - the sweep and channel parameters;
  - D/r₀;
  - `r0_measured` (median r₀ measured over `--r0-frames` frames);
  - mean/std of eta, mean scintillation, rms tilt.
- **Reproducible.** Sample *k* uses the seed `SeedSequence([base_seed, point,
  realization])`, recorded in `labels.csv` and `manifest.json`.
- **Resumable.** Running the same command with the same `-o` again skips the
  finished samples. `manifest.json` says `"status": "incomplete"` until the last
  sample is written.
  - Without `--seed`, a resume reuses the stored base seed.
  - Different settings (sweep, steps, stride, crop, targets, config…) are an
    error that lists the differences, so a dataset never mixes settings.
  - `--no-resume` deletes the old samples and starts over.
- **Efficient.** One simulator is built per sweep point, and realizations reuse
  it via `reseed` (no graph rebuild). Frames are streamed from the GPU and never
  accumulate on it.

Layout and loading:

```
datasets/r0_wind/
  manifest.json            status, base seed, seed rule, sweep, config, targets, timing
  labels.csv               one row per sample
  samples/sample_00000.npz intensity [F,n,n] float32, phase [F,n,n] float32,
                           steps [F], eta [F], scint [F], tilt [F,2], labels_json
```

```python
import opal

X, y = opal.load_dataset("datasets/r0_wind", target="intensity", label_keys=("r0", "eta_mean"))
# X: [samples, frames, n, n]   y: [samples, 2]

ds = opal.as_tf_dataset("datasets/r0_wind", target="phase", label_keys=("r0",), per_frame=True)
model.fit(ds.shuffle(1024).batch(32), epochs=10)

for sample in opal.iter_samples("datasets/r0_wind"):     # streaming, one .npz at a time
    ...
```

The loaders refuse incomplete datasets and unknown targets or label keys with a
clear message. `opal view datasets/r0_wind` browses the samples
([section 9](#9-visualization-opal-view)).

The optional last section of `examples/05_ml_dataset.ipynb` (set `TRAIN = True`)
fits a small CNN that reads r₀ off single wrapped-phase frames (fed as cos/sin) of
the notebook's r₀ × wind dataset. On held-out realizations it reaches ≈ 0.5 cm
mean absolute error, against ≈ 2.7 cm for always predicting the mean.

---

## 8. Analysis

### 8.1 Commands

```text
opal analyze r0    RUN [--frames N|all] [--start I] [--stop J] [--stride K] [--aperture D] [-o DIR] [--no-save] [--plot] [--batch-mb MB] [-q]
opal analyze fiber RUN [--frames N|all] [--start I] [--stop J] [--stride K] [--aperture D] [--a A] [-o DIR] [--no-save] [--plot] [--batch-mb MB] [-q]
```

- **`r0`** measures the Fried parameter of each selected frame from the 1/e
  point of its mutual coherence function (MCF), with a von Kármán correction for
  the outer scale. `--aperture` (m) restricts the measurement to a central disk
  (default: the full grid).
- **`fiber`** computes the single-mode-fiber coupling efficiency η of each frame
  (Dikmelik & Davidson overlap with the matched fiber mode) over an aperture of
  `--aperture` m (default 40 % of the grid). `--a` is the coupling geometry
  D/(2w₀) (default 1.12, the optimum for a plane wave). The result is compared
  with von Kármán theory at the configured r₀, or at the per-frame target r₀ for
  passes and scheduled runs.

`RUN` can be a run directory (device, host, chunked or `no_save`), a pass
directory or a frames `.npy` file.

```bash
opal analyze fiber runs/qs --frames all --plot            # every frame, with a plot
opal analyze r0 runs/qs --frames 50                       # 50 evenly spaced frames
opal analyze r0 runs/b2 --start 1000 --stop 2000 --stride 10
opal analyze fiber runs/qs --aperture 0.3 --no-save       # print only
```

### 8.2 Frame selection

| Option | Effect |
|---|---|
| `--frames N` | N evenly spaced frames of the selected range (default 20) |
| `--frames all` | every frame of the range |
| `--start I`, `--stop J` | restrict to frames I … J−1 |
| `--stride K` | every K-th frame of the range (combine with `--frames all`) |

### 8.3 Outputs and their format

Results go to `RUN/analysis/` (or `-o DIR`):

| File | Content |
|---|---|
| `r0.csv` | `frame, step, r0_m[, target_r0_m]`: one row per analysed frame |
| `fiber.csv` | `frame, step, eta[, target_r0_m, eta_theory]` |
| `r0_summary.json`, `fiber_summary.json` | settings (frames, aperture, a), statistics (count, mean, std, median, min, max), theory and deviation from it, frames with no finite value, OPAL version, timing |
| `r0.png`, `fiber.png` (with `--plot`) | time series against the tick plus a histogram, with the theory/target drawn in |

`target_r0_m` appears when the run has a target per frame: the pass target from
`r0_tracking.csv`, or the scheduled path r₀ from `schedule.npy`. The console
prints the same statistics. CSV and JSON load with any tool:

```python
import pandas as pd, json
df = pd.read_csv("runs/qs/analysis/fiber.csv")
summary = json.load(open("runs/qs/analysis/fiber_summary.json"))
# or: opal.load_analysis("runs/qs", "fiber") -> (steps, values, extra_columns)
```

### 8.4 Viewing the results

- `opal view RUN` shows the analysis next to the frames: a strip under the
  images plots η and/or r₀ (and the theory or target) against the tick, with a
  cursor at the current frame.
- `--plot` writes a static PNG.

### 8.5 Memory behaviour

`opal analyze` never loads a run into memory. Frames are read from the files
(memory-mapped `frames.npy`, chunk files or `wavefronts.npy`) in batches of at
most `--batch-mb` MiB (default 256), processed, and released. Analysing all
1000 frames of a 7.8 GiB 1024² run keeps the working set around 0.25 GiB.

### 8.6 Ensembles

The frames of a single run are correlated in time. For ensemble statistics, use
independent realizations: `opal dataset -k K`, or `GpuSimulator.reseed` in a
loop (see `examples/06_fiber_coupling.ipynb`).

### 8.7 Python analysis functions

`opal.analysis` (no TensorFlow needed) provides:

| Function | Purpose |
|---|---|
| `analyze_run(path, what, frames=..., stride=..., start=..., stop=..., aperture=..., plot=...)` | The engine behind `opal analyze` |
| `estimate_r0_from_wavefront(field, dx, aperture=None, L0=...)` | r₀ from the 1/e point of the measured mutual coherence function, von Kármán corrected |
| `compute_fiber_coupling(fields, dx, a=1.12, aperture_diameter=..., batch_size=None)` | Single-mode-fiber coupling efficiency per frame (Dikmelik & Davidson), batched |
| `coupling_efficiency_theoretical_vk(D, r0, L0, a=1.12)` | Theoretical mean coupling under a von Kármán MCF |
| `remove_tilt_com(fields, dx, aperture_diameter=...)` | Centroid tip/tilt removal (ideal fast-steering mirror) |
| `scintillation_index`, `tilt_centroid`, `aperture_mask` | Irradiance and angle-of-arrival statistics |
| `circular_orbit_pass`, `per_layer_r0_along_pass`, `effective_path_r0` | Pass geometry and slant-path turbulence |

Validation from the paper:
- the measured r₀ tracks the prediction to a median ratio of 0.99 over a full
  pass;
- Monte-Carlo fiber coupling agrees with von Kármán theory to within ±7 %.

---

## 9. Visualization (`opal view`)

### 9.1 Usage

```text
opal view PATH [--views V1,V2,...] [--start I] [--stop J] [--stride K] [--frame I | --tick T]
               [--sample S] [--no-series] [--save OUT.gif|OUT.mp4|OUT.png] [--fps F] [--max-frames M]
```

`PATH` can be:
- a run directory (single `frames.npy`, chunked, or `no_save` with only the
  final frame);
- a pass directory (shows the target and measured r₀);
- a dataset directory or a single `sample_XXXXX.npz`;
- a frames `.npy` file.

Frames are read one at a time, so a run of any size opens instantly.

| Option | Effect |
|---|---|
| `--views` | panels to show: `intensity`, `log_intensity`, `phase`, `phase_tt` (phase with tip/tilt removed) or `all`. Default `intensity,phase` |
| `--start`, `--stop`, `--stride` | which frames to step through |
| `--frame I` / `--tick T` | open at frame index I, or at the frame nearest simulation tick T |
| `--sample S` | dataset sample to open (default 0) |
| `--no-series` | hide the time-series strip |
| `--save FILE` | render instead of opening a window: `.gif`, `.mp4` (needs ffmpeg) or `.png` (the frame from `--frame`/`--tick`) |
| `--fps`, `--max-frames` | playback rate; frame limit for saved animations (default 300, 0 = none) |

### 9.2 Window controls

| Control | Action |
|---|---|
| Play button / space | play / pause |
| ← / → | previous / next frame |
| ↓ / ↑ (or PgDn / PgUp) | 10 frames back / forward |
| Home / End | first / last frame |
| slider | scrub through the selected frames |
| "go to tick" box | type a tick and press Enter |
| `t` | toggle tip/tilt removal on the phase panels |
| `l` | toggle linear / log irradiance |
| `s` | snapshot: PNG of the window plus the raw complex frame as `.npy`, written to `<source>/snapshots/` |
| `n` / `p` | next / previous sample (datasets) |

The title shows the frame index, the simulation tick, the time and, for passes,
the target r₀. The strip under the images shows whatever post-processed data
exists for the source, with a cursor at the current frame:
- `opal analyze` results (`analysis/fiber.csv`, `analysis/r0.csv`, with theory
  or target);
- a pass's target and measured r₀ (`r0_tracking.csv`);
- a dataset sample's per-frame labels (η, scintillation, tilt).

### 9.3 Saving movies and snapshots

```bash
opal view runs/qs --save qs.gif --fps 15                    # animation
opal view runs/qs --views all --stride 5 --save qs.mp4      # all four panels, every 5th frame
opal view runs/qs --tick 500 --save tick500.png             # one frame
opal view datasets/r0_wind --sample 12 --save s12.gif
```

Saving renders frame by frame and needs no display. Without a display (e.g. SSH)
the interactive window cannot open, and OPAL suggests `--save` instead.

---

## 10. SavedModel export

### 10.1 What it is and why

`opal export` freezes a configured simulator into a TensorFlow SavedModel: a
directory with the compiled simulation graphs and every tensor they need. You
can then run the *same* channel:
- on another machine, without OPAL or its Python dependencies installed (plain
  TensorFlow, TF Serving, or the C++ API);
- inside another TensorFlow program, for example as a channel layer in an
  end-to-end-trained optical communication or wavefront-sensing model;
- with `opal run --model`, skipping the CPU build step (the operators are
  already computed).

```bash
opal export -c configs/default_1024.json -o exports/opal_1024
opal run --model exports/opal_1024 --steps 500 -o runs/from_model
```

### 10.2 What is inside

- each layer's base screen, RNG state, and its *A*/*B* extrusion operators;
- the propagator's precomputed phase factors and absorbing boundary;
- the signatures listed below;
- `opal_config.json`: the config, seed, grid, layer count and per-layer r₀ it
  was exported with.

| Signature | Inputs | Outputs |
|---|---|---|
| `simulate_steps` | `num_steps` (int32), `b_scale_schedule` (float32 `[T or 1, L]`) | `output_0`: frames complex64 `[T, N, N]` |
| `simulate_steps_no_save` | same | `output_0`: final frame `[N, N]` |
| `simulate_steps_capture` | + `capture_indices` | `frames`, `captured_screens`, `captured_r0`, `captured_b_scales`, … |
| `simulate_steps_sparse_capture` | + `capture_indices`, `wavefront_indices` | `captured_wavefronts`, `final_frame`, … |
| `get_state` / `reset_to_base` | – | `phase_screens`, `rng_states`, `global_step`, `b_scales`, `r0`, `base_r0` |
| `set_state` | `phase_screens`, `rng_states`, `global_step`, `b_scales` | the new state |
| `set_b_scale_profile` | `b_scales` `[L]` | the applied b |

### 10.3 Loading it

With OPAL, `opal.load` wraps the model in the `GpuSimulator` call API, so
`opal.run`, checkpoints and schedules work unchanged:

```python
model = opal.load("exports/opal_1024")
opal.run(model, 500, mode="chunked", out_dir="runs/from_model")
```

With plain TensorFlow only:

```python
import tensorflow as tf

m = tf.saved_model.load("exports/opal_1024")
L = m.signatures["get_state"]()["phase_screens"].shape[0]
frames = m.signatures["simulate_steps"](num_steps=tf.constant(100),
                                        b_scale_schedule=tf.ones([1, L]))["output_0"]   # [100, N, N] complex64
```

### 10.4 Limitations

- The grid, layers, r₀ profile and winds are fixed at export time. A different
  config needs a new export (`opal run --model` refuses config flags for that
  reason). Turbulence schedules still work.
- `host` mode needs a Python callback that cannot be exported, so exported
  models run in `device`, `chunked` or `no_save` mode.
- `opal pass` rebuilds operators on the CPU and needs the live simulator.

---

## 11. Python API

| Object | Description |
|---|---|
| `opal.load_config(path_or_dict=None, overrides=None)` / `opal.save_config` | Load, fill defaults, apply aliases, validate (`None` = defaults) |
| `opal.GpuSimulator(config, seed=None, log_dir=None)` | The simulator (`tf.Module`) |
| `opal.run(sim, num_steps=None, *, mode, out_dir, chunk_size, schedule, schedule_kind, capture_indices, state_in, state_out, log)` | Run with the memory guard; returns a `RunResult` (`frames`, `final_frame`, `mode`, `run_time_s`, `steps_per_second`, `manifest`) |
| `opal.open_frames(path)` → `FrameSeries` | Lazy, indexable view of any run, pass, chunked run or `.npy` (`len`, `shape`, `[i]`, `[a:b]`, `iter_batches`) |
| `opal.load_frames(run_dir)` / `opal.iter_frames(run_dir, stride)` | Memory-mapped array (single file) or `FrameSeries` (chunked); streaming iterator |
| `opal.save_state` / `opal.load_state` | Checkpoints |
| `opal.build_schedule(schedule, T, L, base_r0, kind)` | Normalize a schedule to per-tick b-scales |
| `opal.plan_pass(config)` / `opal.run_pass(config, out_dir)` | Satellite pass with exact r₀ switching |
| `opal.generate_dataset(...)`, `opal.load_dataset`, `opal.iter_samples`, `opal.as_tf_dataset` | ML datasets |
| `opal.analyze_run(path, what, ...)` / `opal.load_analysis(path, what)` | Run analysis, as in `opal analyze` |
| `opal.export(sim, dir)` / `opal.load(dir)` | SavedModel |
| `opal.viewer.view(path, ...)` | The viewer, from Python |

`GpuSimulator` methods (TensorFlow graphs):

| Method | Returns |
|---|---|
| `simulate_steps(n, schedule)` | `[n, N, N]` frames on the GPU |
| `simulate_steps_no_save(n, schedule)` | final `[N, N]` frame |
| `simulate_steps_offload(n, schedule, frame_callback=...)` | streams each frame to the host |
| `simulate_steps_capture` / `simulate_steps_sparse_capture` | frames plus all layer screens at chosen ticks |
| `simulate_pass_replacement(...)` | segmented pass with operator rebuilds |
| `get_state()` / `set_state(...)` / `reset_to_base()` | state control |
| `reseed(seed)` | new independent realization, no rebuild |
| `describe()` | per-layer height, r₀, wind, C<sub>n</sub>², rows per tick, rows extruded, effective wind |

The `schedule` argument is a `[n, L]` float32 b-scale tensor; use
`tf.ones([1, L])` for constant turbulence.

---

## 12. Output formats

```
runs/<run>/
  manifest.json               config, seed, OPAL version, channel description (per-layer table),
                              modes (requested/selected/final), guard reason, timing, GPU memory,
                              memory estimate, schedule, warnings
  frames.npy                  complex64 [T, N, N]           device / host
  frames_chunk_000.npy ...    complex64 [chunk, N, N]       chunked
  frames_chunk_manifest.json  per-chunk step ranges and compute times
  final_frame.npy             complex64 [N, N]              no_save
  schedule.npy                applied b-scale [T, L]        when --schedule is given
  captures.npz                layer screens / r0 / b-scales at --capture ticks
  analysis/                   opal analyze results (section 8.3)
  snapshots/                  frames saved from the viewer with `s`

runs/<pass>/
  wavefronts.npy              complex64 [captures, N, N] (memory-mapped)
  wavefront_steps.npy         tick of each capture
  schedule.npz                per-layer r0 schedule, segment starts, pass geometry
  r0_tracking.csv             per capture: tick, segment, pass time, elevation, target r0, measured r0
  segments.csv                per segment: start tick, elevation, target / measured r0, wall time
  manifest.json
```

Frames are the complex field received at the ground-station pupil. The input is a
unit-amplitude plane wave, and a super-Gaussian absorbing boundary damps the
grid edges. Irradiance is `abs(U)**2`, and phase is `np.angle(U)`. Every `.npy`
opens with `np.load(path, mmap_mode="r")` without reading it all.

`manifest.json → timing` holds `startup_s`, `build_s`, `plan_s`, `simulate_s`,
`write_frames_s`, `run_call_s` and `since_process_start_s`. If the final frame
contains non-finite values, the run logs a warning and lists it under
`warnings`.

---

## 13. Architecture

### 13.1 Module map

```
src/opal/
  cli.py            `opal` command: argument parsing, default output dirs, phase timing, error reporting
  config.py         defaults, aliases, validation, renamed keys
  simulator.py      GpuSimulator: owns the layers and the propagator; all tf.function graphs (tick loop,
                    capture, offload, pass replacement); describe(), reseed(), SavedModel signatures
  geometry.py       GPUGeometry: slices the HV profile into layers, per-layer r0/height/wind/seed,
                    builds one GpuLayer per layer, rows per tick
  profiles.py       Hufnagel-Valley Cn^2, Bufton wind + slew, per-layer r0 and heights
  layer.py          GpuLayer: one screen on the GPU; extrusion (get_new_row/add_row), b-scale,
                    operator replacement for a new r0, reseed
  phase_screens.py  von Karman covariance, extrusion operators A/B (+ stability check),
                    initial FFT + subharmonic screens
  propagation.py    Propagator: angular-spectrum split-step propagation through the layer stack
  runners.py        run(): memory guard, the four modes, fallbacks, schedules, checkpoints, manifest
  memory.py         GPU/RAM measurement and the mode-selection guard
  passes.py         plan_pass/run_pass: pass geometry -> per-layer r0 segments -> simulate_pass_replacement
  dataset.py        sweeps x realizations, labels, resume, loaders
  frames.py         FrameSeries/open_frames: lazy access to every output layout
  viewer.py         interactive viewer and GIF/MP4/PNG rendering
  savedmodel.py     export/load
  analysis/         TF-free tools: r0.py, fiber.py, stats.py, pass_geometry.py, runs.py (opal analyze)
```

### 13.2 What happens in `opal run`

```
cli.main → cmd_run
  ├─ config.load_config            defaults + file + flags, validated
  ├─ default_output_dir            runs/<date>_<params>
  ├─ GpuSimulator(config)                                      [build]
  │    ├─ GPUGeometry → profiles.calc_r0_profile               layer heights, r0, winds
  │    │    └─ GpuLayer × L → PhaseScreenVonKarman             covariance, A/B, stability check, initial screen
  │    ├─ Propagator(N, λ, dx, heights)                        Q1/Q2/Q3 phase factors, absorbing boundary
  │    └─ rows extruded per tick (extrusion multiplier)
  └─ runners.run(sim, ...)
       ├─ build_schedule                                       → b-scale [T, L]
       ├─ memory.measure_baseline_peak_bytes                   [plan] one traced test step
       ├─ memory.estimate_memory + select_mode                 device / host / chunked / no_save
       ├─ sim.simulate_steps / _offload / chunks / _no_save    [simulate] the compiled tick loop
       └─ frames.npy / chunks / final_frame.npy + manifest.json   [write]
```

### 13.3 One tick

Inside the compiled graph (`GpuSimulator.simulate_steps` and its siblings):

1. **Schedule:** row *t* of the b-scale schedule is applied to every layer
   (`set_b_scale_profile` → `GpuLayer.set_b_scale`).
2. **Propagate:** the plane wave travels down from above the top layer. At each
   layer, it is propagated to the layer (angular spectrum) and multiplied by
   `exp(i·screen)` and the absorbing boundary (`Propagator.propagate`). The
   result is the received field, frame *t*.
3. **Store:** the frame is written to the device array, streamed to the host, or
   dropped, depending on the mode.
4. **Advance:** every layer extrudes `rows_extruded` new rows (`GpuLayer.add_row`
   → `get_new_row`), which moves its screen with the wind. The screens used in
   frame *t + 1* are the advanced ones.

---

## 14. Modifying the engine

### 14.1 Where things live

| To change… | Look at | Notes |
|---|---|---|
| Propagation (distances, sampling, boundary, focusing optics) | `propagation.py`, `Propagator.__init__` (precomputed `Q1`, `Q2`, `Q3`, `sg`) and `Propagator.propagate` (per tick) | Precompute everything you can in `__init__` as NumPy, then store it as a `tf.constant`. `propagate` runs inside the graph |
| Phase-screen statistics (spectrum, covariance) | `phase_screens.py`: `phase_covariance` / `_covariance_shape` (used by the extrusion operators), `ft_sh_phase_screen` (initial screen) | The initial screen and the covariance must describe the same statistics, otherwise the screen relaxes after start-up |
| How screens move (extrusion operators) | `phase_screens.PhaseScreenVonKarman`: `make_covmats`, `makeAMatrix`, `makeBMatrix` | Built on the CPU once per layer. Keep the stability check |
| Per-tick screen shift | `layer.GpuLayer.get_new_row` / `add_row`, and `GpuSimulator._tf_move_one_tick_all_layers` (how many rows per layer per tick) | `tf_extr_per_tick` holds rows per tick per layer |
| Turbulence profile, heights, winds | `profiles.calc_r0_profile` (HV C<sub>n</sub>², Bufton wind, slew) and `geometry.GPUGeometry` | Changes the build only, not the graph |
| Wind/time scaling (extrusion multiplier) | `simulator.GpuSimulator.__init__` (`extrusion_mult`, `tf_extr_per_tick`) | See [section 15](#15-performance-model) |
| What is saved per tick | `simulator.simulate_steps*`, `runners.run` (modes and files) | All modes must stay bit-identical |
| Pass segmentation and operator rebuilds | `passes.py`, `simulator.simulate_pass_replacement`, `layer.GpuLayer.replace_matrices_for_r0` | |
| New CLI option | `cli.build_parser` and the matching `cmd_*` | |

### 14.2 Rules that keep the engine correct

- **Graphs.** Code in a `@tf.function` runs as a graph, not as Python per tick.
  Use `tf` ops, not NumPy, inside it, and keep shapes static (`N`, `L`). Python
  `for` loops over layers are unrolled at trace time, while `tf.range` loops
  stay loops.
- **Dtypes.** Screens are float32 radians and fields complex64. Constants have
  to match, because TensorFlow does not up-cast.
- **State.** Everything that evolves (screens, RNG states, tick counter,
  b-scales) lives in `tf.Variable`s and goes through `get_state`/`set_state`.
  Anything new that evolves must be added there, or checkpoints and the
  bit-exact segmentation break.
- **Randomness.** Each layer has its own `tf.random.Generator`. Do not draw
  random numbers from a global RNG, or results stop being reproducible from the
  seed.
- **Check after a change:** re-run the notebooks `examples/02_run_modes.ipynb`
  (all modes must give identical frames) and `examples/03_checkpoint_resume.ipynb`
  (bit-exact segmentation), e.g. headless with
  `jupyter nbconvert --to notebook --execute --inplace`. Then compare
  `opal analyze r0` and `opal analyze fiber` with theory on a run with known r₀.

---

## 15. Performance model

On the RTX 4080, with 10 layers at 1024²:

- **Wall time:** t ≈ 0.123·T + 15.5·N<sub>seg</sub> s.
  - The first term is GPU propagation, per tick, through all planes, including
    screen extrusion.
  - The second is the per-segment operator rebuild of `opal pass`: a CPU
    covariance build + SVD per layer. At 512² this is ≈ 5 s per segment.
    (OPAL 1.0, and the paper, needed 53.9 s at 1024² and ≈ 14 s at 512²; since
    1.1 the r₀-independent part of the covariance is computed once and
    reused.)
- **Step time depends on rows, not pixels.** Each extruded row is one small
  matrix-vector product launched separately, so the step time grows with the
  total number of rows extruded per tick over all layers. It barely grows with
  N: per-tick time is nearly flat from 128² to 2048², at 0.12–0.28 s. Below
  ~2048² the GPU is launch-bound rather than FLOP-bound, so larger grids are
  almost free, and a 256² grid is not much faster than 1024².
- **Extrusion multiplier.** Screens advance by whole rows. A layer physically
  moves `v·dt/dx` rows per tick (often < 1 for the slow, low layers). OPAL
  multiplies every layer's rate by an integer factor so the slowest layer
  advances at least one row per tick, then rounds up:
  - the factor is `ceil(1 / r)` when the lowest layer's rate *r* is below 1,
    and `ceil(r)` otherwise;
  - every screen therefore moves `factor` times faster than the physical wind
    (×2 for the default 10-layer channel);
  - `opal plan` and `describe()` print the physical `rows_per_tick`, the actual
    `rows_extruded` and the resulting `effective_wind_mps` per layer, and every
    run logs the factor.

  To keep the factor small, choose `dt` and `dx` so that `v·dt/dx` of the
  lowest layer is close to 1.
- **Peak GPU memory:** ≈ 9·B·L + α·3·B·s, with B = 8·N² bytes per frame and s
  the number of captured ticks per graph call.
  - The first term is the resident state, about nine frame-equivalents per
    layer.
  - α = 2/3 for one device-capture call (1 when calls are pipelined), and
    α = 0 for host streaming and chunked offload.
- **Construction** (CPU covariance, Cholesky, SVD and stability check of the
  extrusion operators): about 7 s at 256², ≈ 25 s at 1024² and a few minutes at
  2048² for 10 layers on a desktop CPU. It happens once per simulator; `reseed`
  does not repeat it, and `opal run --model` skips it.

---

## 16. Examples

The examples are Jupyter notebooks, saved with their outputs (console logs,
plots and animations), so they can be read on GitHub without running anything.

| Notebook | Shows |
|---|---|
| [`01_quickstart`](examples/01_quickstart.ipynb) | Load a config, describe the layers, run, look at the frames, fiber coupling over the run |
| [`02_run_modes`](examples/02_run_modes.ipynb) | Memory-guard dry run; the four modes give bit-identical frames; speed and GPU memory per mode; lazy reading |
| [`03_checkpoint_resume`](examples/03_checkpoint_resume.ipynb) | 4 × 30 ticks == 1 × 120 ticks through a checkpoint; resuming in a fresh simulator |
| [`04_leo_pass`](examples/04_leo_pass.ipynb) | Reduced LEO pass: plan, exact r₀ switching, measured vs target r₀, the captured wavefronts as a GIF |
| [`05_ml_dataset`](examples/05_ml_dataset.ipynb) | r₀ × wind dataset, its labels against theory, the three loaders; optional CNN that estimates r₀ from phase |
| [`06_fiber_coupling`](examples/06_fiber_coupling.ipynb) | Independent-realization fiber coupling vs von Kármán theory, with and without tip/tilt correction |
| [`07_cli_tour`](examples/07_cli_tour.ipynb) | Every `opal` command: `info`, `config`, `plan`, `run`, `analyze`, `view`, `pass`, `dataset`, `export` |
| [`08_turbulence_schedules`](examples/08_turbulence_schedules.ipynb) | An r₀ ramp with `--schedule`: the b values, measured r₀ vs target and the refresh lag, the four schedule kinds |

To run them yourself, install Jupyter into the OPAL environment and open the
notebooks in JupyterLab or VS Code with that environment as the kernel:

```bash
pip install -e ".[examples]"         # JupyterLab + matplotlib
jupyter lab examples/
```

Each notebook starts by moving to the repository root, so outputs go to `runs/`
(ignored by git) wherever it is opened from. To re-execute one headless and
store the new outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/01_quickstart.ipynb
```

The stored outputs come from an RTX 4080 (16 GiB). Timings and the mode the
memory guard picks differ on other GPUs; the statistics do not. Animations are
written to `examples/media/` and shown from a markdown cell, because GitHub does
not display GIF cell outputs.

---

## 17. Citation

If you use OPAL in your research, please cite:

```bibtex
@inproceedings{akhlaghi2026opal,
  title     = {{OPAL}: a {GPU}-accelerated multilayer atmospheric turbulence simulator
               for space-to-ground optical links},
  author    = {Akhlaghi, Hamed and Taylor, Michael and Hranilovic, Steve},
  booktitle = {International Conference on Space Optics (ICSO)},
  year      = {2026}
}
```

See also [`CITATION.cff`](CITATION.cff).

## 18. License

MIT; see [`LICENSE`](LICENSE).

Developed at the Department of Electrical and Computer Engineering, McMaster
University.
