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
- **Dynamic turbulence.** Per-tick turbulence schedules (b-scale, C<sub>n</sub>²
  or r₀), plus *exact* per-layer r₀ switching for full satellite passes.
  Operators are rebuilt for the new statistics while the screens persist.
- **Four run modes and a memory guard.** Keep frames on the GPU, stream them to
  host RAM or a memory-mapped file, write bounded chunks to disk, or keep only
  the final field. `auto` picks a mode that fits and falls back on
  out-of-memory errors.
- **Bit-exact segmentation.** The full simulator state fits in one small
  checkpoint. Splitting a run into pieces reproduces the uninterrupted run
  exactly.
- **ML dataset generation.** Seeded parameter sweeps × independent realizations
  produce labelled, time-correlated irradiance/phase sequences, with fiber
  coupling, scintillation, angle of arrival and measured r₀ labels.
- **Analysis tools.**
  - r₀ estimation from the mutual coherence function.
  - Single-mode-fiber coupling vs von Kármán theory.
  - Tip/tilt removal, scintillation, pass geometry.
- **Portable models.** Export a configured simulator as a TensorFlow SavedModel
  and run it without the Python source.

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Configuration](#configuration)
4. [Run modes and the memory guard](#run-modes-and-the-memory-guard)
5. [Dynamic turbulence and satellite passes](#dynamic-turbulence-and-satellite-passes)
6. [Checkpoints and bit-exact segmentation](#checkpoints-and-bit-exact-segmentation)
7. [ML dataset generation](#ml-dataset-generation)
8. [Analysis](#analysis)
9. [SavedModel export](#savedmodel-export)
10. [Python API](#python-api)
11. [Output formats](#output-formats)
12. [Performance model](#performance-model)
13. [Examples](#examples)
14. [Citation](#citation) · [License](#license)

---

## Installation

OPAL needs a CUDA-capable NVIDIA GPU and TensorFlow ≥ 2.10. It was developed
and benchmarked on an RTX 4080 (16 GiB).

### Windows (native GPU)

TensorFlow 2.10 is the last release with native Windows CUDA support. The conda
environment pins it, together with CUDA 11.2 and cuDNN 8.1:

```bash
git clone https://github.com/Markeloen/OPAL.git
cd OPAL
conda env create -f environment.yml      # creates "opal" and installs OPAL in editable mode
conda activate opal
```

### Linux / WSL2

```bash
git clone https://github.com/Markeloen/OPAL.git
cd OPAL
python -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]"
pip install -e ".[viewer]"
```

### Verify

```bash
opal info                       # OPAL/TensorFlow versions, CUDA build, GPU, host RAM
opal run -c configs/quickstart_256.json --steps 50
```

TensorFlow is an optional extra (`pip install -e ".[tf]"`) rather than a hard
dependency, so pip never replaces a working GPU build with a CPU-only wheel.
`matplotlib` is needed only for `opal view`, and `ffmpeg` only for `.mp4` output.

---

## Quick start

### Command line

```bash
# 200 ticks of the 256x256, 10-layer quick-start channel
opal run -c configs/quickstart_256.json -o runs/qs

# Look at it
opal view runs/qs --save qs.gif
opal analyze r0 runs/qs

# Change anything from the command line
opal run -c configs/quickstart_256.json --grid 512 --r0 0.08 --set wind=12 --steps 1000 -o runs/strong
```

Every run directory holds the frames and a `manifest.json`. The manifest records
the resolved config, seed, mode, timing, GPU memory and OPAL version, so any
run can be reproduced.

### Python

```python
import opal

config = opal.load_config("configs/quickstart_256.json", overrides={"steps": 500})
sim = opal.GpuSimulator(config)            # builds screens + operators on the GPU

result = opal.run(sim, mode="auto", out_dir="runs/qs")
frames = result.frames                     # complex64 [500, 256, 256] (memory-mapped)
print(result.mode, f"{result.steps_per_second:.1f} ticks/s")

sim.reseed(7)                              # new independent realization, no rebuild
```

---

## Configuration

A config is a JSON file with two sections, `simulation_params` and
`channel_params`, plus an optional `pass_params`. Unspecified keys take their
defaults. `opal config -c my.json` prints the fully resolved config, and
`-o resolved.json` saves it.

```json
{
  "simulation_params": { "num_steps": 1000, "time_per_tick": 0.001, "seed": 1 },
  "channel_params": {
    "phase_screen_size": 1024, "pixel_size": 0.01, "num_layers": 10,
    "total_r0": 0.2, "wavefront_wavelength": 1.55e-6, "L0": 25.0,
    "ground_wind_speed": 5.0, "satellite_orbit": 600000.0, "satellite_zenith": 0.0
  }
}
```

| Key | Short alias | Default | Meaning |
|---|---|---|---|
| `simulation_params.num_steps` | `steps` | 1000 | Ticks to simulate |
| `simulation_params.time_per_tick` | `dt` | 1e-3 | Tick duration (s) |
| `simulation_params.seed` | `seed` | `null` | Master seed. Per-layer seeds are derived from it; `null` = random, recorded in the manifest |
| `channel_params.phase_screen_size` | `N`, `grid` | 512 | Grid size *N* (pixels, even) |
| `channel_params.pixel_size` | `dx` | 0.01 | Grid spacing (m) |
| `channel_params.num_layers` | `layers` | 10 | Number of phase-screen layers |
| `channel_params.total_r0` | `r0` | 0.2 | Total path Fried parameter (m) at the wavelength |
| `channel_params.wavefront_wavelength` | `wavelength`, `wvl` | 1.55e-6 | Wavelength (m) |
| `channel_params.L0` | `L0` | 25 | Von Kármán outer scale (m) |
| `channel_params.ground_wind_speed` | `wind` | 5 | Ground wind speed in the Bufton model (m/s) |
| `channel_params.satellite_orbit` | `orbit` | 6e5 | Circular-orbit altitude (m); sets the slew-induced wind |
| `channel_params.satellite_zenith` | `zenith` | 0 | Zenith angle (deg) where the slew rate is evaluated |

Aliases work everywhere overrides are accepted:
- `--set KEY=VALUE` on the CLI;
- `load_config(path, overrides={...})`;
- dataset sweeps.

`--grid`, `--layers`, `--r0`, `--seed` and `--steps` are shortcuts.

**Ready-made configs** (`configs/`):

| File | Case |
|---|---|
| `quickstart_256.json` | 256², 10 layers, r₀ = 10 cm: seconds to build, for trying things out |
| `default_1024.json` | 1024², 10 layers: the paper's reference resolution |
| `leo_pass_b2.json` | 700 km LEO overhead pass, 20°–90°–20° elevation, 30 exact r₀ segments (`opal pass`) |
| `fiber_100us_b3.json` | 1024², 100 µs ticks, fiber-coupling case (0.8 m aperture) |

---

## Run modes and the memory guard

One complex64 frame is 8·N² bytes: 8 MiB at 1024², 32 MiB at 2048². A long,
densely captured run therefore outgrows GPU memory quickly. OPAL offers four
ways to run the *same* channel. For a given seed they produce **bit-identical
frames**.

| Mode | Where frames go | GPU memory | Use it for |
|---|---|---|---|
| `device` | Accumulate on the GPU, copied out at the end | grows ≈ 2 frames per tick | Short bursts that fit on the GPU; frames stay on the device for further in-graph processing |
| `host` | Streamed per tick to host RAM, or to a memory-mapped `frames.npy` with `-o` | **flat** | Long, dense captures; the default for most saved runs |
| `chunked` | Written to disk in bounded `frames_chunk_*.npy` files | bounded by the chunk size | Beyond RAM; exported SavedModels; resumable disk datasets |
| `no_save` | Only the final field is kept | minimal | Benchmarks, warm-up, in-graph statistics, Monte-Carlo draws |
| `auto` | The guard picks `device` → `host` → `chunked` → `no_save` | | Default |

The guard measures the simulator's real static footprint with one step, and
predicts each mode's peak:
- `device`: baseline + 2 × output + margin;
- `host`: baseline + 4 frames + margin;
- `chunked`: baseline + 2 × chunk + margin.

The margin is max(1.5 GiB, 20 % of the GPU). The guard also checks host RAM:
the output must fit in 60 % of the available RAM, unless it is written to a
memory-mapped file. It then picks the first safe mode. If a mode still hits an
out-of-memory error, the run falls back along
`device → host → chunked (halving the chunk) → no_save` and records it in the
manifest.

```bash
opal plan -c configs/default_1024.json --steps 20000      # dry run: which mode, and why
opal run  -c configs/default_1024.json --steps 20000 --mode host -o runs/long
opal run  -c configs/default_1024.json --steps 20000 --mode chunked --chunk-size 100 -o runs/chunks
```

**Measured, RTX 4080, 10 layers, 512², seed 20260716** (peak GPU memory, including
the ≈ 150 MiB resident simulator state):

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
against 24 MiB per tick for device capture. That is 2.4× faster than the five
memory-forced segments device capture needs.

---

## Dynamic turbulence and satellite passes

### Turbulence schedules

`--schedule` / `schedule=` accepts a `[T, L]` array: one row per tick, one column
per layer, and a single row is held constant. It can be an `.npy`, `.npz`,
`.csv` or `.json` file, or an array in Python. `--schedule-kind` selects how the
values are read:

| Kind | Value | Effect |
|---|---|---|
| `b_scale` | innovation multiplier *b* | C<sub>n</sub>² scales by *b*² |
| `cn2` | C<sub>n</sub>² multiplier *m* | *b* = √*m* |
| `r0` | absolute per-layer r₀ (m) | *b* = (r₀,base / r₀)<sup>5/6</sup> |

Schedules act inside the compiled graph at no extra cost. They scale the
innovations of the extrusion, which is exact for slowly varying strength.

### Exact satellite passes (`opal pass`)

For a full pass the r₀ of every layer changes by an order of magnitude from the
horizon to zenith. `opal pass`:

1. builds the circular-orbit pass geometry (elevation, range, slew rate);
2. derives each layer's slant-path r₀ along it;
3. splits the pass into segments.

At each segment boundary every layer's covariance-based extrusion operators are
**rebuilt for the new r₀**, while the current screens are kept. The field
evolves continuously, and its statistics follow the pass. The whole pass, or a
`--window` in seconds around zenith, is mapped onto the run's ticks. The
captured wavefronts are streamed to a memory-mapped file, and their r₀ is
measured against the target.

Two `pass_params` control the screens themselves:

- `match_initial_r0` (default `true`) rescales the starting screens to the first
  segment's r₀ before the pass begins. A von Kármán screen drawn from the same
  random numbers scales exactly as r₀^(−5/6), so the first captures already have
  the right statistics instead of relaxing from the configured r₀. Set it to
  `false` to reproduce the original (paper) behaviour bit for bit.
- `regenerate_on_switch` (default `false`) draws a fresh screen at every switch
  instead of keeping the current one. The switch is then instantaneous but the
  field is discontinuous. Screens are seeded from `(seed, layer, segment)`, so
  runs stay reproducible.

With kept screens, new statistics enter as rows are extruded: a layer needs about
`N / rows_per_tick` ticks to refresh fully (`opal plan` and `describe()` print
rows per tick). Use segments long enough for the slow, low layers, e.g. ≥ 100
ticks per segment at 256².

```bash
opal pass -c configs/leo_pass_b2.json -o runs/b2                     # 30 segments, 200 captures
opal pass -c configs/leo_pass_b2.json -o runs/b2_zen --window 120 --segments 10 --steps 1000
```

`runs/b2/r0_tracking.csv` lists target vs measured r₀ per capture. In the paper's
B2 case (512², 3000 ticks, 30 segments) the median measured/predicted ratio is
0.99. The per-segment cost is one CPU-side covariance build + SVD per layer; see
the [performance model](#performance-model).

---

## Checkpoints and bit-exact segmentation

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

---

## ML dataset generation

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
  Cartesian product. Any config alias works, plus `elevation` (degrees, maps to
  the zenith angle).
- **Targets.** `field` (complex), `intensity`, `phase`.
- **Per-frame labels**, over a `--label-aperture` (default 40 % of the grid):
  - `eta`: single-mode fiber coupling efficiency;
  - `scint`: scintillation index;
  - `tilt`: angle of arrival in rad, from the far-field centroid.
- **Per-sample labels:**
  - the sweep and channel parameters;
  - D/r₀;
  - `r0_measured` (median r₀ measured over `--r0-frames` frames);
  - mean/std of eta, mean scintillation, rms tilt.
- **Reproducible.** Sample *k* uses the seed `SeedSequence([base_seed, point,
  realization])`, recorded in `labels.csv` and `manifest.json`. Re-running the
  command **resumes**, skipping finished samples; `--no-resume` regenerates them.
- **Efficient.** One simulator is built per sweep point, and realizations reuse
  it via `reseed` (no graph rebuild). Frames are streamed from the GPU and never
  accumulate on it.

Layout and loading:

```
datasets/r0_wind/
  manifest.json            base seed, seed rule, sweep, config, targets, timing
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

`examples/05_ml_dataset.py --train` generates a small r₀ × wind dataset and fits
a small CNN that reads r₀ off single wrapped-phase frames (fed as cos/sin). On
held-out realizations it reaches ≈ 0.5 cm mean absolute error, against ≈ 2.7 cm
for always predicting the mean.

---

## Analysis

```bash
opal analyze r0 runs/qs                      # MCF-based r0 over evenly spaced frames
opal analyze fiber runs/qs --aperture 0.3    # SMF coupling efficiency vs von Karman theory
```

`opal analyze` works on any run directory: device, host, chunked or pass. The
frames of a single run are correlated in time. For ensemble statistics, use
independent realizations: `opal dataset -k K`, or `GpuSimulator.reseed` in a
loop (see `examples/06_fiber_coupling.py`).

`opal.analysis` provides:

| Function | Purpose |
|---|---|
| `estimate_r0_from_wavefront(field, dx, aperture=None, L0=...)` | r₀ from the 1/e point of the measured mutual coherence function, von Kármán corrected |
| `compute_fiber_coupling(fields, dx, a=1.12, aperture_diameter=...)` | Single-mode-fiber coupling efficiency per frame (Dikmelik & Davidson) |
| `coupling_efficiency_theoretical_vk(D, r0, L0, a=1.12)` | Theoretical mean coupling under a von Kármán MCF |
| `remove_tilt_com(fields, dx, aperture_diameter=...)` | Centroid tip/tilt removal (ideal fast-steering mirror) |
| `scintillation_index`, `tilt_centroid`, `aperture_mask` | Irradiance and angle-of-arrival statistics |
| `circular_orbit_pass`, `per_layer_r0_along_pass`, `effective_path_r0` | Pass geometry and slant-path turbulence |

Validation from the paper:
- the measured r₀ tracks the prediction to a median ratio of 0.99 over a full
  pass;
- Monte-Carlo fiber coupling agrees with von Kármán theory to within ±7 %.

---

## SavedModel export

```bash
opal export -c configs/default_1024.json -o exports/opal_1024
opal run --model exports/opal_1024 --steps 500 -o runs/from_model
```

```python
model = opal.load("exports/opal_1024")    # LoadedSimulator: same call API, no OPAL source needed
opal.run(model, 500, mode="chunked", out_dir="runs/from_model")
```

The export contains the screens, the extrusion operators and the propagator.
Its signatures are:
- `simulate_steps`, `simulate_steps_no_save`, `simulate_steps_capture`;
- `simulate_steps_sparse_capture`;
- `get_state`, `set_state`, `reset_to_base`;
- `set_b_scale_profile`.

It can be served, or embedded in another TensorFlow graph, for example as an
in-the-loop channel layer. Exported models run in `device`, `chunked` or
`no_save` mode (host streaming needs the live Python object).

---

## Python API

| Object | Description |
|---|---|
| `opal.load_config(path_or_dict, overrides=None)` / `opal.save_config` | Load, fill defaults, apply aliases, validate |
| `opal.GpuSimulator(config, seed=None, log_dir=None)` | The simulator (`tf.Module`) |
| `opal.run(sim, num_steps=None, *, mode, out_dir, chunk_size, schedule, schedule_kind, capture_indices, state_in, state_out, log)` | Run with the memory guard; returns a `RunResult` (`frames`, `final_frame`, `mode`, `run_time_s`, `steps_per_second`, `manifest`) |
| `opal.load_frames(run_dir)` / `opal.iter_frames(run_dir, stride)` | Read any run directory (memory-mapped / streaming) |
| `opal.save_state` / `opal.load_state` | Checkpoints |
| `opal.build_schedule(schedule, T, L, base_r0, kind)` | Normalize a schedule to per-tick b-scales |
| `opal.plan_pass(config)` / `opal.run_pass(config, out_dir)` | Satellite pass with exact r₀ switching |
| `opal.generate_dataset(...)`, `opal.load_dataset`, `opal.iter_samples`, `opal.as_tf_dataset` | ML datasets |
| `opal.export(sim, dir)` / `opal.load(dir)` | SavedModel |

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
| `describe()` | per-layer height, r₀, wind, C<sub>n</sub>², rows/tick |

The `schedule` argument is a `[n, L]` float32 b-scale tensor; use
`tf.ones([1, L])` for constant turbulence.

---

## Output formats

```
runs/<name>/
  manifest.json               config, seed, OPAL version, modes (requested/selected/final),
                              guard reason, timings, GPU memory, memory estimate
  frames.npy                  complex64 [T, N, N]           device / host
  frames_chunk_000.npy ...    complex64 [chunk, N, N]       chunked
  frames_chunk_manifest.json  per-chunk step ranges and compute times
  final_frame.npy             complex64 [N, N]              no_save
  captures.npz                layer screens / r0 / b-scales at --capture ticks

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
grid edges. Irradiance is `abs(U)**2`, and phase is `np.angle(U)`.

---

## Performance model

On the RTX 4080, with 10 layers at 1024²:

- **Wall time:** t ≈ 0.123·T + 53.9·N<sub>seg</sub> s.
  - The first term is GPU propagation, per tick, through all planes, including
    screen extrusion.
  - The second is the per-segment operator rebuild of `opal pass`: a CPU
    covariance build + SVD per layer. At 512² this is ≈ 14 s per segment.
  - Per-tick time is nearly flat from 128² to 2048², at 0.12–0.28 s. Below
    ~2048² the GPU is launch-bound rather than FLOP-bound, so larger grids are
    almost free.
- **Peak GPU memory:** ≈ 9·B·L + α·3·B·s, with B = 8·N² bytes per frame and s
  the number of captured ticks per graph call.
  - The first term is the resident state, about nine frame-equivalents per
    layer.
  - α = 2/3 for one device-capture call (1 when calls are pipelined), and
    α = 0 for host streaming and chunked offload.
- **Construction** (CPU covariance/SVD of the extrusion operators): about 15 s
  at 256², 30 s at 1024² and 2 min at 2048² for 10 layers. It happens once per
  simulator; `reseed` does not repeat it.

---

## Examples

| Script | Shows |
|---|---|
| `examples/01_quickstart.py` | Build, describe the layers, run, basic statistics |
| `examples/02_run_modes.py` | Guard decisions; all four modes give identical frames; memory/throughput per mode |
| `examples/03_checkpoint_resume.py` | 4 × 30 ticks == 1 × 120 ticks through a checkpoint file |
| `examples/04_leo_pass_replacement.py` | Reduced LEO pass with exact r₀ switching and r₀ tracking |
| `examples/05_ml_dataset.py [--train]` | r₀ × wind dataset; optional CNN that estimates r₀ from phase |
| `examples/06_fiber_coupling.py` | Independent-realization fiber coupling vs von Kármán theory, with and without tip/tilt correction |

Run them from the repository root. Outputs go to `runs/`, which is ignored by
git.

---

## Citation

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

## License

MIT; see [`LICENSE`](LICENSE).

Developed at the Department of Electrical and Computer Engineering, McMaster
University.
