# Changelog

All notable changes to OPAL are listed here. Versions follow [semantic versioning](https://semver.org).

## 1.1.0

This release follows a round of hands-on testing on an 8 GiB laptop GPU. It focuses on memory safety, clearer
timing, default output folders, a real viewer and a much more complete README.

### Breaking changes

- Config keys renamed:
  - `satellite_orbit` → `satellite_orbit_altitude`;
  - `satellite_zenith` → `satellite_zenith_angle`.

  Old configs now stop with a message naming the new key. Saved runs and datasets made by 1.0 still load, and
  `opal dataset` resumes them.
- Unknown keys in a known config section are now an error, with a "did you mean" hint. Before, they were silently
  ignored.
- `load_frames()` on a chunked run returns a lazy `FrameSeries` instead of concatenating every chunk in RAM. Call
  `np.asarray(...)` on a slice if you need an array.

### Fixed

- **`opal analyze` could exhaust RAM and freeze the machine on large runs.** It loaded every selected frame at once
  and made two full-size complex128 copies (about 32 GiB for 1000 frames of 1024²). Frames are now read lazily
  through a memmap in batches (`--batch-mb`, default 256 MiB), so the working set no longer depends on the run
  length. The coupling values are unchanged (relative difference < 2e-15).
- Console timing missed most of the wall time. The clock started inside the run, so Python/TensorFlow import, CUDA
  initialisation, the simulator build (CPU covariance and SVD per layer), the final flush and teardown were not
  counted. Every phase is now reported, and a `total wall time` line closes the log. The manifest gains a `timing`
  block.
- `opal run` and `opal pass` without `-o` saved nothing. They now write to a dated folder by default (see Added).
- Chunked mode without an output directory silently discarded every chunk. The guard now says so.
- Turbulence schedules:
  - a 1-D schedule is one constant row of per-layer values;
  - the new `path_r0` kind takes the total path r0 per tick;
  - the schedule used is saved as `schedule.npy`;
  - `opal analyze r0` compares each frame with its own scheduled target.
- `--capture` was ignored in chunked mode. Captures now work in every mode and are saved to `captures.npz`.
  Capture ticks outside the run are reported instead of silently dropped.
- Large `L0`/pixel ratios produced NaN screens without warning, because the float32 covariance is ill-conditioned.
  OPAL now checks the extrusion operator when it builds a layer and stops with a clear message. The limit is
  L0/pixel ≤ 2500 (the default 25 m at 1 cm pixels is exactly 2500).
- Dataset resume:
  - it now checks that the stored settings match;
  - it reports an incomplete or corrupt dataset clearly;
  - it accepts datasets written by 1.0.

### Added

- **Default output folders** when `-o` is omitted:
  - `runs/<date>_<time>_N<grid>_L<layers>_r0-<r0>_T<steps>_seed<seed>` for `opal run`;
  - `runs/<date>_<time>_pass_...` for `opal pass`;
  - `datasets/...` for `opal dataset`;
  - `exports/...` for `opal export`.

  `--no-save` keeps a run in memory only.
- **`opal analyze` writes results** to `RUN/analysis/`:
  - `r0.csv` / `fiber.csv`, one row per frame;
  - `*_summary.json`;
  - with `--plot`, a time-series and histogram PNG.

  It also accepts `--frames N|all`, `--start/--stop/--stride` and `-o`/`--no-save`, and works on runs, chunked
  runs, passes and `final_frame.npy`.
- **`opal view` rewrite.** It opens runs (single file, chunked or final frame), passes, datasets and bare `.npy`
  files, reading one frame at a time.
  - Controls: play/pause, slider, step, ±10, home/end, go-to-tick, tip/tilt removal, log intensity and snapshots.
  - Analysis results appear as a time-series strip under the frames.
  - `--save` writes `.gif`/`.mp4` animations or a single-frame `.png` without a display.
- `opal plan` shows the per-layer extrusion table, and the build log shows the effective wind multiplier.
- `opal dataset --no-resume`. Datasets record their status (`incomplete`/`complete`).
- Python API: `open_frames`, `FrameSeries`, `analyze_run`, `load_analysis`. Also `python -m opal`.
- The build log reports progress per layer.

### Changed

- The simulator build is about 2× faster with bit-identical operators and screens (53 s → 25 s for 1024², 10 layers
  on an RTX 4080 workstation). The r0 switch at each `opal pass` segment is about 3× faster for the same reason
  (1024²: 53.9 s → 15.5 s per segment; 512²: ≈ 14 s → 5 s). The float64 host operators are released once copied to
  the GPU.
- The README is rewritten with numbered sections. It adds units for every parameter, a full `pass_params` reference
  with an example, and a clearer explanation of turbulence schedules (what `b` is and why `b = (r0_base/r0)^(5/6)`).
  New sections cover analysis outputs, the viewer, the SavedModel, architecture, modifying the engine and
  low-RAM machines.
- The examples are now Jupyter notebooks committed with their results (logs, plots and short GIFs), so they can be
  read on GitHub without running anything. They replace the `examples/*.py` scripts. Two notebooks are new:
  `07_cli_tour` (every `opal` command) and `08_turbulence_schedules`. The GIFs are stored in `examples/media/`
  because GitHub does not render GIF cell outputs. A new `examples` extra (`pip install -e ".[examples]"`) installs
  JupyterLab and Matplotlib. The CNN in `05_ml_dataset` is enabled with `TRAIN = True` instead of `--train`.

## 1.0.0

First public release.
