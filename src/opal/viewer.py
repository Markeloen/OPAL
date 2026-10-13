"""Interactive viewer and movie export for OPAL outputs (requires matplotlib).

::

    opal view runs/<run>                          # run, chunked run or pass directory
    opal view datasets/<ds> --sample 3            # dataset (n / p switch samples)
    opal view runs/<run> --views all              # intensity, log, phase, tilt-removed phase
    opal view runs/<run> --save movie.gif         # render instead of opening a window
    opal view runs/<run> --tick 500 --save f.png  # one frame

Frames are read from disk one at a time (see :mod:`opal.frames`), so runs of
any size open instantly and use little RAM.

Window controls
---------------
==================  ==================================================
space / Play        play / pause
left / right        previous / next frame
down / up           10 frames back / forward
home / end          first / last frame
slider              scrub through the selected frames
"go to tick" box    jump to a simulation tick (type it, press Enter)
t                   toggle tip/tilt removal on the phase panels
l                   toggle linear / log irradiance
s                   snapshot: PNG of the window + ``.npy`` of the frame,
                    written to ``<source>/snapshots/``
n / p               next / previous sample (datasets)
==================  ==================================================

A time-series strip under the images shows, when available, the results of
``opal analyze`` (``analysis/fiber.csv``, ``analysis/r0.csv``), the measured
and target r0 of a pass (``r0_tracking.csv``) or the per-frame labels of a
dataset sample (eta, scintillation, tilt), with a cursor at the current frame.
"""

from __future__ import annotations

import csv
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

from .frames import FrameSeries, is_dataset_dir, open_frames, read_manifest, select_indices

__all__ = ["VIEWS", "Source", "open_source", "view", "field_images", "animate_frames", "animate_run"]

VIEWS = {
    "intensity": "irradiance / mean",
    "log_intensity": "log10(irradiance / mean)",
    "phase": "phase (rad)",
    "phase_tt": "phase, tip/tilt removed (rad)",
}
_VIEWS_FOR = {
    "field": ("intensity", "log_intensity", "phase", "phase_tt"),
    "intensity": ("intensity", "log_intensity"),
    "phase": ("phase", "phase_tt"),
}
_NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
_KEYMAPS = ("keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.yscale",
            "keymap.xscale", "keymap.pan", "keymap.quit_all")
_OUR_KEYS = {" ", "left", "right", "up", "down", "pageup", "pagedown", "home", "end", "t", "l", "s", "n", "p"}
_PHASE_MASK_FRACTION = 1e-4  # phase hidden where irradiance < this x mean


def field_images(frame):
    """Normalised irradiance and wrapped phase of a complex field."""
    frame = np.asarray(frame)
    intensity = np.abs(frame) ** 2
    return intensity / max(float(intensity.mean()), 1e-30), np.angle(frame)


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
def _plain(section):
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in (section or {}).items()}


def _npz_member(path, key):
    """Memory-map array ``key`` of an uncompressed ``.npz``; load it if compressed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        stored = info.compress_type == zipfile.ZIP_STORED
    if stored:
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            fh.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            offset = fh.tell()
        if not dtype.hasobject:
            return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                             order="F" if fortran else "C")
    with np.load(path) as data:
        return data[key]


class _SampleFrames:
    """Lazy complex-field view of one dataset sample (``field`` or ``intensity`` + ``phase``)."""

    def __init__(self, path):
        with zipfile.ZipFile(path) as zf:
            names = {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
        self.members = {k: _npz_member(path, k) for k in ("field", "intensity", "phase") if k in names}
        if not self.members:
            raise ValueError(f"{path} holds no field, intensity or phase array")
        first = next(iter(self.members.values()))
        self.shape = tuple(first.shape)
        m = self.members
        self.content = "field" if ("field" in m or {"intensity", "phase"} <= set(m)) else next(iter(m))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        m = self.members
        if "field" in m:
            return np.asarray(m["field"][i])
        if "intensity" in m and "phase" in m:
            return (np.sqrt(np.maximum(m["intensity"][i], 0)) * np.exp(1j * m["phase"][i])).astype(np.complex64)
        if "intensity" in m:
            return np.sqrt(np.maximum(m["intensity"][i], 0)).astype(np.complex64)
        return np.exp(1j * np.asarray(m["phase"][i])).astype(np.complex64)


class Source:
    """What the viewer shows: lazily indexed frames plus metadata and time series.

    ``frames[i]`` returns one ``[n, n]`` complex field; ``steps[i]`` its tick.
    ``series`` is a list of ``(label, steps, values, reference_or_None)``.
    """

    def __init__(self, frames, *, steps=None, dx=None, wavelength=None, dt=None, title="OPAL",
                 content="field", series=(), snap_dir=None, info=None, dataset=None, sample=None,
                 num_samples=None):
        self.frames = frames
        n = len(frames)
        self.steps = np.arange(n, dtype=np.int64) if steps is None else np.asarray(steps, np.int64)
        self.dx, self.wavelength, self.dt = dx, wavelength, dt
        self.title, self.content = title, content
        self.series = list(series)
        self.snap_dir = Path(snap_dir) if snap_dir is not None else Path.cwd() / "snapshots"
        self.info = info or {}
        self.dataset, self.sample, self.num_samples = dataset, sample, num_samples

    def __len__(self):
        return len(self.frames)

    def frame(self, i):
        return np.asarray(self.frames[int(i)])


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _analysis_series(run_dir):
    from .analysis.runs import load_analysis

    out = []
    for what, label, ref_key in (("fiber", "fiber coupling η", "eta_theory"), ("r0", "measured r₀ (m)", "target_r0_m")):
        got = load_analysis(run_dir, what)
        if got is not None:
            steps, values, extra = got
            out.append((label + " [analysis]", steps, values, extra.get(ref_key)))
    return out


def _run_source(path):
    fs = open_frames(path)
    manifest = fs.manifest
    cfg = manifest.get("config") or {}
    ch, sim = _plain(cfg.get("channel_params")), _plain(cfg.get("simulation_params"))
    dx = float(ch["pixel_size"]) if "pixel_size" in ch else None
    wvl = float(ch["wavefront_wavelength"]) if "wavefront_wavelength" in ch else None
    dt = float(sim["time_per_tick"]) if "time_per_tick" in sim else None
    root = fs.source if fs.source is not None and fs.source.is_dir() else Path(path).parent
    parts = [f"OPAL {fs.kind}"]
    if "phase_screen_size" in ch:
        parts.append(f"N={int(ch['phase_screen_size'])}")
    if "total_r0" in ch:
        parts.append(f"r0={float(ch['total_r0']):g} m")
    series, info = [], {}
    if fs.kind == "pass" and (root / "r0_tracking.csv").exists():
        rows = _read_csv(root / "r0_tracking.csv")
        steps = np.array([int(r["step"]) for r in rows])
        meas = np.array([float(r["measured_r0"]) for r in rows])
        theo = np.array([float(r["theoretical_r0"]) for r in rows])
        if np.isfinite(meas).any():
            series.append(("pass r₀ (m): measured vs target", steps, meas, theo))
        else:
            series.append(("pass target r₀ (m)", steps, theo, None))
        info = {int(r["step"]): f"elev {float(r['elevation_deg']):.1f}° | target r₀ "
                                f"{float(r['theoretical_r0']):.3f} m" for r in rows}
    series += _analysis_series(root)
    return Source(fs, steps=fs.steps, dx=dx, wavelength=wvl, dt=dt, title=" ".join(parts),
                  series=series[:3], snap_dir=root / "snapshots", info=info)


def _dataset_source(ds_dir, sample_path, index):
    frames = _SampleFrames(sample_path)
    with np.load(sample_path) as data:
        steps = data["steps"] if "steps" in data.files else None
        labels = json.loads(str(data["labels_json"])) if "labels_json" in data.files else {}
        series = []
        if "eta" in data.files:
            series.append(("fiber coupling η", steps, np.asarray(data["eta"]), None))
        if "scint" in data.files:
            series.append(("scintillation index", steps, np.asarray(data["scint"]), None))
        if "tilt" in data.files:
            tilt = np.asarray(data["tilt"])
            series.append(("|tilt| (µrad)", steps, 1e6 * np.hypot(tilt[:, 0], tilt[:, 1]), None))
    manifest = read_manifest(ds_dir) if ds_dir is not None else {}
    ch, sim = _plain((manifest.get("config") or {}).get("channel_params")), \
        _plain((manifest.get("config") or {}).get("simulation_params"))
    dx = labels.get("dx", ch.get("pixel_size"))
    wvl = labels.get("wavelength", ch.get("wavefront_wavelength"))
    dt = labels.get("dt", sim.get("time_per_tick"))
    num = int(manifest["num_samples"]) if "num_samples" in manifest else None
    desc = " ".join(f"{k}={labels[k]:g}" for k in ("r0", "wind", "elevation") if isinstance(labels.get(k), (int, float)))
    title = f"OPAL dataset sample {index}" + (f"/{num - 1}" if num else "") + (f" | {desc}" if desc else "")
    if labels.get("seed") is not None:
        title += f" | seed {labels['seed']}"
    root = ds_dir if ds_dir is not None else Path(sample_path).parent
    return Source(frames, steps=steps, dx=float(dx) if dx else None, wavelength=float(wvl) if wvl else None,
                  dt=float(dt) if dt else None, title=title, content=frames.content, series=series,
                  snap_dir=Path(root) / "snapshots", dataset=ds_dir, sample=index, num_samples=num)


def open_source(path, sample=0):
    """Open a run / pass / dataset directory, a dataset sample ``.npz`` or a frames ``.npy``."""
    if isinstance(path, Source):
        return path
    if isinstance(path, (np.ndarray, FrameSeries)) or hasattr(path, "__array__") and not isinstance(path, (str, Path)):
        arr = path if isinstance(path, FrameSeries) else np.asarray(path)
        if getattr(arr, "ndim", 3) == 2:
            arr = arr[None]
        return Source(arr, steps=getattr(arr, "steps", None))
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".npz":
        ds_dir = path.parent.parent if path.parent.name == "samples" and is_dataset_dir(path.parent.parent) else None
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError:
            index = 0
        return _dataset_source(ds_dir, path, index)
    if is_dataset_dir(path):
        sample_path = path / "samples" / f"sample_{int(sample):05d}.npz"
        if not sample_path.exists():
            have = sorted((path / "samples").glob("sample_*.npz"))
            raise FileNotFoundError(f"{sample_path} does not exist ({len(have)} samples in {path})")
        return _dataset_source(path, sample_path, int(sample))
    return _run_source(path)


# ----------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------
def _resolve_views(views, content):
    available = _VIEWS_FOR[content]
    if views is None:
        wanted = [v for v in ("intensity", "phase") if v in available] or list(available[:1])
    elif views == "all" or views == ["all"]:
        wanted = list(available)
    else:
        wanted = [views] if isinstance(views, str) else list(views)
    bad = [v for v in wanted if v not in VIEWS]
    if bad:
        raise ValueError(f"unknown view(s) {bad}; choose from {list(VIEWS)} or 'all'")
    missing = [v for v in wanted if v not in available]
    if missing:
        raise ValueError(f"view(s) {missing} need data this source does not have ({content} only); "
                         f"available: {list(available)}")
    return wanted


class _Viewer:
    def __init__(self, src, views, positions, *, series=True, interactive=False, fps=20, start_pos=0,
                 selection=None, log=print):
        self.src, self.views, self.positions = src, list(views), np.asarray(positions)
        self.show_series, self.interactive, self.fps = series, interactive, fps
        self.selection = selection or {}
        self.log = log or (lambda _m: None)
        self.pos, self.playing, self._syncing = 0, False, False
        self._build()
        self.show(start_pos)

    # -- layout -----------------------------------------------------------
    def _build(self):
        nv = len(self.views)
        series = self.src.series if self.show_series else []
        ns = len(series)
        width = max(7.5, 4.3 * nv + 0.6)
        height = 4.3 + (1.9 if ns else 0) + (0.7 if self.interactive else 0.2)
        if self.interactive:
            import matplotlib.pyplot as plt

            self.fig = plt.figure(figsize=(width, height))
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=(width, height))
            FigureCanvasAgg(self.fig)
        bottom = (0.75 if self.interactive else 0.25) / height
        ratios = [4.3, 1.9] if ns else [4.3]
        gs = self.fig.add_gridspec(len(ratios), 1, height_ratios=ratios, left=0.06, right=0.97,
                                   bottom=bottom + (0.45 / height if ns else 0.1 / height),
                                   top=1 - 0.55 / height, hspace=0.35)
        top = gs[0].subgridspec(1, nv, wspace=0.35)
        self.axes, self.images = [], []
        first = self.src.frame(self.positions[0]) if len(self.positions) else None
        n = first.shape[-1]
        extent = None
        if self.src.dx:
            half = n * self.src.dx / 2.0
            extent = (-half, half, -half, half)
        self._aperture = 0.4 * n * (self.src.dx or 1.0)
        I0 = np.abs(first) ** 2
        self._imax = float(np.percentile(I0 / max(float(I0.mean()), 1e-30), 99.5)) or 1.0
        for k, view in enumerate(self.views):
            ax = self.fig.add_subplot(top[0, k])
            im = ax.imshow(np.zeros((n, n), np.float32), origin="lower", extent=extent,
                           interpolation="antialiased")
            self._style(ax, im, view)
            ax.set_xlabel("x (m)" if self.src.dx else "x (px)", fontsize=8)
            ax.tick_params(labelsize=7)
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
            self.axes.append(ax)
            self.images.append(im)
        self.series_axes, self.cursors = [], []
        if ns:
            sub = gs[1].subgridspec(1, ns, wspace=0.3)
            for k in range(ns):
                self.series_axes.append(self.fig.add_subplot(sub[0, k]))
            self._plot_series()
        self.suptitle = self.fig.suptitle("", fontsize=10)
        if self.interactive:
            self._controls(height)

    def _style(self, ax, im, view):
        if view == "intensity":
            im.set_cmap("inferno")
            im.set_clim(0, self._imax)
        elif view == "log_intensity":
            im.set_cmap("inferno")
            im.set_clim(-2.0, 1.0)
        else:
            import matplotlib

            cmap = matplotlib.colormaps["twilight"].copy()
            cmap.set_bad("0.55")
            im.set_cmap(cmap)
            im.set_clim(-np.pi, np.pi)
        ax.set_title(VIEWS[view] + (f"\n(aperture {self._aperture:.2f} m)" if view == "phase_tt" else ""),
                     fontsize=9)

    def _plot_series(self):
        self.cursors = []
        for ax, (label, steps, values, ref) in zip(self.series_axes, self.src.series):
            ax.clear()
            steps = np.asarray(steps if steps is not None else np.arange(len(values)))
            ax.plot(steps, values, ".-", lw=0.8, ms=2.5, color="tab:blue")
            if ref is not None:
                ax.plot(steps, np.broadcast_to(ref, np.shape(values)), "--", lw=1.0, color="tab:red")
            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("tick", fontsize=7)
            ax.grid(alpha=0.3)
            self.cursors.append(ax.axvline(self.src.steps[self.positions[0]] if len(self.positions) else 0,
                                           color="k", lw=1.0, alpha=0.7))

    def _controls(self, height):
        from matplotlib.widgets import Button, Slider, TextBox

        h = 0.32 / height
        y = 0.2 / height
        self.widgets = {}
        if len(self.positions) > 1:
            ax_prev = self.fig.add_axes([0.02, y, 0.05, h])
            ax_play = self.fig.add_axes([0.075, y, 0.06, h])
            ax_next = self.fig.add_axes([0.14, y, 0.05, h])
            ax_slider = self.fig.add_axes([0.28, y + h * 0.15, 0.42, h * 0.7])
            self.widgets["prev"] = Button(ax_prev, "◀")
            self.widgets["play"] = Button(ax_play, "Play")
            self.widgets["next"] = Button(ax_next, "▶")
            self.widgets["prev"].on_clicked(lambda _e: self.show(self.pos - 1))
            self.widgets["next"].on_clicked(lambda _e: self.show(self.pos + 1))
            self.widgets["play"].on_clicked(lambda _e: self.toggle_play())
            self.slider = Slider(ax_slider, "frame", 0, len(self.positions) - 1, valinit=0, valstep=1)
            self.slider.valtext.set_visible(False)
            self.slider.on_changed(self._on_slider)
        else:
            self.slider = None
        ax_box = self.fig.add_axes([0.86, y, 0.1, h])
        self.textbox = TextBox(ax_box, "go to tick ", initial="")
        self.textbox.on_submit(self._on_tick)
        self.timer = self.fig.canvas.new_timer(interval=max(1, int(1000 / max(1, self.fps))))
        self.timer.add_callback(self._advance)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -- rendering --------------------------------------------------------
    def _render(self, view, field):
        if view in ("intensity", "log_intensity"):
            inten = np.abs(field) ** 2
            inten = inten / max(float(inten.mean()), 1e-30)
            return np.log10(np.maximum(inten, 1e-6)) if view == "log_intensity" else inten
        # Phase is undefined where there is no light (outside the beam, deep
        # fades); those pixels are left blank instead of showing wrap noise.
        inten = np.abs(field) ** 2
        dark = inten < _PHASE_MASK_FRACTION * max(float(inten.mean()), 1e-30)
        if view == "phase_tt":
            from .analysis.fiber import remove_tilt_com

            corrected, _ = remove_tilt_com(field, self.src.dx or 1.0, aperture_diameter=self._aperture)
            n = field.shape[-1]
            yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
            outside = np.hypot(xx, yy) * (self.src.dx or 1.0) > self._aperture / 2.0
            return np.ma.masked_array(np.angle(corrected[0]), mask=dark | outside)
        return np.ma.masked_array(np.angle(field), mask=dark)

    def show(self, pos):
        if not len(self.positions):
            return
        pos = int(pos) % len(self.positions)
        self.pos = pos
        i = int(self.positions[pos])
        step = int(self.src.steps[i])
        field = self.src.frame(i)
        for view, im in zip(self.views, self.images):
            im.set_data(self._render(view, field))
        for line in self.cursors:
            line.set_xdata([step, step])
        text = f"{self.src.title} | frame {i} of {len(self.src)} | tick {step}"
        if self.src.dt:
            text += f" | t = {step * self.src.dt:.4g} s"
        if step in self.src.info:
            text += f" | {self.src.info[step]}"
        self.suptitle.set_text(text)
        if self.interactive and self.slider is not None and not self._syncing:
            self._syncing = True
            self.slider.set_val(pos)
            self._syncing = False
        if self.interactive:
            self.fig.canvas.draw_idle()

    # -- interaction ------------------------------------------------------
    def _on_slider(self, val):
        if not self._syncing:
            self._syncing = True
            self.show(int(round(val)))
            self._syncing = False

    def _on_tick(self, text):
        text = text.strip()
        if not text:
            return
        try:
            tick = int(float(text))
        except ValueError:
            self.log(f"not a tick number: {text!r}")
            return
        steps = self.src.steps[self.positions]
        self.show(int(np.argmin(np.abs(steps - tick))))

    def _advance(self):
        if self.pos + 1 >= len(self.positions):
            self.show(0)
        else:
            self.show(self.pos + 1)

    def toggle_play(self):
        if len(self.positions) < 2:
            return
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.widgets["play"].label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _toggle(self, a, b):
        changed = False
        for k, (view, ax, im) in enumerate(zip(self.views, self.axes, self.images)):
            if view in (a, b):
                new = b if view == a else a
                if new not in _VIEWS_FOR[self.src.content]:
                    continue
                self.views[k] = new
                self._style(ax, im, new)
                changed = True
        if changed:
            self.show(self.pos)

    def snapshot(self):
        i = int(self.positions[self.pos])
        step = int(self.src.steps[i])
        self.src.snap_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"sample_{self.src.sample:05d}_" if self.src.sample is not None else ""
        stem = self.src.snap_dir / f"{prefix}frame_{i:06d}_tick_{step}"
        self.fig.savefig(stem.with_suffix(".png"), dpi=130)
        np.save(stem.with_suffix(".npy"), self.src.frame(i))
        self.log(f"snapshot: {stem}.png and .npy")

    def _switch_sample(self, delta):
        src = self.src
        if src.dataset is None or src.num_samples is None:
            return
        new = (src.sample + delta) % src.num_samples
        self.src = open_source(src.dataset, sample=new)
        sel = self.selection
        self.positions = select_indices(len(self.src), None, stride=sel.get("stride", 1),
                                        start=sel.get("start"), stop=sel.get("stop"))
        if self.slider is not None:
            self.slider.valmax = max(1, len(self.positions) - 1)
            self.slider.ax.set_xlim(0, self.slider.valmax)
        if self.series_axes:
            self._plot_series()
        self.show(0)

    def _on_key(self, event):
        if getattr(self, "textbox", None) is not None and self.textbox.capturekeystrokes:
            return
        key = event.key
        if key == " ":
            self.toggle_play()
        elif key == "right":
            self.show(self.pos + 1)
        elif key == "left":
            self.show(self.pos - 1)
        elif key in ("up", "pageup"):
            self.show(min(self.pos + 10, len(self.positions) - 1))
        elif key in ("down", "pagedown"):
            self.show(max(self.pos - 10, 0))
        elif key == "home":
            self.show(0)
        elif key == "end":
            self.show(len(self.positions) - 1)
        elif key == "t":
            self._toggle("phase", "phase_tt")
        elif key == "l":
            self._toggle("intensity", "log_intensity")
        elif key == "s":
            self.snapshot()
        elif key == "n":
            self._switch_sample(+1)
        elif key == "p":
            self._switch_sample(-1)

    # -- output -----------------------------------------------------------
    def save(self, path, fps=20, max_frames=300, single=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if single or suffix in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
            self.fig.savefig(path, dpi=130)
            return path
        from matplotlib import animation

        if suffix == ".gif":
            writer = _GifWriter(fps=fps)
        elif suffix in (".mp4", ".mkv", ".mov", ".avi"):
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(f"writing {suffix} needs ffmpeg on PATH; install it or save a .gif")
            writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
        else:
            raise ValueError(f"unsupported output {path.name}; use .gif, .mp4 or .png")
        count = len(self.positions)
        if max_frames and count > max_frames:
            self.log(f"rendering the first {max_frames} of {count} selected frames "
                     "(use --stride, --start/--stop or --max-frames 0 for more)")
            count = int(max_frames)
        report = max(1, count // 10)
        with writer.saving(self.fig, str(path), dpi=100):
            for k in range(count):
                self.show(k)
                writer.grab_frame()
                if (k + 1) % report == 0 or k + 1 == count:
                    self.log(f"rendered {k + 1}/{count} frames")
        return path


def _gif_writer_class():
    from matplotlib import animation

    class GifWriter(animation.PillowWriter):
        """PillowWriter that keeps frames as 8-bit palette images (4x less RAM than RGBA)."""

        def grab_frame(self, **savefig_kwargs):
            super().grab_frame(**savefig_kwargs)
            try:
                self._frames[-1] = self._frames[-1].convert("RGB").quantize(colors=256)
            except Exception:  # noqa: BLE001 - keep the RGBA frame if Pillow internals differ
                pass

    return GifWriter


def _GifWriter(fps):
    return _gif_writer_class()(fps=fps)


def view(path, *, views=None, start=None, stop=None, stride=1, frame=None, tick=None, sample=0,
         series=True, save=None, fps=20, max_frames=300, dx=None, steps=None, dt=None, title=None,
         log=print):
    """Open the interactive viewer, or render to ``save`` (``.gif``, ``.mp4`` or ``.png``).

    Parameters
    ----------
    path : run / pass / dataset directory, sample ``.npz``, frames ``.npy``,
        or an in-memory ``[T, N, N]`` array / :class:`FrameSeries`.
    views : list of {"intensity", "log_intensity", "phase", "phase_tt"} or "all"
    start, stop, stride : frame range to show
    frame, tick : initial frame (by index or simulation tick); with a
        ``.png`` ``save`` this is the frame that is rendered.
    sample : dataset sample index.
    series : show the time-series strip when data is available.
    max_frames : cap on frames rendered into an animation (0 = no cap).
    dx, steps, dt, title : metadata for in-memory arrays.

    Returns the written path when ``save`` is given, else ``None`` (after the
    window is closed).
    """
    src = open_source(path, sample=sample)
    if dx is not None:
        src.dx = float(dx)
    if steps is not None:
        src.steps = np.asarray(steps, np.int64)
    if dt is not None:
        src.dt = float(dt)
    if title is not None:
        src.title = title
    resolved = _resolve_views(views, src.content)
    positions = select_indices(len(src), None, stride=stride, start=start, stop=stop)
    if not len(positions):
        raise ValueError(f"no frames selected (source has {len(src)}; check --start/--stop/--stride)")
    start_pos = 0
    if frame is not None:
        start_pos = int(np.argmin(np.abs(positions - int(frame))))
    elif tick is not None:
        start_pos = int(np.argmin(np.abs(src.steps[positions] - int(tick))))

    if save:
        single = Path(save).suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg")
        viewer = _Viewer(src, resolved, positions, series=series, interactive=False, fps=fps,
                         start_pos=start_pos, log=log)
        if not single and start_pos:
            viewer.positions = positions[start_pos:]
        return viewer.save(save, fps=fps, max_frames=max_frames, single=single)

    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() in _NON_INTERACTIVE:
        raise RuntimeError(f"matplotlib backend {matplotlib.get_backend()!r} cannot open a window (no display?). "
                           "Use --save movie.gif / frame.png instead.")
    saved = {k: list(matplotlib.rcParams[k]) for k in _KEYMAPS if k in matplotlib.rcParams}
    try:
        for k, keys in saved.items():
            matplotlib.rcParams[k] = [x for x in keys if x not in _OUR_KEYS]
        viewer = _Viewer(src, resolved, positions, series=series, interactive=True, fps=fps,
                         start_pos=start_pos,
                         selection={"start": start, "stop": stop, "stride": stride}, log=log)
        if log and len(positions) > 1:
            log("controls: space play/pause | left/right step | up/down (PgUp/PgDn) 10 frames | home/end | "
                "t tilt removal | l log scale | s snapshot" + (" | n/p sample" if src.dataset else ""))
        if log and not src.series and series and src.dataset is None:
            log("tip: run `opal analyze fiber <run>` (or r0) to add a time-series strip")
        plt.show()
    finally:
        for k, keys in saved.items():
            matplotlib.rcParams[k] = keys
    return None


# ----------------------------------------------------------------------
# Backward-compatible helpers (OPAL 1.0)
# ----------------------------------------------------------------------
def animate_frames(frames, *, dx=None, steps=None, dt=None, title="OPAL", save=None, fps=20):
    """Animate ``[F, N, N]`` complex frames as irradiance + phase panels.

    ``save`` ending in ``.mp4`` needs ffmpeg; ``.gif`` uses Pillow. Without
    ``save`` an interactive window opens.
    """
    return view(frames, dx=dx, steps=steps, dt=dt, title=title, save=save, fps=fps, max_frames=0,
                series=False)


def animate_run(run_dir, *, save=None, stride=1, fps=20, max_frames=300):
    """Animate a run or pass directory (see :func:`view` for the full set of options)."""
    return view(run_dir, save=save, stride=stride, fps=fps, max_frames=max_frames)
