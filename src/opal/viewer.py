"""Quick-look visualisation of saved runs (requires matplotlib)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["field_images", "animate_frames", "animate_run"]


def field_images(frame):
    """Normalised irradiance and wrapped phase of a complex field."""
    frame = np.asarray(frame)
    intensity = np.abs(frame) ** 2
    return intensity / max(float(intensity.mean()), 1e-30), np.angle(frame)


def animate_frames(frames, *, dx=None, steps=None, dt=None, title="OPAL", save=None, fps=20):
    """Animate ``[F, N, N]`` complex frames as irradiance + phase panels.

    ``save`` ending in ``.mp4`` needs ffmpeg; ``.gif`` uses Pillow. Without
    ``save`` an interactive window opens.
    """
    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    frames = np.asarray(frames)
    n = frames.shape[-1]
    extent = None
    if dx:
        half = n * dx / 2.0
        extent = (-half, half, -half, half)
    I0, P0 = field_images(frames[0])
    vmax = float(np.percentile(I0, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    im_i = axes[0].imshow(I0, cmap="inferno", vmin=0, vmax=vmax, extent=extent, origin="lower")
    im_p = axes[1].imshow(P0, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent, origin="lower")
    axes[0].set_title("irradiance / mean")
    axes[1].set_title("phase (rad)")
    for ax in axes:
        ax.set_xlabel("x (m)" if dx else "x (px)")
    fig.colorbar(im_i, ax=axes[0], shrink=0.85)
    fig.colorbar(im_p, ax=axes[1], shrink=0.85)
    sup = fig.suptitle(title)

    def update(k):
        I, P = field_images(frames[k])
        im_i.set_data(I)
        im_p.set_data(P)
        step = int(steps[k]) if steps is not None else k
        label = f"{title} | tick {step}"
        if dt:
            label += f" | t = {step * dt * 1e3:.1f} ms"
        sup.set_text(label)
        return im_i, im_p, sup

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if save:
        save = str(save)
        writer = animation.PillowWriter(fps=fps) if save.lower().endswith(".gif") else \
            animation.FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(save, writer=writer)
        plt.close(fig)
        return save
    plt.show()
    return None


def animate_run(run_dir, *, save=None, stride=1, fps=20, max_frames=300):
    """Animate ``frames.npy`` (``opal run``) or ``wavefronts.npy`` (``opal pass``) of a run directory."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ch = manifest["config"]["channel_params"]
    dt = manifest["config"]["simulation_params"]["time_per_tick"]
    if (run_dir / "frames.npy").exists():
        frames = np.load(run_dir / "frames.npy", mmap_mode="r")
        steps = np.arange(frames.shape[0]) + int(manifest.get("start_step", 0))
    elif (run_dir / "wavefronts.npy").exists():
        frames = np.load(run_dir / "wavefronts.npy", mmap_mode="r")
        steps = np.load(run_dir / "wavefront_steps.npy")
    else:
        raise FileNotFoundError(f"{run_dir} has no frames.npy or wavefronts.npy")
    idx = np.arange(0, frames.shape[0], max(1, int(stride)))[: int(max_frames)]
    title = f"OPAL N={ch['phase_screen_size']} r0={ch['total_r0']} m"
    return animate_frames(np.asarray(frames[idx]), dx=float(ch["pixel_size"]), steps=steps[idx], dt=dt,
                          title=title, save=save, fps=fps)
