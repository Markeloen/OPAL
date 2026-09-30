"""ML dataset generation (and a tiny model trained on it).

    python examples/05_ml_dataset.py            # generate + inspect
    python examples/05_ml_dataset.py --train    # also fit a small CNN that reads r0 off the phase

Sweeps r0 and wind, draws several independent realizations per point, and
stores centre-cropped irradiance/phase sequences with per-frame physical labels
(fiber coupling, scintillation, angle of arrival). Every sample is reproducible
from ``manifest.json`` (base seed + point + realization), and re-running the
command resumes where it stopped. CLI equivalent::

    opal dataset -c configs/quickstart_256.json -o runs/dataset \
        --sweep r0=0.05:0.15:3 --sweep wind=5,15 -k 4 --steps 40 --stride 4 --crop 128
"""

import sys
from pathlib import Path

import numpy as np

import opal
from opal.dataset import load_labels

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "runs" / "dataset"

opal.generate_dataset(
    ROOT / "configs" / "quickstart_256.json",
    DS,
    sweep=["r0=0.05:0.15:3", "wind=5,15"],
    realizations=4,
    num_steps=40,
    stride=4,
    crop=128,
    targets=("intensity", "phase"),
    seed=2026,
    label_aperture=0.3,
)

labels = load_labels(DS)
print(f"{len(labels)} samples")
for r0 in sorted({row["r0"] for row in labels}):
    rows = [row for row in labels if row["r0"] == r0]
    print(f"  r0 {r0:.3f} m: eta {np.mean([r['eta_mean'] for r in rows]):.3f}, "
          f"scintillation {np.mean([r['scint_mean'] for r in rows]):.3f}, "
          f"measured r0 {np.nanmean([r['r0_measured'] for r in rows]):.3f} m")

if "--train" in sys.argv:
    import tensorflow as tf

    X, y = opal.load_dataset(DS, target="phase", label_keys=("r0",))        # X [S, F, n, n]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    split = int(0.75 * len(X))                              # split by sample, not by frame:
    train, test = order[:split], order[split:]              # frames of one sample are correlated

    def frames_of(idx):
        p = X[idx].reshape(-1, *X.shape[2:])                 # every frame is an image
        x = np.stack([np.cos(p), np.sin(p)], axis=-1)       # wrapped phase -> no 2*pi jumps
        return x.astype(np.float32), np.repeat(y[idx], X.shape[1], axis=0)

    (x_tr, y_tr), (x_te, y_te) = frames_of(train), frames_of(test)
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(x_tr.shape[1:]),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="mse")
    model.fit(x_tr, y_tr, epochs=30, batch_size=16, verbose=0, validation_data=(x_te, y_te))
    pred = model.predict(x_te, verbose=0).ravel()
    print(f"held-out r0 MAE: {np.mean(np.abs(pred - y_te[:, 0])) * 100:.2f} cm "
          f"(predicting the mean would give {np.mean(np.abs(y_te[:, 0] - y_tr.mean())) * 100:.2f} cm)")
    for r0 in np.unique(y_te[:, 0]):
        m = y_te[:, 0] == r0
        print(f"  true r0 {r0:.3f} m -> predicted {pred[m].mean():.3f} +- {pred[m].std():.3f} m")
