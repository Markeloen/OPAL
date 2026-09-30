"""TensorFlow runtime setup shared by all GPU modules.

Imported for its side effect: GPU memory growth is enabled (so the memory
guard can measure real usage instead of TensorFlow's up-front reservation)
and XLA auto-clustering is disabled, matching the configuration the
performance model was measured with. Both are no-ops if TensorFlow has
already initialised the GPU.
"""

import tensorflow as tf


def _configure():
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


_configure()
