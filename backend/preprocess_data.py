"""
Preprocessing pipeline for character recognition neural network training.

Provides:
- Input validation and dtype casting
- Pixel normalization (simple /255 or mean/std standardization)
- Optional per-dataset mean/std computation
- Train/validation split
- Data augmentation via tf.keras layers (rotation, zoom, shift)
- Efficient tf.data.Dataset pipelines with prefetching
- One-hot label encoding

Typical usage
-------------
    from preprocess_data import preprocess_pipeline

    train_ds, val_ds, test_ds = preprocess_pipeline(
        x_train, y_train, x_test, y_test,
        num_classes=62,
        val_split=0.1,
        augment=True,
        batch_size=128,
        normalization="minmax",   # or "standardize"
        seed=42,
    )

    model.fit(train_ds, validation_data=val_ds, epochs=30)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

NormMode = Literal["minmax", "standardize"]

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_inputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> None:
    """
    Sanity-check shapes, dtypes, and label ranges before any processing.

    Raises
    ------
    ValueError
        If any array has an unexpected shape, or labels are out of range.
    """
    for name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        if x.ndim != 4:
            raise ValueError(
                f"x_{name} must be 4-D (N, H, W, C), got shape {x.shape}"
            )
        if x.shape[1:] != (28, 28, 1):
            raise ValueError(
                f"x_{name} spatial shape must be (28, 28, 1), got {x.shape[1:]}"
            )
        if y.ndim != 1:
            raise ValueError(
                f"y_{name} must be 1-D integer labels, got shape {y.shape}"
            )
        if len(x) != len(y):
            raise ValueError(
                f"x_{name} and y_{name} length mismatch: {len(x)} vs {len(y)}"
            )
        label_min, label_max = int(y.min()), int(y.max())
        if label_min < 0 or label_max >= num_classes:
            raise ValueError(
                f"y_{name} labels out of range [0, {num_classes - 1}]: "
                f"found [{label_min}, {label_max}]"
            )

    logger.info(
        "Input validation passed — train: %d samples | test: %d samples | classes: %d",
        len(x_train), len(x_test), num_classes,
    )


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

def normalize(
    x_train: np.ndarray,
    x_test: np.ndarray,
    mode: NormMode = "minmax",
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Normalize pixel values using one of two strategies.

    Parameters
    ----------
    x_train, x_test : np.ndarray  uint8 (N, 28, 28, 1)
    mode : "minmax"
        Scales pixels to [0, 1] by dividing by 255.
        Fast and works well for most CNN architectures.
    mode : "standardize"
        Subtracts training-set mean and divides by std, producing
        zero-mean unit-variance inputs.  Useful when combining
        heterogeneous datasets or using architectures sensitive to scale.
    mean, std : float, optional
        Pre-computed statistics. If not supplied they are derived from
        x_train so that test data never leaks into the statistics.

    Returns
    -------
    x_train_norm, x_test_norm : np.ndarray  float32
    mean_used, std_used : float
        The statistics actually applied (useful for inference-time normalization).
    """
    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")

    if mode == "minmax":
        mean_used = 0.0
        std_used  = 255.0
        x_train /= 255.0
        x_test  /= 255.0

    elif mode == "standardize":
        if mean is None:
            mean = float(x_train.mean())
        if std is None:
            std = float(x_train.std())
            if std < 1e-7:
                raise ValueError("Training data std is near-zero; cannot standardize.")
        mean_used = mean
        std_used  = std
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std

    else:
        raise ValueError(f"Unknown normalization mode '{mode}'. Use 'minmax' or 'standardize'.")

    logger.info("Normalization: mode=%s  mean=%.4f  std=%.4f", mode, mean_used, std_used)
    return x_train, x_test, mean_used, std_used


# ──────────────────────────────────────────────
# Train / validation split
# ──────────────────────────────────────────────

def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split arrays into train and validation subsets.

    Parameters
    ----------
    x, y    : matched arrays of length N
    val_split : fraction of samples to hold out (default 0.10)
    seed    : reproducibility seed

    Returns
    -------
    x_tr, y_tr, x_val, y_val
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split   = int(len(x) * (1.0 - val_split))

    train_idx, val_idx = indices[:split], indices[split:]
    logger.info(
        "Train/val split — train: %d  val: %d  (val_split=%.2f)",
        len(train_idx), len(val_idx), val_split,
    )
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


# ──────────────────────────────────────────────
# Augmentation layer
# ──────────────────────────────────────────────

def build_augmentation_layer(
    rotation_factor: float = 0.08,
    zoom_factor: float = 0.10,
    translation_factor: float = 0.10,
    seed: int = 42,
) -> tf.keras.Sequential:
    """
    Build a Keras preprocessing model for on-the-fly augmentation.

    Applied only during training (the layers are no-ops at inference time
    when `training=False` is passed, as tf.data does via `.map`).

    Parameters
    ----------
    rotation_factor    : max rotation as a fraction of 2π (e.g. 0.08 ≈ ±29°)
    zoom_factor        : max zoom as a fraction of image size
    translation_factor : max horizontal/vertical shift as a fraction of size
    seed               : random seed for reproducibility

    Returns
    -------
    tf.keras.Sequential  (stateless preprocessing model)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=rotation_factor, fill_mode="nearest", seed=seed
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_factor, zoom_factor),
                width_factor=(-zoom_factor, zoom_factor),
                fill_mode="nearest",
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=translation_factor,
                width_factor=translation_factor,
                fill_mode="nearest",
                seed=seed,
            ),
        ],
        name="augmentation",
    )


# ──────────────────────────────────────────────
# tf.data pipeline builders
# ──────────────────────────────────────────────

def _make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    augmentation_layer: Optional[tf.keras.Sequential],
    seed: int,
) -> tf.data.Dataset:
    """Internal helper — construct one tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    if augment and augmentation_layer is not None:
        ds = ds.map(
            lambda imgs, labels: (augmentation_layer(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# One-hot encoding
# ──────────────────────────────────────────────

def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label array to one-hot float32 matrix."""
    return tf.keras.utils.to_categorical(y, num_classes).astype("float32")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def preprocess_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    *,
    val_split: float = 0.1,
    normalization: NormMode = "minmax",
    augment: bool = True,
    batch_size: int = 128,
    seed: int = 42,
    augmentation_kwargs: Optional[dict] = None,
    precomputed_stats: Optional[Tuple[float, float]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Full preprocessing pipeline: validate → normalize → split → encode → batch.

    Parameters
    ----------
    x_train, y_train : raw training arrays (uint8 images, int labels)
    x_test,  y_test  : raw test arrays
    num_classes      : total number of output classes
    val_split        : fraction of training data held out for validation
    normalization    : "minmax" (÷255) or "standardize" (zero-mean unit-var)
    augment          : apply random rotation / zoom / shift to training batches
    batch_size       : samples per mini-batch
    seed             : global reproducibility seed
    augmentation_kwargs : dict of kwargs forwarded to build_augmentation_layer()
    precomputed_stats   : (mean, std) tuple; only used when normalization="standardize"
                          and you want to reuse statistics from a previous run

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
        Each dataset yields (image_batch, one_hot_label_batch).
        train_ds is shuffled and optionally augmented.
        val_ds and test_ds are unshuffled and unaugmented.
    """
    # 1. Validate
    validate_inputs(x_train, y_train, x_test, y_test, num_classes)

    # 2. Normalize (fit statistics on train only)
    mean, std = precomputed_stats if precomputed_stats else (None, None)
    x_train_f, x_test_f, mean_used, std_used = normalize(
        x_train, x_test, mode=normalization, mean=mean, std=std
    )
    logger.info("Normalization stats stored: mean=%.5f  std=%.5f", mean_used, std_used)

    # 3. Train / validation split
    x_tr, y_tr, x_val, y_val = train_val_split(
        x_train_f, y_train, val_split=val_split, seed=seed
    )

    # 4. One-hot encode labels
    y_tr_oh  = to_one_hot(y_tr,    num_classes)
    y_val_oh = to_one_hot(y_val,   num_classes)
    y_te_oh  = to_one_hot(y_test,  num_classes)

    # 5. Build augmentation layer (shared across calls for consistency)
    aug_layer = None
    if augment:
        aug_kwargs = augmentation_kwargs or {}
        aug_layer  = build_augmentation_layer(seed=seed, **aug_kwargs)
        logger.info("Augmentation enabled with kwargs: %s", aug_kwargs)

    # 6. Build tf.data pipelines
    train_ds = _make_dataset(
        x_tr, y_tr_oh,
        batch_size=batch_size, shuffle=True, augment=augment,
        augmentation_layer=aug_layer, seed=seed,
    )
    val_ds = _make_dataset(
        x_val, y_val_oh,
        batch_size=batch_size, shuffle=False, augment=False,
        augmentation_layer=None, seed=seed,
    )
    test_ds = _make_dataset(
        x_test_f, y_te_oh,
        batch_size=batch_size, shuffle=False, augment=False,
        augmentation_layer=None, seed=seed,
    )

    logger.info(
        "Pipeline ready — train batches: %d | val batches: %d | test batches: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    return train_ds, val_ds, test_ds


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Simulate a tiny dataset for testing without downloading anything
    rng = np.random.default_rng(0)
    x_tr_dummy = rng.integers(0, 255, (1000, 28, 28, 1), dtype=np.uint8)
    y_tr_dummy = rng.integers(0, 10, (1000,), dtype=np.int32)
    x_te_dummy = rng.integers(0, 255, (200, 28, 28, 1), dtype=np.uint8)
    y_te_dummy = rng.integers(0, 10, (200,), dtype=np.int32)

    train_ds, val_ds, test_ds = preprocess_pipeline(
        x_tr_dummy, y_tr_dummy,
        x_te_dummy, y_te_dummy,
        num_classes=10,
        val_split=0.1,
        normalization="minmax",
        augment=True,
        batch_size=32,
        seed=42,
    )

    for images, labels in train_ds.take(1):
        print(f"Train batch — images: {images.shape}  labels: {labels.shape}")
        print(f"  Pixel range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"  Label sample: {labels[0].numpy()}")

    print("Smoke-test passed ✓")