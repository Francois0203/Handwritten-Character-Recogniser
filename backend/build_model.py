"""
Model builder for character recognition neural network.

Provides three architectures of increasing capacity:

  "lite"     — small CNN, fast to train, good for MNIST / EMNIST Digits
  "standard" — deeper CNN with residual-style skip connections, recommended
               for EMNIST ByClass (62 classes) and similar
  "large"    — wider/deeper variant for the full combined dataset (72 classes)

All builders follow the same signature and return a compiled tf.keras.Model
ready for model.fit().

Typical usage
-------------
    from build_model import build_model

    model = build_model(
        num_classes=62,
        architecture="standard",   # "lite" | "standard" | "large"
        input_shape=(28, 28, 1),
        dropout_rate=0.4,
        learning_rate=1e-3,
        label_smoothing=0.1,
    )
    model.summary()
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

Architecture = Literal["lite", "standard", "large"]

# ──────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────

def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    l2: float = 1e-4,
    name_prefix: str = "",
) -> tf.Tensor:
    """Conv2D → BatchNorm → ReLU block (standard backbone unit)."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,                          # BN handles bias
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}_conv" if name_prefix else None,
    )(x)
    x = layers.BatchNormalization(
        name=f"{name_prefix}_bn" if name_prefix else None
    )(x)
    x = layers.Activation(
        "relu",
        name=f"{name_prefix}_relu" if name_prefix else None
    )(x)
    return x


def _residual_block(
    x: tf.Tensor,
    filters: int,
    l2: float = 1e-4,
    name_prefix: str = "",
) -> tf.Tensor:
    """
    Pre-activation residual block (He et al., 2016).

    If the channel count changes, a 1×1 conv projects the shortcut.
    Spatial dimensions are preserved (no downsampling inside the block).
    """
    shortcut = x
    in_filters = x.shape[-1]

    x = _conv_bn_relu(x, filters, 3, l2=l2, name_prefix=f"{name_prefix}_a")
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}_b_conv" if name_prefix else None,
    )(x)
    x = layers.BatchNormalization(
        name=f"{name_prefix}_b_bn" if name_prefix else None
    )(x)

    # Project shortcut if channel dimensions differ
    if in_filters != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(l2),
            name=f"{name_prefix}_proj" if name_prefix else None,
        )(shortcut)
        shortcut = layers.BatchNormalization(
            name=f"{name_prefix}_proj_bn" if name_prefix else None
        )(shortcut)

    x = layers.Add(name=f"{name_prefix}_add" if name_prefix else None)([x, shortcut])
    x = layers.Activation(
        "relu",
        name=f"{name_prefix}_out_relu" if name_prefix else None
    )(x)
    return x


# ──────────────────────────────────────────────
# Architecture builders
# ──────────────────────────────────────────────

def _build_lite(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
    l2: float,
) -> tf.keras.Model:
    """
    Lightweight CNN — ~200K parameters.

    Best for: MNIST, EMNIST Digits, quick experiments.
    Structure: 2× ConvBlock → MaxPool → 2× ConvBlock → MaxPool → Dense head
    """
    inputs = layers.Input(shape=input_shape, name="input")

    x = _conv_bn_relu(inputs, 32,  name_prefix="c1")
    x = _conv_bn_relu(x,      32,  name_prefix="c2")
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    x = _conv_bn_relu(x, 64, name_prefix="c3")
    x = _conv_bn_relu(x, 64, name_prefix="c4")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop2")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name="CharNet_Lite")


def _build_standard(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
    l2: float,
) -> tf.keras.Model:
    """
    Standard residual CNN — ~1.2M parameters.

    Best for: EMNIST ByClass (62 classes), EMNIST Letters, most use-cases.
    Structure: Stem → 3× Residual stage with MaxPool → Dense head
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Stem
    x = _conv_bn_relu(inputs, 32, kernel_size=3, name_prefix="stem")

    # Stage 1 — 32 filters
    x = _residual_block(x, 32,  l2=l2, name_prefix="res1a")
    x = _residual_block(x, 32,  l2=l2, name_prefix="res1b")
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.3, name="drop1")(x)

    # Stage 2 — 64 filters
    x = _residual_block(x, 64,  l2=l2, name_prefix="res2a")
    x = _residual_block(x, 64,  l2=l2, name_prefix="res2b")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.3, name="drop2")(x)

    # Stage 3 — 128 filters
    x = _residual_block(x, 128, l2=l2, name_prefix="res3a")
    x = _residual_block(x, 128, l2=l2, name_prefix="res3b")
    x = layers.Dropout(dropout_rate * 0.3, name="drop3")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name="CharNet_Standard")


def _build_large(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
    l2: float,
) -> tf.keras.Model:
    """
    Large residual CNN — ~4M parameters.

    Best for: Full combined dataset (72 classes), maximum accuracy targets.
    Structure: Stem → 4× Residual stage with MaxPool → Dense head
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Stem
    x = _conv_bn_relu(inputs, 64, kernel_size=3, name_prefix="stem")

    # Stage 1 — 64 filters
    x = _residual_block(x, 64,  l2=l2, name_prefix="res1a")
    x = _residual_block(x, 64,  l2=l2, name_prefix="res1b")
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.25, name="drop1")(x)

    # Stage 2 — 128 filters
    x = _residual_block(x, 128, l2=l2, name_prefix="res2a")
    x = _residual_block(x, 128, l2=l2, name_prefix="res2b")
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.25, name="drop2")(x)

    # Stage 3 — 256 filters
    x = _residual_block(x, 256, l2=l2, name_prefix="res3a")
    x = _residual_block(x, 256, l2=l2, name_prefix="res3b")
    x = layers.Dropout(dropout_rate * 0.25, name="drop3")(x)

    # Stage 4 — 256 filters (extra depth, no pool — feature map is 7×7 here)
    x = _residual_block(x, 256, l2=l2, name_prefix="res4a")
    x = layers.Dropout(dropout_rate * 0.25, name="drop4")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(1024, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop5")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="fc2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop6")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name="CharNet_Large")


# ──────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────

def build_optimizer(
    learning_rate: float = 1e-3,
    optimizer_name: Literal["adam", "adamw", "sgd"] = "adam",
    weight_decay: float = 1e-4,
) -> optimizers.Optimizer:
    """
    Construct an optimizer.

    Parameters
    ----------
    learning_rate  : initial learning rate
    optimizer_name : "adam" | "adamw" | "sgd"
        - "adam"  — sensible default, works well out of the box
        - "adamw" — Adam with decoupled weight decay; often better generalisation
        - "sgd"   — SGD + Nesterov momentum; can match AdamW with careful LR scheduling
    weight_decay   : used by "adamw" and "sgd" only

    Returns
    -------
    tf.keras.optimizers.Optimizer
    """
    name = optimizer_name.lower()
    if name == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif name == "adamw":
        opt = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == "sgd":
        opt = optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Use 'adam', 'adamw', or 'sgd'.")

    logger.info("Optimizer: %s  lr=%.2e", optimizer_name, learning_rate)
    return opt


# ──────────────────────────────────────────────
# Loss factory
# ──────────────────────────────────────────────

def build_loss(label_smoothing: float = 0.0) -> tf.keras.losses.Loss:
    """
    Categorical cross-entropy, with optional label smoothing.

    Label smoothing (e.g. 0.1) prevents the model from becoming
    overconfident and tends to improve generalisation on noisy datasets
    like EMNIST where some labels are ambiguous (e.g. 'O' vs '0').
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    logger.info("Loss: CategoricalCrossentropy  label_smoothing=%.2f", label_smoothing)
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def build_model(
    num_classes: int,
    *,
    architecture: Architecture = "standard",
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    dropout_rate: float = 0.4,
    l2: float = 1e-4,
    learning_rate: float = 1e-3,
    optimizer_name: Literal["adam", "adamw", "sgd"] = "adam",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
) -> tf.keras.Model:
    """
    Build and compile a character recognition CNN.

    Parameters
    ----------
    num_classes     : number of output classes (e.g. 10, 47, 62, 72)
    architecture    : "lite" | "standard" | "large"
    input_shape     : (H, W, C) — default (28, 28, 1) for all supported datasets
    dropout_rate    : base dropout probability applied in the dense head
    l2              : L2 weight-decay coefficient applied to all Conv/Dense kernels
    learning_rate   : initial learning rate
    optimizer_name  : "adam" | "adamw" | "sgd"
    weight_decay    : used by AdamW / SGD optimizers
    label_smoothing : smoothing factor for cross-entropy loss (0 = off)

    Returns
    -------
    tf.keras.Model  — compiled, ready for model.fit()

    Examples
    --------
    >>> # Recommended defaults for EMNIST ByClass
    >>> model = build_model(num_classes=62)
    >>> model.summary()

    >>> # Lightweight model for digit-only tasks
    >>> model = build_model(num_classes=10, architecture="lite", dropout_rate=0.3)

    >>> # Full dataset, maximum capacity
    >>> model = build_model(
    ...     num_classes=72,
    ...     architecture="large",
    ...     optimizer_name="adamw",
    ...     label_smoothing=0.1,
    ... )
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be ≥ 2, got {num_classes}")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    logger.info(
        "Building '%s' architecture — classes: %d  input: %s",
        architecture, num_classes, input_shape,
    )

    # 1. Select architecture
    builders = {
        "lite":     _build_lite,
        "standard": _build_standard,
        "large":    _build_large,
    }
    if architecture not in builders:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from: {list(builders)}"
        )
    model = builders[architecture](input_shape, num_classes, dropout_rate, l2)

    # 2. Build optimizer and loss
    optimizer = build_optimizer(learning_rate, optimizer_name, weight_decay)
    loss      = build_loss(label_smoothing)

    # 3. Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    total_params = model.count_params()
    logger.info("Model compiled — total parameters: %s", f"{total_params:,}")

    return model


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for arch in ("lite", "standard", "large"):
        print(f"\n{'=' * 55}")
        print(f"  Architecture: {arch.upper()}")
        print(f"{'=' * 55}")
        m = build_model(num_classes=62, architecture=arch)  # type: ignore[arg-type]
        m.summary(line_length=70)

    print("\nAll architectures built successfully ✓")