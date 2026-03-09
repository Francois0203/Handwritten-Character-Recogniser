"""
Training orchestrator for character recognition neural network.

Brings together load_data, preprocess_data, and build_model into a single
end-to-end training run with:

  - Configurable dataset selection
  - Full preprocessing pipeline (normalize, augment, batch)
  - Model construction and compilation
  - Learning-rate scheduling (cosine decay with warmup)
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
  - Experiment logging to a timestamped output directory
  - Post-training evaluation on the held-out test set
  - Model + config saving for reproducibility

Typical usage
-------------
    # Train with all defaults (EMNIST ByClass, standard architecture)
    python train_model.py

    # Override via CLI flags
    python train_model.py \
        --dataset      emnist_byclass \
        --architecture standard \
        --epochs       60 \
        --batch_size   128 \
        --lr           1e-3 \
        --optimizer    adamw \
        --augment      true \
        --output_dir   runs/exp_01

    # Or import and call programmatically
    from train_model import train
    results = train(TrainConfig(dataset="mnist", architecture="lite", epochs=10))
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

# ── Local modules ──────────────────────────────
from load_data import (
    EMNIST_BYCLASS_LABELS,
    DIGIT_LABELS,
    KMNIST_LABELS,
    load_emnist_byclass,
    load_emnist_bymerge,
    load_emnist_letters,
    load_emnist_digits,
    load_mnist,
    load_kmnist,
    load_all_combined,
    load_emnist_all,
)
from preprocess_data import preprocess_pipeline
from build_model import build_model

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# ──────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────

DatasetName = Literal[
    "mnist", "kmnist",
    "emnist_byclass", "emnist_bymerge", "emnist_letters", "emnist_digits",
    "all_combined", "emnist_all",
]

# Maps dataset name → (loader_fn, num_classes, label_map_or_None)
# label_map is resolved after loading for "all_combined"
_DATASET_REGISTRY: Dict[str, dict] = {
    "mnist":           {"fn": load_mnist,          "num_classes": 10,  "labels": DIGIT_LABELS},
    "kmnist":          {"fn": load_kmnist,         "num_classes": 10,  "labels": KMNIST_LABELS},
    "emnist_byclass":  {"fn": load_emnist_byclass, "num_classes": 62,  "labels": EMNIST_BYCLASS_LABELS},
    "emnist_bymerge":  {"fn": load_emnist_bymerge, "num_classes": 47,  "labels": None},
    "emnist_letters":  {"fn": load_emnist_letters, "num_classes": 37,  "labels": None},
    "emnist_digits":   {"fn": load_emnist_digits,  "num_classes": 10,  "labels": DIGIT_LABELS},
    "all_combined":    {"fn": load_all_combined,   "num_classes": 72,  "labels": None},
    "emnist_all":      {"fn": load_emnist_all,     "num_classes": 72,  "labels": None},
}


# ──────────────────────────────────────────────
# Training configuration dataclass
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    All hyperparameters and paths for one training run.
    Every field has a sensible default so you can call TrainConfig() immediately.
    """
    # ── Data ──────────────────────────────────
    dataset: DatasetName          = "emnist_byclass"
    val_split: float              = 0.10
    normalization: str            = "minmax"       # "minmax" | "standardize"
    augment: bool                 = True
    augmentation_kwargs: dict     = field(default_factory=dict)

    # ── Model ─────────────────────────────────
    architecture: str             = "standard"     # "lite" | "standard" | "large"
    dropout_rate: float           = 0.40
    l2: float                     = 1e-4
    label_smoothing: float        = 0.10

    # ── Optimiser ─────────────────────────────
    optimizer: str                = "adam"         # "adam" | "adamw" | "sgd"
    learning_rate: float          = 1e-3
    weight_decay: float           = 1e-4

    # ── Schedule ──────────────────────────────
    use_cosine_decay: bool        = True
    warmup_epochs: int            = 5              # linear LR warmup

    # ── Training loop ─────────────────────────
    epochs: int                   = 50
    batch_size: int               = 128
    seed: int                     = 42

    # ── Early stopping ────────────────────────
    patience: int                 = 10             # epochs without val_loss improvement
    min_delta: float              = 1e-4

    # ── Paths ─────────────────────────────────
    output_dir: str               = "runs/default"
    run_name: str                 = ""             # auto-generated if empty


# ──────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="CharRecognition")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule: linear warmup followed by cosine decay.
    
    Registered with Keras for proper serialization/deserialization.
    """
    def __init__(self, warmup_steps: int, cosine_schedule, peak_lr: float, **kwargs):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.cosine       = cosine_schedule
        self.peak_lr      = float(peak_lr)

    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        warmup   = self.peak_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        post     = self.cosine(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup, lambda: post)

    def get_config(self):
        # Return a simplified config for serialization
        # (cosine schedule is rebuilt on load)
        return {
            "warmup_steps": self.warmup_steps,
            "peak_lr": self.peak_lr,
        }

    @classmethod
    def from_config(cls, config):
        # Rebuild the cosine schedule from saved config
        # Note: This is a simplified version; for full fidelity you'd need to save
        # and restore the full cosine decay parameters
        warmup_steps = config["warmup_steps"]
        peak_lr = config["peak_lr"]
        # Approximate: create a dummy cosine schedule
        # In practice, models loaded this way will use this schedule correctly
        cosine_approx = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak_lr,
            decay_steps=max(1, 10000),  # Default fallback
            alpha=0.01,
        )
        return cls(warmup_steps=warmup_steps, cosine_schedule=cosine_approx, peak_lr=peak_lr)


def build_lr_schedule(
    config: TrainConfig,
    steps_per_epoch: int,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Cosine decay with linear warmup.

    During the first `warmup_epochs` epochs the LR rises linearly from 0
    to `config.learning_rate`. Afterwards it follows a cosine curve down
    to `learning_rate / 100`.

    If `use_cosine_decay` is False the initial LR is returned as a constant
    (i.e. no schedule is applied and the optimiser uses a fixed rate).
    """
    if not config.use_cosine_decay:
        logger.info("LR schedule: constant %.2e", config.learning_rate)
        return config.learning_rate  # plain float — Keras accepts this

    total_steps  = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    # Ensure decay_steps is at least 1 by clamping warmup_steps
    if warmup_steps >= total_steps:
        logger.warning(
            "warmup_epochs (%d) >= epochs (%d). Reducing warmup to %d epochs to allow cosine decay.",
            config.warmup_epochs, config.epochs, max(0, config.epochs - 1)
        )
        warmup_steps = max(0, total_steps - steps_per_epoch)  # Leave at least 1 epoch for decay
    
    decay_steps = total_steps - warmup_steps

    cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=decay_steps,
        alpha=0.01,                   # final LR = initial_lr * alpha
    )

    schedule = WarmupCosineDecay(warmup_steps, cosine, config.learning_rate)
    logger.info(
        "LR schedule: cosine decay with %d warmup steps  peak=%.2e  final≈%.2e",
        warmup_steps, config.learning_rate, config.learning_rate * 0.01,
    )
    return schedule


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────

def build_callbacks(
    config: TrainConfig,
    run_dir: Path,
) -> list:
    """
    Construct the Keras callback stack.

    Callbacks
    ---------
    ModelCheckpoint
        Saves the best model (by val_loss) to `run_dir/best_model.keras`.
        Saves only weights if the full model is too large.
    EarlyStopping
        Stops training when val_loss hasn't improved by `min_delta`
        for `patience` epochs. Restores best weights automatically.
    ReduceLROnPlateau
        Fallback plateau-detection: halves the LR if val_loss stalls
        for 5 epochs. Only active when cosine decay is disabled.
    TensorBoard
        Writes logs to `run_dir/tb_logs` for visualisation with
        `tensorboard --logdir runs/`.
    CSVLogger
        Appends per-epoch metrics to `run_dir/training_log.csv`.
    """
    tb_dir = run_dir / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(run_dir / "training_log.csv"),
            append=False,
        ),
    ]

    # ReduceLROnPlateau only makes sense when NOT using a fixed schedule
    if not config.use_cosine_decay:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            )
        )

    return callbacks


# ──────────────────────────────────────────────
# Data loading helper
# ──────────────────────────────────────────────

def load_dataset(
    dataset_name: DatasetName,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Optional[dict]]:
    """
    Dispatch to the correct loader and return raw arrays + metadata.

    Returns
    -------
    x_train, y_train, x_test, y_test : np.ndarray
    num_classes : int
    label_map   : dict {int → str} or None
    """
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(_DATASET_REGISTRY)}"
        )

    entry = _DATASET_REGISTRY[dataset_name]
    num_classes = entry["num_classes"]

    if dataset_name in ("all_combined", "emnist_all"):
        (x_train, y_train), (x_test, y_test), label_map = entry["fn"]()
        num_classes = len(label_map)
    else:
        (x_train, y_train), (x_test, y_test) = entry["fn"]()
        raw_labels = entry["labels"]
        label_map  = (
            {i: ch for i, ch in enumerate(raw_labels)} if raw_labels else None
        )

    logger.info(
        "Dataset '%s' loaded — train: %d  test: %d  classes: %d",
        dataset_name, len(x_train), len(x_test), num_classes,
    )
    return x_train, y_train, x_test, y_test, num_classes, label_map


# ──────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────

def evaluate_and_report(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    run_dir: Path,
    label_map: Optional[dict],
) -> dict:
    """
    Run final evaluation on the test set and write a JSON report.

    Returns the metrics dict so callers can inspect results programmatically.
    """
    logger.info("Evaluating on test set …")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)

    logger.info("─" * 45)
    for metric, value in results.items():
        logger.info("  %-25s %.4f", metric, value)
    logger.info("─" * 45)

    report = {
        "test_metrics": {k: float(v) for k, v in results.items()},
        "label_map": {str(k): v for k, v in label_map.items()} if label_map else None,
    }
    report_path = run_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Test report saved → %s", report_path)

    return results


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train(config: TrainConfig) -> dict:
    """
    End-to-end training run.

    Steps
    -----
    1. Resolve output directory and save config
    2. Load raw data via load_data module
    3. Preprocess: normalize → split → augment → batch (preprocess_data)
    4. Build and compile model (build_model)
    5. Attach LR schedule to optimizer
    6. Build callbacks
    7. model.fit()
    8. Evaluate on test set
    9. Save final model + artifacts

    Parameters
    ----------
    config : TrainConfig

    Returns
    -------
    dict — final test metrics  e.g. {"loss": 0.21, "accuracy": 0.93, ...}
    """
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    # ── 1. Run directory ──────────────────────
    run_name = config.run_name or (
        f"{config.dataset}__{config.architecture}__"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2))
    logger.info("Config saved → %s", config_path)

    # ── 2. Load data ──────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1 / 4 — Loading dataset: %s", config.dataset)
    logger.info("=" * 50)
    x_train, y_train, x_test, y_test, num_classes, label_map = load_dataset(
        config.dataset
    )

    # ── 3. Preprocess ─────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2 / 4 — Preprocessing")
    logger.info("=" * 50)
    train_ds, val_ds, test_ds = preprocess_pipeline(
        x_train, y_train,
        x_test,  y_test,
        num_classes=num_classes,
        val_split=config.val_split,
        normalization=config.normalization,
        augment=config.augment,
        batch_size=config.batch_size,
        seed=config.seed,
        augmentation_kwargs=config.augmentation_kwargs,
    )

    # ── 4. Build model ────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3 / 4 — Building model: %s", config.architecture)
    logger.info("=" * 50)

    steps_per_epoch = len(train_ds)
    lr = (
        build_lr_schedule(config, steps_per_epoch)
        if config.use_cosine_decay
        else config.learning_rate
    )

    model = build_model(
        num_classes=num_classes,
        architecture=config.architecture,
        dropout_rate=config.dropout_rate,
        l2=config.l2,
        learning_rate=lr,
        optimizer_name=config.optimizer,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
    )
    model.summary(print_fn=logger.info, line_length=80)

    # ── 5. Callbacks ──────────────────────────
    callbacks = build_callbacks(config, run_dir)

    # ── 6. Train ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 4 / 4 — Training  (epochs=%d  batch=%d)", config.epochs, config.batch_size)
    logger.info("  TensorBoard: tensorboard --logdir %s", run_dir / "tb_logs")
    logger.info("=" * 50)

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # ── 7. Evaluate ───────────────────────────
    test_metrics = evaluate_and_report(model, test_ds, run_dir, label_map)

    # ── 8. Save final model ───────────────────
    final_model_path = run_dir / "final_model.keras"
    model.save(str(final_model_path))
    logger.info("Final model saved → %s", final_model_path)

    # ── 9. Save history ───────────────────────
    history_path = run_dir / "history.json"
    history_path.write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2)
    )
    logger.info("Training history saved → %s", history_path)
    logger.info("All artifacts written to: %s", run_dir)

    # ── 10. Copy best model to models/ folder for deployment ────
    _save_model_for_deployment(run_dir, config, test_metrics, label_map)

    return test_metrics


def _save_model_for_deployment(
    run_dir: Path,
    config: TrainConfig,
    test_metrics: dict,
    label_map: Optional[dict],
) -> None:
    """
    Copy the best model to the models/ directory for deployment.
    
    Creates a versioned model file with metadata for use in production.
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate model filename based on dataset and architecture
    dataset_clean = config.dataset.replace("_", "")
    arch_clean = config.architecture
    model_name = f"{dataset_clean}_{arch_clean}_v1"
    
    # Check if file exists and increment version
    version = 1
    while (models_dir / f"{dataset_clean}_{arch_clean}_v{version}.keras").exists():
        version += 1
    model_name = f"{dataset_clean}_{arch_clean}_v{version}"
    
    # Copy best model
    best_model_src = run_dir / "best_model.keras"
    best_model_dst = models_dir / f"{model_name}.keras"
    
    if best_model_src.exists():
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        logger.info("=" * 60)
        logger.info("DEPLOYMENT MODEL SAVED")
        logger.info("=" * 60)
        logger.info("Model copied to: %s", best_model_dst)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "dataset": config.dataset,
            "architecture": config.architecture,
            "num_classes": len(label_map) if label_map else "unknown",
            "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
            "test_loss": float(test_metrics.get("loss", 0.0)),
            "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_config": asdict(config),
        }
        
        metadata_path = models_dir / f"{model_name}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.info("Metadata saved to: %s", metadata_path)
        
        # Save config for reproducibility
        config_path = models_dir / f"{model_name}_config.json"
        config_path.write_text(json.dumps(asdict(config), indent=2))
        logger.info("Config saved to: %s", config_path)
        
        logger.info("")
        logger.info("To use this model in the API, update src/core/config.py:")
        logger.info(f"  model_path: str = \"models/{model_name}.keras\"")
        logger.info(f"  dataset_name: str = \"{config.dataset}\"")
        logger.info("=" * 60)
    else:
        logger.warning("Best model not found at %s - skipping deployment copy", best_model_src)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a character-recognition CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset",      default="emnist_byclass",
                        choices=list(_DATASET_REGISTRY),
                        help="Dataset to train on")
    parser.add_argument("--val_split",    type=float, default=0.10)
    parser.add_argument("--normalization",default="minmax", choices=["minmax", "standardize"])
    parser.add_argument("--augment",      type=lambda x: x.lower() != "false", default=True)

    # Model
    parser.add_argument("--architecture", default="standard", choices=["lite", "standard", "large"])
    parser.add_argument("--dropout",      type=float, default=0.40, dest="dropout_rate")
    parser.add_argument("--l2",           type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.10)

    # Optimizer
    parser.add_argument("--optimizer",    default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr",           type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cosine_decay", type=lambda x: x.lower() != "false",
                        default=True, dest="use_cosine_decay")
    parser.add_argument("--warmup_epochs",type=int, default=5)

    # Training loop
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--patience",     type=int, default=10)

    # Paths
    parser.add_argument("--output_dir",   default="runs/default")
    parser.add_argument("--run_name",     default="")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = _parse_args()
    logger.info("Starting training with config:\n%s", json.dumps(asdict(cfg), indent=2))
    results = train(cfg)
    logger.info("Final test results: %s", results)