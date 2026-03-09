"""
Evaluation module for character recognition neural network.

Loads a saved model and a dataset, then produces a full suite of diagnostics:

  - Overall test loss / accuracy / top-3 accuracy
  - Per-class precision, recall, F1-score
  - Confusion matrix (raw counts + normalised)
  - Most-confused class pairs
  - Worst predicted samples (highest-loss individual images)
  - All plots saved to the run output directory

Typical usage
-------------
    # Evaluate best checkpoint from a training run
    python evaluate_model.py \
        --model_path  runs/my_run/best_model.keras \
        --dataset     emnist_byclass \
        --output_dir  runs/my_run/eval

    # Or import programmatically
    from evaluate_model import evaluate
    results = evaluate(EvalConfig(
        model_path="runs/my_run/best_model.keras",
        dataset="emnist_byclass",
    ))
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

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
)
from preprocess_data import normalize

# Import custom learning rate schedule to register it with Keras
from train import WarmupCosineDecay  # noqa: F401

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ──────────────────────────────────────────────
# Dataset registry  (mirrors train_model.py)
# ──────────────────────────────────────────────

_DATASET_REGISTRY: Dict[str, dict] = {
    "mnist":           {"fn": load_mnist,          "num_classes": 10,  "labels": DIGIT_LABELS},
    "kmnist":          {"fn": load_kmnist,         "num_classes": 10,  "labels": KMNIST_LABELS},
    "emnist_byclass":  {"fn": load_emnist_byclass, "num_classes": 62,  "labels": EMNIST_BYCLASS_LABELS},
    "emnist_bymerge":  {"fn": load_emnist_bymerge, "num_classes": 47,  "labels": None},
    "emnist_letters":  {"fn": load_emnist_letters, "num_classes": 37,  "labels": None},
    "emnist_digits":   {"fn": load_emnist_digits,  "num_classes": 10,  "labels": DIGIT_LABELS},
    "all_combined":    {"fn": load_all_combined,   "num_classes": 72,  "labels": None},
}

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class EvalConfig:
    model_path:       str   = "runs/default/best_model.keras"
    dataset:          str   = "emnist_byclass"
    normalization:    str   = "minmax"          # must match training config
    batch_size:       int   = 256
    output_dir:       str   = ""                # defaults to model_path/../eval/
    top_k:            int   = 3                 # top-k accuracy to report
    confusion_top_n:  int   = 30                # show only the N most frequent classes in matrix
    worst_n_samples:  int   = 16                # how many worst predictions to visualise
    seed:             int   = 42


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def _load_test_data(
    dataset_name: str,
    normalization: str,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[Dict[int, str]]]:
    """
    Load and normalize the test split only.
    Returns x_test (float32), y_test (int), num_classes, label_map.
    """
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(_DATASET_REGISTRY)}")

    entry = _DATASET_REGISTRY[dataset_name]

    if dataset_name == "all_combined":
        (x_train, _), (x_test, y_test), label_map = entry["fn"]()
        num_classes = len(label_map)
    else:
        (x_train, _), (x_test, y_test) = entry["fn"]()
        num_classes = entry["num_classes"]
        raw_labels  = entry["labels"]
        label_map   = {i: ch for i, ch in enumerate(raw_labels)} if raw_labels else None

    # Normalize using training-set statistics (x_train used only for mean/std)
    x_train_f = x_train.astype("float32")
    x_test_f  = x_test.astype("float32")
    _, x_test_norm, _, _ = normalize(x_train_f, x_test_f, mode=normalization)

    logger.info("Test set: %d samples  |  classes: %d", len(x_test_norm), num_classes)
    return x_test_norm, y_test, num_classes, label_map


def _make_test_dataset(
    x: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    """Batch and prefetch for fast inference. No shuffling."""
    return (
        tf.data.Dataset.from_tensor_slices(x)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


# ──────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────

def get_predictions(
    model: tf.keras.Model,
    x_test: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over the full test set.

    Returns
    -------
    y_pred_classes : (N,)  int   — argmax class for each sample
    y_probs        : (N, C) float — full softmax probability matrix
    """
    logger.info("Running inference on %d samples …", len(x_test))
    ds = _make_test_dataset(x_test, batch_size)
    y_probs = model.predict(ds, verbose=1)
    y_pred_classes = np.argmax(y_probs, axis=1)
    return y_pred_classes, y_probs


def compute_scalar_metrics(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    batch_size: int,
    top_k: int,
) -> dict:
    """
    Compute loss, top-1, and top-k accuracy using the model's compiled metrics.
    """
    y_onehot = tf.keras.utils.to_categorical(y_test, num_classes).astype("float32")
    ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_onehot))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    results = model.evaluate(ds, verbose=1, return_dict=True)
    logger.info("Scalar metrics: %s", {k: f"{v:.4f}" for k, v in results.items()})
    return {k: float(v) for k, v in results.items()}


# ──────────────────────────────────────────────
# Per-class report
# ──────────────────────────────────────────────

def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]],
    num_classes: int,
    output_dir: Path,
) -> dict:
    """
    sklearn classification_report → JSON + log output.
    Returns the report dict.
    """
    target_names = (
        [label_map.get(i, str(i)) for i in range(num_classes)]
        if label_map else [str(i) for i in range(num_classes)]
    )
    report_str  = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True
    )

    logger.info("\nPer-class classification report:\n%s", report_str)

    report_path = output_dir / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info("Per-class report saved → %s", report_path)

    return report_dict


# ──────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]],
    num_classes: int,
    output_dir: Path,
    top_n: int = 30,
) -> None:
    """
    Plot and save both a raw-count and a row-normalised confusion matrix.

    For datasets with many classes only the `top_n` most frequent true classes
    are shown to keep the plot readable.
    """
    # Select top_n classes by frequency in y_true
    class_counts = np.bincount(y_true, minlength=num_classes)
    top_classes  = np.argsort(class_counts)[::-1][:top_n]
    top_classes  = np.sort(top_classes)

    mask     = np.isin(y_true, top_classes)
    y_t_sub  = y_true[mask]
    y_p_sub  = y_pred[mask]

    # Remap labels to 0..top_n for the matrix
    label_to_idx = {cls: i for i, cls in enumerate(top_classes)}
    y_t_mapped   = np.array([label_to_idx[c] for c in y_t_sub])
    y_p_mapped   = np.array([label_to_idx.get(c, -1) for c in y_p_sub])

    valid = y_p_mapped >= 0
    cm = confusion_matrix(y_t_mapped[valid], y_p_mapped[valid], labels=list(range(len(top_classes))))

    tick_labels = (
        [label_map.get(c, str(c)) for c in top_classes]
        if label_map else [str(c) for c in top_classes]
    )

    def _save_cm(matrix: np.ndarray, title: str, fmt: str, path: Path) -> None:
        fig_size = max(10, len(top_classes) // 2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=range(len(top_classes)),
            yticks=range(len(top_classes)),
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            xlabel="Predicted label",
            ylabel="True label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        if len(top_classes) <= 20:
            thresh = matrix.max() / 2.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, format(matrix[i, j], fmt),
                            ha="center", va="center", fontsize=6,
                            color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", path)

    _save_cm(cm, f"Confusion Matrix (top {len(top_classes)} classes — counts)",
             "d", output_dir / "confusion_matrix_counts.png")

    cm_norm = cm.astype("float64")
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums > 0)
    _save_cm(cm_norm, f"Confusion Matrix (top {len(top_classes)} classes — normalised)",
             ".2f", output_dir / "confusion_matrix_normalised.png")


# ──────────────────────────────────────────────
# Most confused pairs
# ──────────────────────────────────────────────

def most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]],
    num_classes: int,
    top_n: int = 20,
    output_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Find and log the most commonly confused (true, predicted) class pairs.
    Saves a bar chart and returns a list of dicts.
    """
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)          # zero out correct predictions

    pairs = []
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            if cm[true_cls, pred_cls] > 0:
                true_lbl = label_map.get(true_cls, str(true_cls)) if label_map else str(true_cls)
                pred_lbl = label_map.get(pred_cls, str(pred_cls)) if label_map else str(pred_cls)
                pairs.append({
                    "true_class":  true_cls,
                    "pred_class":  pred_cls,
                    "true_label":  true_lbl,
                    "pred_label":  pred_lbl,
                    "count":       int(cm[true_cls, pred_cls]),
                })

    pairs.sort(key=lambda p: p["count"], reverse=True)
    top_pairs = pairs[:top_n]

    logger.info("\nTop %d most confused pairs:", top_n)
    for i, p in enumerate(top_pairs, 1):
        logger.info("  %2d. '%s' → '%s'  (%d times)", i, p["true_label"], p["pred_label"], p["count"])

    if output_dir:
        # Bar chart
        labels_axis = [f"'{p['true_label']}' → '{p['pred_label']}'" for p in top_pairs]
        counts      = [p["count"] for p in top_pairs]
        fig, ax = plt.subplots(figsize=(12, max(4, len(top_pairs) // 2)))
        ax.barh(labels_axis[::-1], counts[::-1], color="steelblue")
        ax.set_xlabel("Error count")
        ax.set_title(f"Top {top_n} most confused class pairs")
        plt.tight_layout()
        fig.savefig(output_dir / "most_confused_pairs.png", dpi=150)
        plt.close(fig)
        logger.info("Most confused pairs chart saved → %s", output_dir / "most_confused_pairs.png")

        pairs_path = output_dir / "most_confused_pairs.json"
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(top_pairs, f, indent=2, ensure_ascii=False)

    return top_pairs


# ──────────────────────────────────────────────
# Worst predicted samples
# ──────────────────────────────────────────────

def plot_worst_samples(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_map: Optional[Dict[int, str]],
    output_dir: Path,
    n: int = 16,
) -> None:
    """
    Find the N samples where the model was most wrong (lowest correct-class
    probability) and plot them in a grid with true vs predicted labels.
    """
    y_pred       = np.argmax(y_probs, axis=1)
    wrong_mask   = y_pred != y_true
    wrong_idx    = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        logger.info("No incorrect predictions found — skipping worst-samples plot.")
        return

    # Sort by ascending probability assigned to the true class (worst first)
    true_class_probs = y_probs[wrong_idx, y_true[wrong_idx]]
    sorted_order     = np.argsort(true_class_probs)
    worst_idx        = wrong_idx[sorted_order[:n]]

    cols = 4
    rows = (len(worst_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, idx in zip(axes, worst_idx):
        true_lbl = label_map.get(int(y_true[idx]), str(y_true[idx])) if label_map else str(y_true[idx])
        pred_cls = int(np.argmax(y_probs[idx]))
        pred_lbl = label_map.get(pred_cls, str(pred_cls)) if label_map else str(pred_cls)
        conf     = float(y_probs[idx, pred_cls])

        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        ax.set_title(
            f"True: '{true_lbl}'\nPred: '{pred_lbl}' ({conf:.1%})",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    for ax in axes[len(worst_idx):]:
        ax.axis("off")

    plt.suptitle(f"Worst {len(worst_idx)} predictions (lowest true-class confidence)", fontsize=11)
    plt.tight_layout()
    path = output_dir / "worst_predictions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Worst predictions plot saved → %s", path)


# ──────────────────────────────────────────────
# Per-class accuracy bar chart
# ──────────────────────────────────────────────

def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]],
    num_classes: int,
    output_dir: Path,
) -> None:
    """Bar chart of per-class accuracy, sorted ascending (hardest classes first)."""
    per_class_acc = []
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(float((y_pred[mask] == cls).mean()))

    labels = (
        [label_map.get(i, str(i)) for i in range(num_classes)]
        if label_map else [str(i) for i in range(num_classes)]
    )

    order  = np.argsort(per_class_acc)
    sorted_labels = [labels[i] for i in order]
    sorted_acc    = [per_class_acc[i] for i in order]

    fig_h = max(6, num_classes // 4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = ["#d9534f" if a < 0.7 else "#5cb85c" for a in sorted_acc]
    ax.barh(sorted_labels, sorted_acc, color=colors)
    ax.axvline(x=np.mean(per_class_acc), color="navy", linestyle="--", linewidth=1.2,
               label=f"Mean: {np.mean(per_class_acc):.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-class accuracy (sorted — red < 70%)")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "per_class_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Per-class accuracy chart saved → %s", path)


# ──────────────────────────────────────────────
# Main evaluate function
# ──────────────────────────────────────────────

def evaluate(config: EvalConfig) -> dict:
    """
    Full evaluation pipeline.

    Steps
    -----
    1.  Resolve output directory
    2.  Load model from disk
    3.  Load + normalize test data
    4.  Compute scalar metrics (loss / accuracy / top-k)
    5.  Run inference to get per-sample predictions
    6.  Per-class classification report
    7.  Confusion matrices (counts + normalised)
    8.  Most confused class pairs
    9.  Per-class accuracy bar chart
    10. Worst predictions visualisation
    11. Save summary JSON

    Returns
    -------
    dict — scalar test metrics
    """
    # ── 1. Output dir ────────────────────────────
    out = (
        Path(config.output_dir)
        if config.output_dir
        else Path(config.model_path).parent / "eval"
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluation output directory: %s", out)

    # Save config
    (out / "eval_config.json").write_text(json.dumps(asdict(config), indent=2))

    # ── 2. Load model ────────────────────────────
    logger.info("Loading model from: %s", config.model_path)
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model not found at: {config.model_path}")
    
    # Load without compiling to avoid issues with custom LR schedules
    # (we only need the forward pass for evaluation, not the optimizer)
    model = tf.keras.models.load_model(config.model_path, compile=False)
    
    # Recompile for evaluation metrics
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=config.top_k, name=f"top{config.top_k}_accuracy"),
        ],
    )
    
    logger.info("Model loaded — parameters: %s", f"{model.count_params():,}")
    model.summary(print_fn=logger.info, line_length=80)

    # ── 3. Load data ─────────────────────────────
    logger.info("=" * 50)
    logger.info("Loading test data: %s", config.dataset)
    logger.info("=" * 50)
    x_test, y_test, num_classes, label_map = _load_test_data(
        config.dataset, config.normalization
    )

    # ── 4. Scalar metrics ────────────────────────
    logger.info("=" * 50)
    logger.info("Computing scalar metrics")
    logger.info("=" * 50)
    scalar_metrics = compute_scalar_metrics(
        model, x_test, y_test, num_classes, config.batch_size, config.top_k
    )

    # ── 5. Predictions ───────────────────────────
    logger.info("=" * 50)
    logger.info("Running full inference")
    logger.info("=" * 50)
    y_pred, y_probs = get_predictions(model, x_test, config.batch_size)
    overall_acc = float((y_pred == y_test).mean())
    logger.info("Overall accuracy (argmax): %.4f", overall_acc)

    # ── 6. Per-class report ──────────────────────
    per_class_report(y_test, y_pred, label_map, num_classes, out)

    # ── 7. Confusion matrices ────────────────────
    logger.info("Plotting confusion matrices …")
    plot_confusion_matrix(y_test, y_pred, label_map, num_classes, out,
                          top_n=config.confusion_top_n)

    # ── 8. Most confused pairs ───────────────────
    most_confused_pairs(y_test, y_pred, label_map, num_classes,
                        top_n=20, output_dir=out)

    # ── 9. Per-class accuracy ────────────────────
    plot_per_class_accuracy(y_test, y_pred, label_map, num_classes, out)

    # ── 10. Worst predictions ────────────────────
    plot_worst_samples(x_test, y_test, y_probs, label_map, out,
                       n=config.worst_n_samples)

    # ── 11. Summary JSON ─────────────────────────
    summary = {
        "model_path":     config.model_path,
        "dataset":        config.dataset,
        "num_test_samples": int(len(x_test)),
        "num_classes":    num_classes,
        **scalar_metrics,
        "overall_accuracy_argmax": overall_acc,
    }
    summary_path = out / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("=" * 50)
    logger.info("Evaluation complete. Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-35s %.4f", k, v)
        else:
            logger.info("  %-35s %s", k, v)
    logger.info("All artifacts saved to: %s", out)
    logger.info("=" * 50)

    return summary


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved character-recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",      required=True,
                        help="Path to saved .keras model file")
    parser.add_argument("--dataset",         default="emnist_byclass",
                        choices=list(_DATASET_REGISTRY))
    parser.add_argument("--normalization",   default="minmax",
                        choices=["minmax", "standardize"])
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--output_dir",      default="",
                        help="Where to save eval artifacts (default: model dir/eval/)")
    parser.add_argument("--top_k",           type=int, default=3)
    parser.add_argument("--confusion_top_n", type=int, default=30,
                        help="Max number of classes shown in confusion matrix")
    parser.add_argument("--worst_n_samples", type=int, default=16)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()
    return EvalConfig(**vars(args))


if __name__ == "__main__":
    cfg = _parse_args()
    evaluate(cfg)