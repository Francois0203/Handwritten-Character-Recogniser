"""
Data loading module for character recognition neural network training.

Supported datasets:
- EMNIST ByClass: 814,255 samples — digits (0–9), uppercase (A–Z), lowercase (a–z) = 62 classes
- EMNIST ByMerge: 814,255 samples — merged similar characters = 47 classes
- EMNIST Letters: 145,600 samples — uppercase + lowercase letters = 37 classes
- EMNIST Digits:  280,000 samples — digits only = 10 classes
- EMNIST MNIST:   70,000 samples  — digits only (MNIST-compatible) = 10 classes
- MNIST:          70,000 samples  — digits only = 10 classes
- KMNIST:         70,000 samples  — 10 Hiragana characters = 10 classes

All images are 28x28 grayscale, returned as numpy arrays with shape (N, 28, 28, 1).
Labels are integer-encoded.

EMNIST ByClass label mapping:
  0–9   → '0'–'9'
  10–35 → 'A'–'Z'
  36–61 → 'a'–'z'
"""

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import random

# ──────────────────────────────────────────────
# Label maps
# ──────────────────────────────────────────────

# EMNIST ByClass / ByMerge: 62 classes
EMNIST_BYCLASS_LABELS = (
    [str(d) for d in range(10)]          # 0–9  → digits
    + [chr(c) for c in range(65, 91)]    # 10–35 → A–Z
    + [chr(c) for c in range(97, 123)]   # 36–61 → a–z
)

# EMNIST Letters: 37 classes (index 0 = N/A, 1–26 → A/a … Z/z)
EMNIST_LETTERS_LABELS = ['N/A'] + [chr(c) for c in range(65, 91)]

# EMNIST Digits / MNIST / KMNIST: 10 classes
DIGIT_LABELS = [str(d) for d in range(10)]

KMNIST_LABELS = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _tfds_to_numpy(dataset):
    """Convert a supervised tf.data.Dataset to (x, y) numpy arrays."""
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def _load_tfds(name, split_train='train', split_test='test'):
    """Generic loader for tensorflow_datasets supervised datasets."""
    print(f"  Downloading / loading '{name}' - train split ...")
    ds_train = tfds.load(name, split=split_train, as_supervised=True)
    print(f"  Downloading / loading '{name}' - test split ...")
    ds_test  = tfds.load(name, split=split_test,  as_supervised=True)

    x_train, y_train = _tfds_to_numpy(ds_train)
    x_test,  y_test  = _tfds_to_numpy(ds_test)

    return (x_train, y_train), (x_test, y_test)


# ──────────────────────────────────────────────
# Public loaders
# ──────────────────────────────────────────────

def load_emnist_byclass():
    """
    EMNIST ByClass — largest split.
    62 classes: digits 0–9, uppercase A–Z, lowercase a–z.
    ~697,932 train / ~116,323 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [0 .. 61]
    """
    print("Loading EMNIST ByClass ...")
    data = _load_tfds('emnist/byclass')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    print(f"  Classes: {len(EMNIST_BYCLASS_LABELS)} -> {EMNIST_BYCLASS_LABELS}")
    return (x_train, y_train), (x_test, y_test)


def load_emnist_bymerge():
    """
    EMNIST ByMerge — 47 classes (uppercase/lowercase merged for visually similar chars).
    ~697,932 train / ~116,323 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [0 .. 46]
    """
    print("Loading EMNIST ByMerge ...")
    data = _load_tfds('emnist/bymerge')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_emnist_letters():
    """
    EMNIST Letters — 37 classes (letters only, upper/lower merged).
    ~124,800 train / ~20,800 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [1 .. 26]
    """
    print("Loading EMNIST Letters ...")
    data = _load_tfds('emnist/letters')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_emnist_digits():
    """
    EMNIST Digits — 10 classes (digits 0–9, balanced).
    ~240,000 train / ~40,000 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [0 .. 9]
    """
    print("Loading EMNIST Digits ...")
    data = _load_tfds('emnist/digits')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    """
    Classic MNIST — 10 classes (digits 0–9).
    60,000 train / 10,000 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [0 .. 9]
    """
    print("Loading MNIST ...")
    data = _load_tfds('mnist')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_kmnist():
    """
    Kuzushiji-MNIST — 10 Hiragana character classes.
    60,000 train / 10,000 test samples.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32  [0 .. 9]
    """
    print("Loading KMNIST ...")
    data = _load_tfds('kmnist')
    (x_train, y_train), (x_test, y_test) = data
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_all_combined():
    """
    Loads EMNIST ByClass (digits + upper + lower) together with KMNIST,
    giving the widest character coverage from a single call.

    EMNIST ByClass labels are kept as-is (0–61).
    KMNIST labels are offset to 62–71 to avoid collision.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32
    label_map : dict  {int → str}
        Maps every integer label to a human-readable character string.
    """
    print("=" * 50)
    print("Loading ALL combined datasets ...")
    print("=" * 50)

    (ex_tr, ey_tr), (ex_te, ey_te) = load_emnist_byclass()
    (kx_tr, ky_tr), (kx_te, ky_te) = load_kmnist()

    # Offset KMNIST labels so they don't overlap with EMNIST's 0–61
    kmnist_offset = len(EMNIST_BYCLASS_LABELS)  # 62
    ky_tr = ky_tr + kmnist_offset
    ky_te = ky_te + kmnist_offset

    x_train = np.concatenate([ex_tr, kx_tr], axis=0)
    y_train = np.concatenate([ey_tr, ky_tr], axis=0)
    x_test  = np.concatenate([ex_te, kx_te], axis=0)
    y_test  = np.concatenate([ey_te, ky_te], axis=0)

    label_map = {i: ch for i, ch in enumerate(EMNIST_BYCLASS_LABELS)}
    for i, ch in enumerate(KMNIST_LABELS):
        label_map[kmnist_offset + i] = ch

    num_classes = len(label_map)
    print(f"\nCombined dataset ready:")
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    print(f"  Total classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), label_map


def load_emnist_all():
    """
    Loads multiple EMNIST datasets combined (without KMNIST).
    This is useful when KMNIST server is unavailable.
    
    Combines:
    - EMNIST ByClass: digits + upper + lower (62 classes) -> labels 0-61
    - MNIST: classic digits (10 classes) -> labels 62-71
    
    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x shape: (N, 28, 28, 1)  uint8
        y shape: (N,)            int32
    label_map : dict  {int -> str}
        Maps every integer label to a human-readable character string.
    """
    print("=" * 50)
    print("Loading EMNIST All (ByClass + MNIST) ...")
    print("=" * 50)

    # Load EMNIST ByClass (62 classes: 0-9, A-Z, a-z)
    (ex_tr, ey_tr), (ex_te, ey_te) = load_emnist_byclass()
    
    # Load MNIST (10 classes: 0-9)
    (mx_tr, my_tr), (mx_te, my_te) = load_mnist()

    # Offset MNIST labels so they don't overlap with EMNIST's 0-61
    mnist_offset = len(EMNIST_BYCLASS_LABELS)  # 62
    my_tr = my_tr + mnist_offset
    my_te = my_te + mnist_offset

    # Combine datasets
    x_train = np.concatenate([ex_tr, mx_tr], axis=0)
    y_train = np.concatenate([ey_tr, my_tr], axis=0)
    x_test  = np.concatenate([ex_te, mx_te], axis=0)
    y_test  = np.concatenate([ey_te, my_te], axis=0)

    # Build label map
    label_map = {i: ch for i, ch in enumerate(EMNIST_BYCLASS_LABELS)}
    for i in range(10):
        label_map[mnist_offset + i] = f"MNIST_{i}"

    num_classes = len(label_map)
    print(f"\nCombined EMNIST dataset ready:")
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")
    print(f"  Total classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), label_map


# ──────────────────────────────────────────────
# Visualisation utility
# ──────────────────────────────────────────────

def visualize_samples(x, y, label_map=None, num_samples=9, title="Sample Images"):
    """
    Plot a random grid of images with their labels.

    Parameters
    ----------
    x : np.ndarray  shape (N, 28, 28, 1)
    y : np.ndarray  shape (N,)
    label_map : dict {int → str} or None
        If provided, integer labels are converted to character strings.
    num_samples : int
    title : str
    """
    plt.figure(figsize=(6, 6))
    indices = random.sample(range(len(x)), min(num_samples, len(x)))
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[idx].squeeze(), cmap='gray')
        label = label_map[y[idx]] if label_map else y[idx]
        plt.title(f"'{label}'")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option A: load the biggest single dataset (recommended for full char coverage)
    (x_train, y_train), (x_test, y_test) = load_emnist_byclass()
    visualize_samples(x_train, y_train,
                      label_map={i: ch for i, ch in enumerate(EMNIST_BYCLASS_LABELS)},
                      title="EMNIST ByClass samples")

    # ── Option B: load everything combined (EMNIST + KMNIST Hiragana)
    # (x_train, y_train), (x_test, y_test), label_map = load_all_combined()
    # visualize_samples(x_train, y_train, label_map=label_map, title="All combined samples")