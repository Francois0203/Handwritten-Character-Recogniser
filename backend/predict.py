"""
predict.py — Run the trained character recognition model on a single image.

Usage
-----
    python predict.py --image path/to/your/image.png
    python predict.py --image path/to/your/image.png --model runs/my_run/best_model.keras
    python predict.py --image path/to/your/image.png --top_k 5

The script will:
    1. Load your image (any common format: PNG, JPG, BMP, etc.)
    2. Convert it to 28x28 grayscale — matching what the model was trained on
    3. Run the model and print the top predictions with confidence scores
    4. Show a preview of the image so you can see what the model saw
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Label maps — edit this to match whichever dataset you trained on
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAPS = {
    "emnist_byclass": (
        [str(d) for d in range(10)]           # 0–9  → digits
        + [chr(c) for c in range(65, 91)]     # 10–35 → A–Z
        + [chr(c) for c in range(97, 123)]    # 36–61 → a–z
    ),
    "emnist_digits":  [str(d) for d in range(10)],
    "mnist":          [str(d) for d in range(10)],
    "kmnist":         ["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"],
}

# ─── CHANGE THIS to match your training dataset ───────────────────────────────
DATASET = "emnist_byclass"
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prepare_image(image_path: str) -> np.ndarray:
    """
    Load any image file and convert it to the 28x28 grayscale format
    the model expects.

    Steps:
        - Open image with Pillow (supports PNG, JPG, BMP, TIFF, etc.)
        - Convert to grayscale
        - Resize to 28x28
        - Invert if background is white (model trained on dark background)
        - Normalize pixels to [0, 1]
        - Reshape to (1, 28, 28, 1)  ← batch of one

    Parameters
    ----------
    image_path : str  path to the input image file

    Returns
    -------
    np.ndarray  shape (1, 28, 28, 1)  float32
    """
    img = Image.open(image_path).convert("L")   # "L" = 8-bit grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    # EMNIST / MNIST images have a BLACK background with WHITE characters.
    # If your image has a white background (typical photo / scan), invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0                           # normalize to [0, 1]
    arr = arr.reshape(1, 28, 28, 1)             # add batch + channel dimensions
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    model_path: str,
    image_path: str,
    top_k: int = 5,
    show_image: bool = True,
) -> list[dict]:
    """
    Load the model, run inference, and return the top-k predictions.

    Parameters
    ----------
    model_path  : path to the saved .keras model file
    image_path  : path to the input image
    top_k       : how many top predictions to return
    show_image  : whether to display the preprocessed image

    Returns
    -------
    list of dicts, e.g.:
        [{"rank": 1, "label": "A", "confidence": 0.97}, ...]
    """
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from:  {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.\n")

    # ── Prepare image ─────────────────────────────────────────────────────────
    image = load_and_prepare_image(image_path)

    # ── Run inference ─────────────────────────────────────────────────────────
    probabilities = model.predict(image, verbose=0)[0]   # shape: (num_classes,)

    # ── Get top-k results ─────────────────────────────────────────────────────
    label_list   = LABEL_MAPS.get(DATASET)
    top_indices  = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        label = label_list[idx] if label_list else str(idx)
        results.append({
            "rank":       rank,
            "label":      label,
            "class_index": int(idx),
            "confidence": float(probabilities[idx]),
        })

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"Image:   {image_path}")
    print(f"Dataset: {DATASET}")
    print("─" * 35)
    print(f"{'Rank':<6} {'Label':<10} {'Confidence':>10}")
    print("─" * 35)
    for r in results:
        bar = "█" * int(r["confidence"] * 20)
        print(f"  #{r['rank']:<4} '{r['label']}'  {r['confidence']:>8.1%}  {bar}")
    print("─" * 35)
    print(f"\nTop prediction:  '{results[0]['label']}'  ({results[0]['confidence']:.1%} confident)\n")

    # ── Show image ────────────────────────────────────────────────────────────
    if show_image:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(
            f"Prediction: '{results[0]['label']}' ({results[0]['confidence']:.1%})",
            fontsize=12,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a character from an image using the trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image file (PNG, JPG, BMP, etc.)",
    )
    parser.add_argument(
        "--model",
        default="runs/default/best_model.keras",
        dest="model_path",
        help="Path to the saved .keras model file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip displaying the image preview",
    )

    args = parser.parse_args()

    predict(
        model_path=args.model_path,
        image_path=args.image,
        top_k=args.top_k,
        show_image=not args.no_preview,
    )