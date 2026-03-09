"""
Image preprocessing utilities for character recognition.

Handles conversion of uploaded images to the format expected by the model.
"""

import io
import numpy as np
from PIL import Image


def preprocess_image_for_prediction(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes to the format expected by the model.
    
    Steps:
        - Load image from bytes
        - Convert to grayscale
        - Resize to 28x28
        - Invert if background is white (model trained on dark background)
        - Normalize pixels to [0, 1]
        - Reshape to (1, 28, 28, 1) for batch inference
    
    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes from file upload
    
    Returns
    -------
    np.ndarray
        Preprocessed image array with shape (1, 28, 28, 1) and dtype float32
    
    Examples
    --------
    >>> with open("digit.png", "rb") as f:
    ...     img_array = preprocess_image_for_prediction(f.read())
    >>> img_array.shape
    (1, 28, 28, 1)
    """
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # "L" = 8-bit grayscale
    
    # Resize to 28x28 using high-quality resampling
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array
    arr = np.array(img, dtype=np.float32)
    
    # Invert if background is white (EMNIST/MNIST have black background, white character)
    # Most scanned/drawn images have white background, so we invert them
    if arr.mean() > 127:
        arr = 255.0 - arr
    
    # Normalize to [0, 1]
    arr = arr / 255.0
    
    # Reshape to (1, 28, 28, 1) - batch dimension + channel dimension
    arr = arr.reshape(1, 28, 28, 1)
    
    return arr


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    Validate that uploaded bytes represent a valid image.
    
    Parameters
    ----------
    image_bytes : bytes
        Raw bytes from file upload
    
    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
        If valid: (True, "")
        If invalid: (False, "error description")
    """
    if not image_bytes:
        return False, "No image data provided"
    
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        return False, "Image file too large (max 10 MB)"
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Verify it's a valid image by attempting to load it
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
