"""
Model loader and management for character recognition.

Handles loading TensorFlow models and maintaining dataset label mappings.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

# Import custom learning rate schedule to register it with Keras
# This ensures models with WarmupCosineDecay can be loaded
try:
    import sys
    # Add parent directory to path to import from training scripts
    backend_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(backend_path))
    from train import WarmupCosineDecay  # noqa: F401
except ImportError:
    logging.warning("Could not import WarmupCosineDecay - models with custom LR schedules may fail to load")


logger = logging.getLogger(__name__)


# Dataset label mappings
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


class ModelLoader:
    """
    Singleton class for loading and caching the character recognition model.
    
    The model is loaded once and reused across requests for efficiency.
    """
    
    _instance: Optional["ModelLoader"] = None
    _model: Optional[tf.keras.Model] = None
    _label_map: Optional[list[str]] = None
    _dataset_name: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(
        self,
        model_path: str | Path,
        dataset_name: str = "emnist_digits",
        force_reload: bool = False,
    ) -> tuple[tf.keras.Model, list[str]]:
        """
        Load the model and label mapping.
        
        Parameters
        ----------
        model_path : str | Path
            Path to the .keras model file
        dataset_name : str
            Name of the dataset the model was trained on (for label mapping)
        force_reload : bool
            If True, reload even if already cached
        
        Returns
        -------
        tuple[tf.keras.Model, list[str]]
            (model, label_list)
        
        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        ValueError
            If dataset name is unknown
        """
        model_path = Path(model_path)
        
        # Return cached model if already loaded and not forcing reload
        if not force_reload and self._model is not None and self._dataset_name == dataset_name:
            logger.info("Using cached model")
            return self._model, self._label_map
        
        # Validate model path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Validate dataset name
        if dataset_name not in LABEL_MAPS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(LABEL_MAPS.keys())}"
            )
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Dataset: {dataset_name}")
        
        # Load model without compiling (we only need forward pass for inference)
        # This avoids issues with custom learning rate schedules
        self._model = tf.keras.models.load_model(str(model_path), compile=False)
        
        # Get label mapping
        self._label_map = LABEL_MAPS[dataset_name]
        self._dataset_name = dataset_name
        
        logger.info(f"Model loaded successfully - {self._model.count_params():,} parameters")
        logger.info(f"Output classes: {len(self._label_map)}")
        
        return self._model, self._label_map
    
    def predict(self, image: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Run prediction on a preprocessed image.
        
        Parameters
        ----------
        image : np.ndarray
            Preprocessed image array with shape (1, 28, 28, 1)
        top_k : int
            Number of top predictions to return
        
        Returns
        -------
        list[dict]
            List of prediction dicts with keys: rank, label, confidence
        
        Raises
        ------
        RuntimeError
            If model hasn't been loaded yet
        """
        if self._model is None or self._label_map is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        probabilities = self._model.predict(image, verbose=0)[0]  # shape: (num_classes,)
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "label": self._label_map[idx],
                "confidence": float(probabilities[idx]),
            })
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns
        -------
        dict
            Model metadata
        """
        if self._model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "dataset": self._dataset_name,
            "num_classes": len(self._label_map),
            "num_parameters": self._model.count_params(),
            "input_shape": str(self._model.input_shape),
            "output_shape": str(self._model.output_shape),
        }


# Global singleton instance
model_loader = ModelLoader()
