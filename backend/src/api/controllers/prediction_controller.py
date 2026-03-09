"""
Prediction controller - handles business logic for character recognition.
"""

import logging
from typing import Optional

from src.core.model_loader import model_loader
from src.utils.image_preprocessing import preprocess_image_for_prediction, validate_image


logger = logging.getLogger(__name__)


class PredictionController:
    """Controller for handling prediction requests."""
    
    @staticmethod
    async def predict_character(
        image_bytes: bytes,
        top_k: int = 5,
    ) -> dict:
        """
        Predict character from uploaded image.
        
        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes from upload
        top_k : int
            Number of top predictions to return
        
        Returns
        -------
        dict
            Prediction results with structure:
            {
                "success": bool,
                "predictions": list[dict],  # if success
                "confidence": float,  # top prediction confidence
                "error": str,  # if not success
            }
        """
        # Validate image
        is_valid, error_msg = validate_image(image_bytes)
        if not is_valid:
            logger.warning(f"Invalid image: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
            }
        
        try:
            # Preprocess image
            image_array = preprocess_image_for_prediction(image_bytes)
            
            # Run prediction
            predictions = model_loader.predict(image_array, top_k=top_k)
            
            logger.info(
                f"Prediction successful - Top result: '{predictions[0]['label']}' "
                f"({predictions[0]['confidence']:.2%})"
            )
            
            return {
                "success": True,
                "predictions": predictions,
                "top_prediction": predictions[0]["label"],
                "confidence": predictions[0]["confidence"],
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
            }
    
    @staticmethod
    def get_model_info() -> dict:
        """
        Get information about the loaded model.
        
        Returns
        -------
        dict
            Model metadata
        """
        return model_loader.get_model_info()


# Singleton instance
prediction_controller = PredictionController()
