"""
Prediction routes - API endpoints for character recognition.
"""

import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.controllers.prediction_controller import prediction_controller


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["Prediction"])


@router.post("/")
@router.post("")
async def predict_character(
    file: UploadFile = File(..., description="Image file containing a handwritten character"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top predictions to return"),
):
    """
    Predict the character from an uploaded image.
    
    **Parameters:**
    - **file**: Image file (PNG, JPG, BMP, etc.) containing a handwritten character
    - **top_k**: Number of top predictions to return (default: 5, max: 20)
    
    **Returns:**
    - **success**: Whether prediction succeeded
    - **predictions**: List of top predictions with labels and confidence scores
    - **top_prediction**: The most likely character
    - **confidence**: Confidence score for the top prediction (0-1)
    
    **Example Response:**
    ```json
    {
        "success": true,
        "predictions": [
            {"rank": 1, "label": "5", "confidence": 0.987},
            {"rank": 2, "label": "3", "confidence": 0.008},
            {"rank": 3, "label": "6", "confidence": 0.003}
        ],
        "top_prediction": "5",
        "confidence": 0.987
    }
    ```
    """
    # Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Process prediction
    result = await prediction_controller.predict_character(image_bytes, top_k=top_k)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return JSONResponse(content=result)


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    **Returns:**
    - **loaded**: Whether a model is loaded
    - **dataset**: Dataset the model was trained on
    - **num_classes**: Number of output classes
    - **num_parameters**: Total model parameters
    - **input_shape**: Model input shape
    - **output_shape**: Model output shape
    
    **Example Response:**
    ```json
    {
        "loaded": true,
        "dataset": "emnist_digits",
        "num_classes": 10,
        "num_parameters": 1234567,
        "input_shape": "(None, 28, 28, 1)",
        "output_shape": "(None, 10)"
    }
    ```
    """
    info = prediction_controller.get_model_info()
    return JSONResponse(content=info)
