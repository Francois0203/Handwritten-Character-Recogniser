"""
FastAPI Server - Character Recognition Backend

Main application entry point that sets up the FastAPI server,
configures middleware, and loads the model.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings, MODEL_PATH
from src.core.model_loader import model_loader
from src.api.routes import predict


# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Loads the model on startup and cleans up on shutdown.
    """
    # Startup: Load the model
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading model...")
        model_loader.load_model(
            model_path=MODEL_PATH,
            dataset_name=settings.dataset_name,
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.warning("Server will start but predictions will fail until model is loaded")
    
    logger.info(f"Server ready at http://{settings.host}:{settings.port}")
    logger.info(f"API documentation: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    REST API for handwritten character recognition using deep learning.
    
    ## Features
    - Upload image files for character recognition
    - Get top-k predictions with confidence scores
    - Support for multiple character datasets (digits, letters, etc.)
    
    ## Usage
    1. Upload an image using the `/api/predict` endpoint
    2. Receive predictions with confidence scores
    3. Check model info at `/api/predict/model-info`
    """,
    lifespan=lifespan,
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(predict.router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - returns API information.
    """
    return JSONResponse(
        content={
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs",
            "endpoints": {
                "predict": "/api/predict",
                "model_info": "/api/predict/model-info",
            },
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint - verifies server and model status.
    """
    model_info = model_loader.get_model_info()
    
    return JSONResponse(
        content={
            "status": "healthy",
            "model_loaded": model_info.get("loaded", False),
            "dataset": model_info.get("dataset", "unknown"),
        }
    )


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
