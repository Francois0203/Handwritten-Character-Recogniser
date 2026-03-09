"""
Configuration management for the backend API.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Character Recognition API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False  # Auto-reload on code changes (dev mode)
    
    # CORS Settings
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]
    
    # Model Settings
    model_path: str = "models/emnist_digits_standard_v1.keras"
    dataset_name: str = "emnist_digits"
    
    # File Upload Settings
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Resolve model path relative to backend directory
BACKEND_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BACKEND_DIR / settings.model_path
