"""
Configuration settings for MealMind application
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application settings
    APP_NAME: str = "MealMind"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=True, env="DEBUG")

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    # Database settings
    DATABASE_URL: str = Field(
        default="sqlite:///./mealmind.db",
        env="DATABASE_URL"
    )

    # Security settings
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )

    # API Keys for external services
    GOOGLE_FLOW_API_KEY: str = Field(default="", env="GOOGLE_FLOW_API_KEY")
    NANO_BANANA_API_KEY: str = Field(default="", env="NANO_BANANA_API_KEY")
    NUTRITIONIX_APP_ID: str = Field(default="", env="NUTRITIONIX_APP_ID")
    NUTRITIONIX_APP_KEY: str = Field(default="", env="NUTRITIONIX_APP_KEY")
    SPOONACULAR_API_KEY: str = Field(default="", env="SPOONACULAR_API_KEY")

    # ML Model settings
    MODEL_PATH: str = Field(default="./ml_models", env="MODEL_PATH")
    INGREDIENT_MODEL_PATH: str = Field(
        default="./ml_models/ingredient_detection/model.pb",
        env="INGREDIENT_MODEL_PATH"
    )
    RECOMMENDER_MODEL_PATH: str = Field(
        default="./ml_models/recommender_system/model.pkl",
        env="RECOMMENDER_MODEL_PATH"
    )

    # File upload settings
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/webm", "video/avi"]

    # Redis settings (for caching and background tasks)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # Email settings (for notifications)
    SMTP_SERVER: str = Field(default="", env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: str = Field(default="", env="SMTP_USERNAME")
    SMTP_PASSWORD: str = Field(default="", env="SMTP_PASSWORD")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/mealmind.log", env="LOG_FILE")

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
os.makedirs(settings.MODEL_PATH, exist_ok=True)
