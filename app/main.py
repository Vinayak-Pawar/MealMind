"""
Main FastAPI application for MealMind - AI-Powered Recipe Recommendation System
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from app.config import settings
from app.database import create_tables
from app.routes import (
    auth,
    recipes,
    ingredients,
    camera,
    video_generation,
    image_generation,
    nutrition
)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("Starting MealMind application...")
    await create_tables()
    logger.info("Database tables created/verified")

    # Load ML models if available
    try:
        # Preload recommender model
        from app.services.recommender import RecipeRecommender
        app.state.recommender = RecipeRecommender()
        logger.info("Recipe recommender model loaded")

        # Preload computer vision model for ingredient detection
        from app.services.camera_service import IngredientDetector
        app.state.ingredient_detector = IngredientDetector()
        logger.info("Ingredient detection model loaded")

    except Exception as e:
        logger.warning(f"Could not load ML models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down MealMind application...")

# Create FastAPI application
app = FastAPI(
    title="MealMind API",
    description="AI-Powered Recipe Recommendation System with Computer Vision",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploads
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(recipes.router, prefix="/api/recipes", tags=["Recipes"])
app.include_router(ingredients.router, prefix="/api/ingredients", tags=["Ingredients"])
app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
app.include_router(video_generation.router, prefix="/api/video", tags=["Video Generation"])
app.include_router(image_generation.router, prefix="/api/image", tags=["Image Generation"])
app.include_router(nutrition.router, prefix="/api/nutrition", tags=["Nutrition"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "recipe_recommendation",
            "ingredient_detection",
            "video_generation",
            "image_generation"
        ]
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to MealMind API",
        "description": "AI-Powered Recipe Recommendation System",
        "docs": "/docs",
        "health": "/health",
        "version": "2.0.0"
    }

def main():
    """Main application entry point for development"""
    logger.info("Starting MealMind development server...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()

