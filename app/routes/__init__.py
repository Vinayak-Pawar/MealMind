"""
API Routes for MealMind
"""

from .auth import router as auth_router
from .recipes import router as recipes_router
from .ingredients import router as ingredients_router
from .camera import router as camera_router
from .video_generation import router as video_router
from .image_generation import router as image_router
from .nutrition import router as nutrition_router

__all__ = [
    'auth_router',
    'recipes_router',
    'ingredients_router',
    'camera_router',
    'video_router',
    'image_router',
    'nutrition_router'
]