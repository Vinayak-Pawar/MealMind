"""
Ingredients API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional

router = APIRouter()


@router.get("/")
async def get_ingredients(
    category: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200)
):
    """Get ingredients with optional category filter"""
    # Placeholder implementation
    return {
        "ingredients": [
            {"id": 1, "name": "Tomato", "category": "vegetables"},
            {"id": 2, "name": "Chicken", "category": "protein"},
            {"id": 3, "name": "Rice", "category": "grains"}
        ],
        "total": 3
    }


@router.get("/{ingredient_id}")
async def get_ingredient(ingredient_id: int):
    """Get specific ingredient by ID"""
    # Placeholder implementation
    return {"id": ingredient_id, "name": "Sample Ingredient", "category": "other"}
