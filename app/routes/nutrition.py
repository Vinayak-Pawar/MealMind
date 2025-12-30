"""
Nutrition API Routes
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List

router = APIRouter()


@router.get("/analyze/{recipe_id}")
async def analyze_recipe_nutrition(recipe_id: int):
    """Analyze nutrition for a recipe"""
    # Placeholder implementation
    return {
        "recipe_id": recipe_id,
        "nutrition": {
            "calories": 450,
            "protein": 25.5,
            "carbohydrates": 35.2,
            "fat": 18.3,
            "fiber": 4.1
        },
        "daily_values": {
            "calories_percent": 22.5,
            "protein_percent": 51.0,
            "carbs_percent": 11.7,
            "fat_percent": 23.5
        }
    }


@router.get("/daily-recommendations")
async def get_daily_nutrition_recommendations(
    age: int = Query(30, ge=1, le=120),
    gender: str = Query("other", pattern="^(male|female|other)$"),
    activity_level: str = Query("moderate", pattern="^(sedentary|light|moderate|active|very_active)$")
):
    """Get daily nutrition recommendations"""
    # Basic recommendations (would be more sophisticated in production)
    base_calories = {
        "male": {25: 2400, 30: 2300, 35: 2200, 40: 2100},
        "female": {25: 2000, 30: 1900, 35: 1800, 40: 1700},
        "other": {25: 2200, 30: 2100, 35: 2000, 40: 1900}
    }

    calories = base_calories.get(gender, base_calories["other"]).get(
        min(base_calories[gender].keys(), key=lambda x: abs(x - age)), 2000
    )

    # Adjust for activity level
    activity_multipliers = {
        "sedentary": 1.0,
        "light": 1.2,
        "moderate": 1.4,
        "active": 1.6,
        "very_active": 1.8
    }

    calories = int(calories * activity_multipliers.get(activity_level, 1.4))

    return {
        "daily_recommendations": {
            "calories": calories,
            "protein_grams": int(calories * 0.15 / 4),  # 15% of calories from protein
            "carbs_grams": int(calories * 0.55 / 4),   # 55% of calories from carbs
            "fat_grams": int(calories * 0.30 / 9),     # 30% of calories from fat
            "fiber_grams": 25 if age < 50 else 38,     # Age-based fiber recommendations
            "sodium_mg": 2300,                          # General sodium limit
            "sugar_grams": int(calories * 0.10 / 4)    # Max 10% from sugar
        },
        "parameters": {
            "age": age,
            "gender": gender,
            "activity_level": activity_level
        }
    }
