"""
Recipes API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json

from app.database import get_db
from app.models import Recipe, User, RecipeIngredient, Ingredient
from app.services.recommender import RecipeRecommender
from app.services.video_service import VideoGenerator
from app.services.image_service import RecipeImageGenerator
from app.config import settings


# Pydantic models for API
class RecipeCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    instructions: str = Field(..., min_length=10)
    prep_time: int = Field(..., gt=0, le=1440)  # Max 24 hours
    cook_time: int = Field(..., gt=0, le=1440)
    servings: int = Field(..., gt=0, le=100)
    ingredients: List[Dict[str, Any]] = Field(..., min_items=1)
    cuisine: Optional[str] = None
    difficulty: Optional[str] = None


class RecipeResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    instructions: str
    prep_time: int
    cook_time: int
    servings: int
    cuisine: str
    difficulty: str
    rating: float
    rating_count: int
    author_id: int
    ai_generated_image_url: Optional[str]
    ai_generated_video_url: Optional[str]
    ingredients: List[Dict[str, Any]]
    nutrition_info: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class RecipeRecommendationRequest(BaseModel):
    available_ingredients: List[str] = Field(..., min_items=1)
    dietary_preferences: Optional[List[str]] = None
    cuisine_preferences: Optional[List[str]] = None
    max_prep_time: Optional[int] = None
    difficulty_preference: Optional[str] = None
    servings: Optional[int] = None


router = APIRouter()


@router.post("/recommend", response_model=List[RecipeResponse])
async def get_recipe_recommendations(
    request: RecipeRecommendationRequest,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(10, ge=1, le=50)
):
    """Get recipe recommendations based on available ingredients"""
    try:
        # Get all recipes from database
        result = await db.execute("SELECT * FROM recipes")
        recipes = result.fetchall()

        # Initialize recommender
        recommender = RecipeRecommender()

        # Get user preferences
        user_preferences = {}
        if request.dietary_preferences:
            user_preferences['dietary_preference'] = request.dietary_preferences[0]
        if request.cuisine_preferences:
            user_preferences['preferred_cuisines'] = request.cuisine_preferences
        if request.max_prep_time:
            user_preferences['max_prep_time'] = request.max_prep_time
        if request.difficulty_preference:
            user_preferences['difficulty_preference'] = request.difficulty_preference

        # Get recommendations
        recommendations = recommender.recommend_recipes(
            request.available_ingredients,
            recipes,
            user_preferences,
            top_k=limit
        )

        # Format response
        response_recipes = []
        for recipe, score in recommendations:
            # Get ingredients for this recipe
            ingredients_result = await db.execute(
                "SELECT ri.*, i.name, i.category FROM recipe_ingredients ri "
                "JOIN ingredients i ON ri.ingredient_id = i.id "
                "WHERE ri.recipe_id = :recipe_id",
                {"recipe_id": recipe.id}
            )
            ingredients_data = ingredients_result.fetchall()

            ingredients_list = [
                {
                    'id': ing.ingredient_id,
                    'name': ing.name,
                    'category': ing.category,
                    'quantity': ing.quantity,
                    'unit': ing.unit
                }
                for ing in ingredients_data
            ]

            # Get nutrition info
            nutrition_result = await db.execute(
                "SELECT * FROM nutrition_info WHERE recipe_id = :recipe_id",
                {"recipe_id": recipe.id}
            )
            nutrition_data = nutrition_result.fetchone()

            recipe_dict = {
                'id': recipe.id,
                'title': recipe.title,
                'description': recipe.description,
                'instructions': recipe.instructions,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time,
                'servings': recipe.servings,
                'cuisine': recipe.cuisine,
                'difficulty': recipe.difficulty,
                'rating': recipe.average_rating,
                'rating_count': recipe.rating_count,
                'author_id': recipe.author_id,
                'ai_generated_image_url': recipe.ai_generated_image_url,
                'ai_generated_video_url': recipe.ai_generated_video_url,
                'ingredients': ingredients_list,
                'nutrition_info': {
                    'calories': nutrition_data.calories if nutrition_data else 0,
                    'protein': nutrition_data.protein if nutrition_data else 0,
                    'carbohydrates': nutrition_data.carbohydrates if nutrition_data else 0,
                    'fat': nutrition_data.fat if nutrition_data else 0
                } if nutrition_data else None
            }

            response_recipes.append(recipe_dict)

        return response_recipes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@router.get("/", response_model=List[RecipeResponse])
async def get_recipes(
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    cuisine: Optional[str] = None,
    difficulty: Optional[str] = None,
    max_prep_time: Optional[int] = None
):
    """Get all recipes with optional filtering"""
    try:
        query = "SELECT * FROM recipes WHERE 1=1"
        params = {}

        if cuisine:
            query += " AND cuisine = :cuisine"
            params['cuisine'] = cuisine

        if difficulty:
            query += " AND difficulty = :difficulty"
            params['difficulty'] = difficulty

        if max_prep_time:
            query += " AND prep_time <= :max_prep_time"
            params['max_prep_time'] = max_prep_time

        query += " ORDER BY rating DESC LIMIT :limit OFFSET :skip"
        params['limit'] = limit
        params['skip'] = skip

        result = await db.execute(query, params)
        recipes = result.fetchall()

        response_recipes = []
        for recipe in recipes:
            # Get ingredients
            ingredients_result = await db.execute(
                "SELECT ri.*, i.name, i.category FROM recipe_ingredients ri "
                "JOIN ingredients i ON ri.ingredient_id = i.id "
                "WHERE ri.recipe_id = :recipe_id",
                {"recipe_id": recipe.id}
            )
            ingredients_data = ingredients_result.fetchall()

            ingredients_list = [
                {
                    'id': ing.ingredient_id,
                    'name': ing.name,
                    'category': ing.category,
                    'quantity': ing.quantity,
                    'unit': ing.unit
                }
                for ing in ingredients_data
            ]

            recipe_dict = {
                'id': recipe.id,
                'title': recipe.title,
                'description': recipe.description,
                'instructions': recipe.instructions,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time,
                'servings': recipe.servings,
                'cuisine': recipe.cuisine,
                'difficulty': recipe.difficulty,
                'rating': recipe.average_rating,
                'rating_count': recipe.rating_count,
                'author_id': recipe.author_id,
                'ai_generated_image_url': recipe.ai_generated_image_url,
                'ai_generated_video_url': recipe.ai_generated_video_url,
                'ingredients': ingredients_list,
                'nutrition_info': None  # Could be expanded
            }

            response_recipes.append(recipe_dict)

        return response_recipes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recipes: {str(e)}")


@router.get("/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(recipe_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific recipe by ID"""
    try:
        # Get recipe
        result = await db.execute(
            "SELECT * FROM recipes WHERE id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        recipe = result.fetchone()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Get ingredients
        ingredients_result = await db.execute(
            "SELECT ri.*, i.name, i.category FROM recipe_ingredients ri "
            "JOIN ingredients i ON ri.ingredient_id = i.id "
            "WHERE ri.recipe_id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        ingredients_data = ingredients_result.fetchall()

        ingredients_list = [
            {
                'id': ing.ingredient_id,
                'name': ing.name,
                'category': ing.category,
                'quantity': ing.quantity,
                'unit': ing.unit
            }
            for ing in ingredients_data
        ]

        return {
            'id': recipe.id,
            'title': recipe.title,
            'description': recipe.description,
            'instructions': recipe.instructions,
            'prep_time': recipe.prep_time,
            'cook_time': recipe.cook_time,
            'servings': recipe.servings,
            'cuisine': recipe.cuisine,
            'difficulty': recipe.difficulty,
            'rating': recipe.average_rating,
            'rating_count': recipe.rating_count,
            'author_id': recipe.author_id,
            'ai_generated_image_url': recipe.ai_generated_image_url,
            'ai_generated_video_url': recipe.ai_generated_video_url,
            'ingredients': ingredients_list,
            'nutrition_info': None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recipe: {str(e)}")


@router.post("/generate-video/{recipe_id}")
async def generate_recipe_video(
    recipe_id: int,
    background_tasks: BackgroundTasks,
    style: str = Query("modern", description="Video style"),
    db: AsyncSession = Depends(get_db)
):
    """Generate an AI cooking video for a recipe"""
    try:
        # Get recipe
        result = await db.execute(
            "SELECT * FROM recipes WHERE id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        recipe = result.fetchone()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Get ingredients
        ingredients_result = await db.execute(
            "SELECT i.name FROM recipe_ingredients ri "
            "JOIN ingredients i ON ri.ingredient_id = i.id "
            "WHERE ri.recipe_id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        ingredients = [row.name for row in ingredients_result.fetchall()]

        # Initialize video generator
        video_generator = VideoGenerator()

        # Generate video in background
        background_tasks.add_task(
            video_generator.generate_recipe_video,
            recipe.title,
            recipe.instructions,
            ingredients,
            recipe.total_time,
            recipe.difficulty,
            style
        )

        return {
            "message": "Video generation started",
            "recipe_id": recipe_id,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting video generation: {str(e)}")


@router.post("/generate-image/{recipe_id}")
async def generate_recipe_image(
    recipe_id: int,
    background_tasks: BackgroundTasks,
    style: str = Query("photorealistic", description="Image style"),
    mood: str = Query("appetizing", description="Image mood"),
    db: AsyncSession = Depends(get_db)
):
    """Generate an AI image for a recipe"""
    try:
        # Get recipe
        result = await db.execute(
            "SELECT * FROM recipes WHERE id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        recipe = result.fetchone()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Get ingredients
        ingredients_result = await db.execute(
            "SELECT i.name FROM recipe_ingredients ri "
            "JOIN ingredients i ON ri.ingredient_id = i.id "
            "WHERE ri.recipe_id = :recipe_id",
            {"recipe_id": recipe_id}
        )
        ingredients = [row.name for row in ingredients_result.fetchall()]

        # Initialize image generator
        async with RecipeImageGenerator() as image_generator:
            # Generate image in background
            background_tasks.add_task(
                image_generator.generate_recipe_image,
                recipe.title,
                ingredients,
                style,
                mood
            )

        return {
            "message": "Image generation started",
            "recipe_id": recipe_id,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting image generation: {str(e)}")


@router.get("/search/")
async def search_recipes(
    query: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(20, ge=1, le=100)
):
    """Search recipes by title, ingredients, or description"""
    try:
        # Simple text search (in production, use full-text search)
        search_query = f"%{query}%"

        result = await db.execute(
            """
            SELECT DISTINCT r.* FROM recipes r
            LEFT JOIN recipe_ingredients ri ON r.id = ri.recipe_id
            LEFT JOIN ingredients i ON ri.ingredient_id = i.id
            WHERE r.title ILIKE :query
               OR r.description ILIKE :query
               OR i.name ILIKE :query
            LIMIT :limit
            """,
            {"query": search_query, "limit": limit}
        )

        recipes = result.fetchall()

        # Format response similar to get_recipes
        response_recipes = []
        for recipe in recipes:
            response_recipes.append({
                'id': recipe.id,
                'title': recipe.title,
                'description': recipe.description,
                'cuisine': recipe.cuisine,
                'difficulty': recipe.difficulty,
                'rating': recipe.average_rating,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time
            })

        return response_recipes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching recipes: {str(e)}")
