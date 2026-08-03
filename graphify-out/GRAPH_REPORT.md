# Graph Report - .  (2026-08-01)

## Corpus Check
- Corpus is ~10,248 words - fits in a single context window. You may not need a graph.

## Summary
- 350 nodes · 550 edges · 27 communities (24 shown, 3 thin omitted)
- Extraction: 87% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 67 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Project Overview & Architecture
- Database Models & Schema
- Camera Ingredient Detection
- Recipe Model & ML Recommender
- AI Video Generation Service
- Recipe API Routes
- Nano-Banana Image Pipeline
- Recommender Core & Schemas
- Image Generator Lifecycle
- Nutrition Calculations
- Authentication Routes
- Database Connection Management
- FastAPI Application Entry
- Ingredient & User Entities
- Image Generation API Routes
- App Startup & Table Init
- Nutrition API Routes
- Config & Camera Modules
- Ingredients API Routes
- Video Generation API Routes
- AI Services Package
- Application Settings
- Global Error Handling
- Async DB Session Dependency
- App Package Init
- Utilities Package
- Test Suite Package

## God Nodes (most connected - your core abstractions)
1. `RecipeRecommender` - 24 edges
2. `Base` - 23 edges
3. `IngredientDetector` - 22 edges
4. `RecipeImageGenerator` - 22 edges
5. `VideoGenerator` - 19 edges
6. `Recipe` - 16 edges
7. `NutritionInfo` - 11 edges
8. `MealMind` - 9 edges
9. `Layered Backend Architecture (routes / services / models / schemas)` - 9 edges
10. `Ingredient` - 8 edges

## Surprising Connections (you probably didn't know these)
- `MkDocs Documentation Site` --semantically_similar_to--> `FastAPI`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `MkDocs Documentation Site` --references--> `MealMind`  [INFERRED]
  requirements.txt → README.md
- `HuggingFace Transformers` --conceptually_related_to--> `Smart Recipe Recommendations`  [AMBIGUOUS]
  requirements.txt → README.md
- `scikit-learn / pandas / numpy` --implements--> `ML Recommender System`  [INFERRED]
  requirements.txt → README.md
- `JWT Authentication & Password Hashing Stack` --implements--> `Layered Backend Architecture (routes / services / models / schemas)`  [INFERRED]
  requirements.txt → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Backend service layer: all business-logic modules behind the route layer** — readme_recommender_service, readme_camera_service, readme_video_service, readme_image_service, readme_nutrition_service, readme_layered_backend_architecture [EXTRACTED 1.00]
- **Camera-to-recipe flow: capture, detect, recommend, shop** — readme_ingredient_detector_component, readme_camera_service, readme_computer_vision, readme_ingredient_detection, readme_recommender_system, readme_shopping_list_generation [INFERRED 0.85]
- **Generative media pipeline: external AI providers, async clients, queue and post-processing** — readme_google_flow_api, readme_nano_banana_api, requirements_async_http_clients, requirements_celery_background_tasks, requirements_moviepy_video_processing, requirements_pillow_imageio [INFERRED 0.85]

## Communities (27 total, 3 thin omitted)

### Community 0 - "Project Overview & Architecture"
Cohesion: 0.06
Nodes (51): Alembic Migrations, Apache-2.0 License, Axios HTTP Client, camera_service (Camera / Computer Vision service), Computer Vision Subsystem, data/ (recipes, ingredients, nutrition datasets), Dietary Management, Docker Deployment (split backend/frontend images) (+43 more)

### Community 1 - "Database Models & Schema"
Cohesion: 0.08
Nodes (34): Base, Base class for all database models, IngredientCategory, IngredientSubstitution, str, Ingredient model for the recipe recommendation system, Ingredient categories, How ingredients should be stored (+26 more)

### Community 2 - "Camera Ingredient Detection"
Cohesion: 0.07
Nodes (28): analyze_image_quality(), batch_detect_ingredients(), detect_ingredients_from_base64(), detect_ingredients_from_image(), get_camera_requirements(), get_supported_ingredients(), Get camera and detection requirements, Analyze image quality for ingredient detection      Returns suggestions for bett (+20 more)

### Community 3 - "Recipe Model & ML Recommender"
Cohesion: 0.10
Nodes (13): Total time for the recipe (prep + cook), Calculate average rating, Recipe, ndarray, Train the recommendation model, Convert available ingredients to feature vector, Recommend recipes based on available ingredients, Fallback simple ingredient matching recommendation (+5 more)

### Community 4 - "AI Video Generation Service"
Cohesion: 0.13
Nodes (13): Any, Create a structured video generation request, Generate video script using AI, Generate a template-based video script, Create video from generated script, Get the status of a video generation job, Get list of supported video styles, Get video generation requirements and capabilities (+5 more)

### Community 5 - "Recipe API Routes"
Cohesion: 0.18
Nodes (15): generate_recipe_image(), generate_recipe_video(), get_recipe(), get_recipe_recommendations(), get_recipes(), AsyncSession, Get all recipes with optional filtering, Get a specific recipe by ID (+7 more)

### Community 6 - "Nano-Banana Image Pipeline"
Cohesion: 0.14
Nodes (8): Any, Create a detailed prompt for image generation, Generate image using Nano-Banana API, Save the generated image to disk, Generate variations of an existing recipe image, Get image generation requirements and capabilities, Get the status of an image generation job, Generate a recipe image using Nano-Banana AI          Args:             recipe_t

### Community 7 - "Recommender Core & Schemas"
Cohesion: 0.18
Nodes (10): Config, BaseModel, RecipeCreate, RecipeResponse, Advanced recipe recommendation system, Get similarity score between two recipes, Get ingredient substitution suggestions, Initialize the recommender system (+2 more)

### Community 8 - "Image Generator Lifecycle"
Cohesion: 0.14
Nodes (8): Service for generating AI-powered recipe images using Nano-Banana, Initialize the image generator, Get list of supported image styles, Get list of supported image moods, Async context manager entry, Validate Nano-Banana API connection, Async context manager exit, RecipeImageGenerator

### Community 9 - "Nutrition Calculations"
Cohesion: 0.15
Nodes (7): NutritionInfo, Nutrition information for recipes, Calculate glycemic index based on carbohydrates and fiber, Calculate protein percentage of calories, Calculate carbohydrate percentage of calories, Calculate fat percentage of calories, Calculate percentage of daily recommended value

### Community 10 - "Authentication Routes"
Cohesion: 0.31
Nodes (7): login(), LoginRequest, BaseModel, Authentication API Routes, User registration endpoint, register(), RegisterRequest

### Community 11 - "Database Connection Management"
Cohesion: 0.25
Nodes (7): check_database_connection(), drop_tables(), get_sync_db(), Database configuration and connection management, Check if database connection is working, Get synchronous database session, Drop all database tables (for testing/cleanup)

### Community 12 - "FastAPI Application Entry"
Cohesion: 0.25
Nodes (7): health_check(), main(), Main FastAPI application for MealMind - AI-Powered Recipe Recommendation System, Health check endpoint, Root endpoint with API information, Main application entry point for development, root()

### Community 13 - "Ingredient & User Entities"
Cohesion: 0.25
Nodes (4): Ingredient, Calculate nutritional density score, User, Recipe Recommender System - Core ML functionality for MealMind

### Community 14 - "Image Generation API Routes"
Cohesion: 0.25
Nodes (7): get_image_generation_status(), get_supported_moods(), get_supported_styles(), Image Generation API Routes, Get image generation status, Get supported image styles, Get supported image moods

### Community 15 - "App Startup & Table Init"
Cohesion: 0.29
Nodes (7): create_tables(), init_database(), Create all database tables, Initialize database with default data, lifespan(), Handle application startup and shutdown events, FastAPI

### Community 16 - "Nutrition API Routes"
Cohesion: 0.29
Nodes (5): API Routes for MealMind, analyze_recipe_nutrition(), get_daily_nutrition_recommendations(), Analyze nutrition for a recipe, Get daily nutrition recommendations

### Community 17 - "Config & Camera Modules"
Cohesion: 0.40
Nodes (3): Configuration settings for MealMind application, Camera API Routes - Ingredient Detection, Camera Service - Ingredient Detection using Computer Vision

### Community 18 - "Ingredients API Routes"
Cohesion: 0.33
Nodes (5): get_ingredient(), get_ingredients(), Ingredients API Routes, Get ingredients with optional category filter, Get specific ingredient by ID

### Community 19 - "Video Generation API Routes"
Cohesion: 0.33
Nodes (5): get_supported_styles(), get_video_generation_status(), Video Generation API Routes, Get video generation status, Get supported video styles

### Community 20 - "AI Services Package"
Cohesion: 0.33
Nodes (3): Image Generation Service - AI-powered recipe images using Nano-Banana, Services for MealMind, Video Generation Service - AI-powered cooking tutorial videos using Google Flow

### Community 21 - "Application Settings"
Cohesion: 0.40
Nodes (5): Config, Application settings loaded from environment variables, Pydantic configuration, Settings, BaseSettings

### Community 22 - "Global Error Handling"
Cohesion: 0.50
Nodes (4): global_exception_handler(), Global exception handler for unhandled errors, Exception, Request

### Community 23 - "Async DB Session Dependency"
Cohesion: 0.67
Nodes (3): get_db(), AsyncSession, Dependency for getting async database session

## Ambiguous Edges - Review These
- `Smart Recipe Recommendations` → `HuggingFace Transformers`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to
- `Ingredient Detection (camera-based)` → `data/ (recipes, ingredients, nutrition datasets)`  [AMBIGUOUS]
  README.md · relation: shares_data_with

## Knowledge Gaps
- **12 isolated node(s):** `Dietary Management`, `Shopping List Generation`, `recommender service module`, `Uvicorn / Gunicorn ASGI serving`, `PostgreSQL` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Smart Recipe Recommendations` and `HuggingFace Transformers`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Ingredient Detection (camera-based)` and `data/ (recipes, ingredients, nutrition datasets)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `IngredientDetector` connect `Camera Ingredient Detection` to `Config & Camera Modules`, `FastAPI Application Entry`, `AI Services Package`, `App Startup & Table Init`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `RecipeImageGenerator` connect `Image Generator Lifecycle` to `AI Services Package`, `Recipe API Routes`, `Nano-Banana Image Pipeline`, `Recommender Core & Schemas`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `RecipeRecommender` connect `Recommender Core & Schemas` to `Recipe Model & ML Recommender`, `Recipe API Routes`, `FastAPI Application Entry`, `Ingredient & User Entities`, `App Startup & Table Init`, `AI Services Package`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `RecipeRecommender` (e.g. with `Config` and `RecipeCreate`) actually correct?**
  _`RecipeRecommender` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `Base` (e.g. with `Ingredient` and `NutritionInfo`) actually correct?**
  _`Base` has 16 INFERRED edges - model-reasoned connections that need verification._