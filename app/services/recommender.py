"""
Recipe Recommender System - Core ML functionality for MealMind
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import pickle
import os
from loguru import logger
from collections import defaultdict

# Optional imports for advanced features
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available, using basic data structures")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using basic similarity measures")

from app.config import settings
from app.models import Recipe, Ingredient, User, RecipeIngredient


class RecipeRecommender:
    """Advanced recipe recommendation system"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the recommender system"""
        self.model_path = model_path or settings.RECOMMENDER_MODEL_PATH
        self.vectorizer = None
        self.similarity_matrix = None
        self.recipe_features = None
        self.ingredient_encoder = None
        self.scaler = None
        self.is_trained = False

        # Load existing model if available
        self.load_model()

    def load_model(self) -> bool:
        """Load pre-trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.similarity_matrix = model_data['similarity_matrix']
                    self.recipe_features = model_data['recipe_features']
                    self.ingredient_encoder = model_data['ingredient_encoder']
                    self.scaler = model_data['scaler']
                    self.is_trained = True
                logger.info("Recipe recommender model loaded successfully")
                return True
            else:
                logger.info("No pre-trained model found, starting fresh")
                return False
        except Exception as e:
            logger.error(f"Error loading recommender model: {e}")
            return False

    def save_model(self) -> bool:
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'vectorizer': self.vectorizer,
                'similarity_matrix': self.similarity_matrix,
                'recipe_features': self.recipe_features,
                'ingredient_encoder': self.ingredient_encoder,
                'scaler': self.scaler
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Recipe recommender model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving recommender model: {e}")
            return False

    def preprocess_recipes(self, recipes: List[Recipe]) -> List[Dict]:
        """Preprocess recipe data for training"""
        recipe_data = []

        for recipe in recipes:
            # Get ingredient names
            ingredient_names = [ri.ingredient.name for ri in recipe.ingredients]

            # Create feature dictionary
            features = {
                'recipe_id': recipe.id,
                'title': recipe.title,
                'description': recipe.description or '',
                'ingredients': ingredient_names,
                'cuisine': recipe.cuisine.value,
                'difficulty': recipe.difficulty.value,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time,
                'servings': recipe.servings,
                'rating': recipe.average_rating,
                'ingredients_text': ' '.join(ingredient_names).lower(),
                'categories': [ing.category.value for ing in recipe.ingredients]
            }
            recipe_data.append(features)

        # Convert to DataFrame if pandas is available, otherwise return list
        if PANDAS_AVAILABLE:
            return pd.DataFrame(recipe_data)
        return recipe_data

    def train(self, recipes: List[Recipe]) -> bool:
        """Train the recommendation model"""
        try:
            logger.info(f"Training recommender with {len(recipes)} recipes")

            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available, using basic training")
                # Store basic recipe data for simple matching
                self.basic_recipes = self.preprocess_recipes(recipes)
                self.is_trained = True
                return True

            # Preprocess recipes
            df = self.preprocess_recipes(recipes)
            if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
                # Text vectorization for ingredients
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                ingredient_vectors = self.vectorizer.fit_transform(df['ingredients_text'])

                # Encode categorical features
                self.ingredient_encoder = MultiLabelBinarizer()
                category_features = self.ingredient_encoder.fit_transform(df['categories'])

                # Numerical features
                numerical_features = df[['prep_time', 'cook_time', 'servings', 'rating']].values
                self.scaler = StandardScaler()
                numerical_features_scaled = self.scaler.fit_transform(numerical_features)

                # Combine all features
                self.recipe_features = np.hstack([
                    ingredient_vectors.toarray(),
                    category_features,
                    numerical_features_scaled
                ])

                # Calculate similarity matrix
                self.similarity_matrix = cosine_similarity(self.recipe_features)
            else:
                # Basic training without ML libraries
                self.basic_recipes = df

            # Save the trained model
            self.save_model()
            self.is_trained = True

            logger.info("Recipe recommender training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error training recommender: {e}")
            return False

    def get_available_ingredients_vector(self, available_ingredients: List[str]) -> np.ndarray:
        """Convert available ingredients to feature vector"""
        if not self.vectorizer:
            raise ValueError("Model not trained or loaded")

        # Create ingredient text
        ingredients_text = ' '.join(available_ingredients).lower()

        # Vectorize ingredients
        ingredient_vector = self.vectorizer.transform([ingredients_text]).toarray()

        # Create category features (assume common categories for available ingredients)
        # This is a simplified approach - in production, you'd get categories from ingredient database
        category_features = np.zeros((1, len(self.ingredient_encoder.classes_)))

        # Numerical features (use averages)
        numerical_features = np.array([[30, 45, 4, 4.0]])  # prep_time, cook_time, servings, rating
        numerical_features_scaled = self.scaler.transform(numerical_features)

        # Combine features
        available_features = np.hstack([
            ingredient_vector,
            category_features,
            numerical_features_scaled
        ])

        return available_features

    def recommend_recipes(
        self,
        available_ingredients: List[str],
        recipe_database: List[Recipe],
        user_preferences: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[Tuple[Recipe, float]]:
        """Recommend recipes based on available ingredients"""
        if not self.is_trained:
            # Fallback to simple ingredient matching if model not trained
            return self._simple_recommendation(available_ingredients, recipe_database, top_k)

        try:
            # Get feature vector for available ingredients
            available_vector = self.get_available_ingredients_vector(available_ingredients)

            # Calculate similarities
            similarities = cosine_similarity(available_vector, self.recipe_features)[0]

            # Get top similar recipes
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates

            recommendations = []
            for idx in top_indices:
                recipe_id = int(self.recipe_features[idx, 0])  # Assuming recipe_id is first feature

                # Find recipe in database
                recipe = next((r for r in recipe_database if r.id == recipe_id), None)
                if recipe:
                    score = similarities[idx]

                    # Apply user preferences filter
                    if user_preferences:
                        score = self._apply_user_preferences(recipe, score, user_preferences)

                    recommendations.append((recipe, score))

            # Sort by score and return top_k
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"Error in recommendation: {e}")
            return self._simple_recommendation(available_ingredients, recipe_database, top_k)

    def _simple_recommendation(
        self,
        available_ingredients: List[str],
        recipe_database: List[Recipe],
        top_k: int = 10
    ) -> List[Tuple[Recipe, float]]:
        """Fallback simple ingredient matching recommendation"""
        available_set = set(ing.lower() for ing in available_ingredients)
        recommendations = []

        for recipe in recipe_database:
            recipe_ingredients = {ri.ingredient.name.lower() for ri in recipe.ingredients}
            matching_ingredients = available_set.intersection(recipe_ingredients)

            if matching_ingredients:
                # Calculate match score
                match_ratio = len(matching_ingredients) / len(recipe_ingredients)
                missing_ratio = 1 - (len(recipe_ingredients - available_set) / len(recipe_ingredients))

                # Combined score
                score = (match_ratio * 0.7) + (missing_ratio * 0.3)
                recommendations.append((recipe, score))

        # Sort by score and return top_k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def _apply_user_preferences(
        self,
        recipe: Recipe,
        base_score: float,
        user_preferences: Dict
    ) -> float:
        """Apply user preferences to recommendation score"""
        score = base_score

        # Dietary preferences
        if 'dietary_preference' in user_preferences:
            diet_pref = user_preferences['dietary_preference']
            if diet_pref == 'vegetarian' and not self._is_vegetarian(recipe):
                score *= 0.3
            elif diet_pref == 'vegan' and not self._is_vegan(recipe):
                score *= 0.1

        # Cuisine preferences
        if 'preferred_cuisines' in user_preferences:
            preferred_cuisines = set(user_preferences['preferred_cuisines'])
            if recipe.cuisine.value in preferred_cuisines:
                score *= 1.2

        # Difficulty preference
        if 'difficulty_preference' in user_preferences:
            pref_difficulty = user_preferences['difficulty_preference']
            if recipe.difficulty.value == pref_difficulty:
                score *= 1.1

        # Time constraints
        if 'max_prep_time' in user_preferences:
            if recipe.prep_time > user_preferences['max_prep_time']:
                score *= 0.5

        if 'max_cook_time' in user_preferences:
            if recipe.cook_time > user_preferences['max_cook_time']:
                score *= 0.5

        return score

    def _is_vegetarian(self, recipe: Recipe) -> bool:
        """Check if recipe is vegetarian"""
        meat_categories = ['protein']  # Simplified check
        for ri in recipe.ingredients:
            if ri.ingredient.category.value in meat_categories:
                # Check if it's actually meat (not tofu, beans, etc.)
                meat_indicators = ['chicken', 'beef', 'pork', 'fish', 'meat']
                if any(indicator in ri.ingredient.name.lower() for indicator in meat_indicators):
                    return False
        return True

    def _is_vegan(self, recipe: Recipe) -> bool:
        """Check if recipe is vegan"""
        if not self._is_vegetarian(recipe):
            return False

        # Check for dairy and eggs
        animal_products = ['dairy', 'eggs', 'milk', 'cheese', 'butter', 'yogurt']
        for ri in recipe.ingredients:
            ingredient_name = ri.ingredient.name.lower()
            if any(product in ingredient_name for product in animal_products):
                return False
        return True

    def get_recipe_similarity(self, recipe1_id: int, recipe2_id: int) -> float:
        """Get similarity score between two recipes"""
        if not self.is_trained:
            return 0.0

        try:
            idx1 = np.where(self.recipe_features[:, 0] == recipe1_id)[0]
            idx2 = np.where(self.recipe_features[:, 0] == recipe2_id)[0]

            if len(idx1) > 0 and len(idx2) > 0:
                return self.similarity_matrix[idx1[0], idx2[0]]
        except:
            pass

        return 0.0

    def get_ingredient_substitutions(self, ingredient_name: str) -> List[Dict]:
        """Get ingredient substitution suggestions"""
        # This would integrate with the ingredient substitution database
        # For now, return basic substitutions
        substitutions = {
            'butter': ['margarine', 'oil', 'applesauce'],
            'milk': ['almond milk', 'oat milk', 'soy milk'],
            'eggs': ['applesauce', 'flax eggs', 'chia eggs'],
            'flour': ['almond flour', 'coconut flour', 'gluten-free flour']
        }

        return substitutions.get(ingredient_name.lower(), [])
