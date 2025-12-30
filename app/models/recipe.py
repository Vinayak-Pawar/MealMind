"""
Recipe model and related models for the recipe recommendation system
"""

from sqlalchemy import String, Integer, Float, Text, DateTime, Boolean, ForeignKey, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import List, Optional
import enum

from app.database import Base


class Difficulty(str, enum.Enum):
    """Recipe difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Cuisine(str, enum.Enum):
    """Cuisine types"""
    ITALIAN = "italian"
    MEXICAN = "mexican"
    CHINESE = "chinese"
    INDIAN = "indian"
    FRENCH = "french"
    JAPANESE = "japanese"
    THAI = "thai"
    MEDITERRANEAN = "mediterranean"
    AMERICAN = "american"
    OTHER = "other"


class Recipe(Base):
    """Recipe model"""
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    prep_time: Mapped[int] = mapped_column(Integer, nullable=False)  # in minutes
    cook_time: Mapped[int] = mapped_column(Integer, nullable=False)  # in minutes
    servings: Mapped[int] = mapped_column(Integer, nullable=False)
    difficulty: Mapped[Difficulty] = mapped_column(String(20), default=Difficulty.MEDIUM)
    cuisine: Mapped[Cuisine] = mapped_column(String(20), default=Cuisine.OTHER)

    # AI-generated content
    ai_generated_image_url: Mapped[Optional[str]] = mapped_column(String(500))
    ai_generated_video_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Metadata
    is_ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)

    # Author
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    author: Mapped["User"] = relationship("User", back_populates="recipes")
    ingredients: Mapped[List["RecipeIngredient"]] = relationship("RecipeIngredient", back_populates="recipe", cascade="all, delete-orphan")
    nutrition_info: Mapped["NutritionInfo"] = relationship("NutritionInfo", back_populates="recipe", uselist=False, cascade="all, delete-orphan")
    favorited_by: Mapped[List["User"]] = relationship(
        "User",
        secondary="user_favorite_recipes",
        back_populates="favorite_recipes"
    )
    reviews: Mapped[List["RecipeReview"]] = relationship("RecipeReview", back_populates="recipe", cascade="all, delete-orphan")

    @property
    def total_time(self) -> int:
        """Total time for the recipe (prep + cook)"""
        return self.prep_time + self.cook_time

    @property
    def average_rating(self) -> float:
        """Calculate average rating"""
        if self.rating_count == 0:
            return 0.0
        return self.rating / self.rating_count

    def __repr__(self):
        return f"<Recipe(id={self.id}, title={self.title}, author_id={self.author_id})>"


class RecipeIngredient(Base):
    """Association table for recipe ingredients with quantities"""
    __tablename__ = "recipe_ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recipe_id: Mapped[int] = mapped_column(Integer, ForeignKey("recipes.id"), nullable=False)
    ingredient_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingredients.id"), nullable=False)

    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "cups", "grams", "pieces"

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="ingredients")
    ingredient: Mapped["Ingredient"] = relationship("Ingredient", back_populates="recipes")

    def __repr__(self):
        return f"<RecipeIngredient(recipe_id={self.recipe_id}, ingredient_id={self.ingredient_id}, quantity={self.quantity} {self.unit})>"


class RecipeReview(Base):
    """Recipe reviews and ratings"""
    __tablename__ = "recipe_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recipe_id: Mapped[int] = mapped_column(Integer, ForeignKey("recipes.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5 stars
    review_text: Mapped[Optional[str]] = mapped_column(Text)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="reviews")
    user: Mapped["User"] = relationship("User")

    def __repr__(self):
        return f"<RecipeReview(recipe_id={self.recipe_id}, user_id={self.user_id}, rating={self.rating})>"
