"""
Ingredient model for the recipe recommendation system
"""

from sqlalchemy import String, Integer, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import List, Optional
import enum

from app.database import Base


class IngredientCategory(str, enum.Enum):
    """Ingredient categories"""
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    GRAINS = "grains"
    PROTEIN = "protein"
    DAIRY = "dairy"
    SPICES = "spices"
    OILS = "oils"
    CONDIMENTS = "condiments"
    BAKING = "baking"
    BEVERAGES = "beverages"
    OTHER = "other"


class StorageType(str, enum.Enum):
    """How ingredients should be stored"""
    REFRIGERATOR = "refrigerator"
    FREEZER = "freezer"
    PANTRY = "pantry"
    COUNTER = "counter"


class Ingredient(Base):
    """Ingredient model"""
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[IngredientCategory] = mapped_column(String(20), default=IngredientCategory.OTHER)

    # Basic information
    description: Mapped[Optional[str]] = mapped_column(Text)
    scientific_name: Mapped[Optional[str]] = mapped_column(String(255))

    # Nutritional information (per 100g)
    calories: Mapped[float] = mapped_column(Float, default=0.0)
    protein: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    carbohydrates: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    fat: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    fiber: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    sugar: Mapped[float] = mapped_column(Float, default=0.0)  # in grams

    # Storage information
    storage_type: Mapped[StorageType] = mapped_column(String(20), default=StorageType.PANTRY)
    shelf_life_days: Mapped[Optional[int]] = mapped_column(Integer)  # days at room temperature

    # Detection information (for computer vision)
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    image_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Metadata
    is_common: Mapped[bool] = mapped_column(Boolean, default=True)
    is_seasonal: Mapped[bool] = mapped_column(Boolean, default=False)
    season_start: Mapped[Optional[int]] = mapped_column(Integer)  # month (1-12)
    season_end: Mapped[Optional[int]] = mapped_column(Integer)  # month (1-12)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    recipes: Mapped[List["RecipeIngredient"]] = relationship("RecipeIngredient", back_populates="ingredient")
    substitutions: Mapped[List["IngredientSubstitution"]] = relationship(
        "IngredientSubstitution",
        foreign_keys="IngredientSubstitution.original_ingredient_id",
        back_populates="original_ingredient"
    )
    can_substitute: Mapped[List["IngredientSubstitution"]] = relationship(
        "IngredientSubstitution",
        foreign_keys="IngredientSubstitution.substitute_ingredient_id",
        back_populates="substitute_ingredient"
    )

    @property
    def nutritional_density(self) -> float:
        """Calculate nutritional density score"""
        # Simple nutritional density calculation
        return (self.protein + self.fiber) / max(self.calories, 1) * 100

    def __repr__(self):
        return f"<Ingredient(id={self.id}, name={self.name}, category={self.category})>"


class IngredientSubstitution(Base):
    """Ingredient substitution relationships"""
    __tablename__ = "ingredient_substitutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    original_ingredient_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingredients.id"), nullable=False)
    substitute_ingredient_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingredients.id"), nullable=False)

    substitution_ratio: Mapped[float] = mapped_column(Float, default=1.0)  # how much substitute to use
    reason: Mapped[Optional[str]] = mapped_column(Text)  # why this substitution works
    is_ai_suggested: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    original_ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient",
        foreign_keys=[original_ingredient_id],
        back_populates="substitutions"
    )
    substitute_ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient",
        foreign_keys=[substitute_ingredient_id],
        back_populates="can_substitute"
    )

    def __repr__(self):
        return f"<IngredientSubstitution(original={self.original_ingredient_id}, substitute={self.substitute_ingredient_id})>"
