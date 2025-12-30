"""
Nutrition information model for recipes and meal planning
"""

from sqlalchemy import Integer, Float, ForeignKey, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import Optional

from app.database import Base


class NutritionInfo(Base):
    """Nutrition information for recipes"""
    __tablename__ = "nutrition_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    recipe_id: Mapped[int] = mapped_column(Integer, ForeignKey("recipes.id"), nullable=False, unique=True)

    # Macronutrients (per serving)
    calories: Mapped[float] = mapped_column(Float, nullable=False)
    protein: Mapped[float] = mapped_column(Float, nullable=False)  # in grams
    carbohydrates: Mapped[float] = mapped_column(Float, nullable=False)  # in grams
    fat: Mapped[float] = mapped_column(Float, nullable=False)  # in grams
    fiber: Mapped[float] = mapped_column(Float, nullable=False)  # in grams
    sugar: Mapped[float] = mapped_column(Float, nullable=False)  # in grams

    # Micronutrients
    vitamin_a: Mapped[float] = mapped_column(Float, default=0.0)  # in IU
    vitamin_c: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    vitamin_d: Mapped[float] = mapped_column(Float, default=0.0)  # in IU
    vitamin_e: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    vitamin_k: Mapped[float] = mapped_column(Float, default=0.0)  # in mcg
    vitamin_b12: Mapped[float] = mapped_column(Float, default=0.0)  # in mcg

    calcium: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    iron: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    magnesium: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    phosphorus: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    potassium: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    sodium: Mapped[float] = mapped_column(Float, default=0.0)  # in mg
    zinc: Mapped[float] = mapped_column(Float, default=0.0)  # in mg

    # Fatty acids
    saturated_fat: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    monounsaturated_fat: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    polyunsaturated_fat: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    trans_fat: Mapped[float] = mapped_column(Float, default=0.0)  # in grams
    cholesterol: Mapped[float] = mapped_column(Float, default=0.0)  # in mg

    # Metadata
    serving_size: Mapped[str] = mapped_column(Float, default="1 serving")  # e.g., "100g", "1 cup"
    is_calculated: Mapped[bool] = mapped_column(Float, default=True)  # True if calculated from ingredients
    data_source: Mapped[Optional[str]] = mapped_column(Float)  # API source or manual entry

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="nutrition_info")

    @property
    def glycemic_index(self) -> Optional[float]:
        """Calculate glycemic index based on carbohydrates and fiber"""
        if self.carbohydrates > 0:
            return (self.carbohydrates - self.fiber) / self.carbohydrates * 100
        return None

    @property
    def protein_percentage(self) -> float:
        """Calculate protein percentage of calories"""
        if self.calories > 0:
            return (self.protein * 4) / self.calories * 100
        return 0.0

    @property
    def carb_percentage(self) -> float:
        """Calculate carbohydrate percentage of calories"""
        if self.calories > 0:
            return (self.carbohydrates * 4) / self.calories * 100
        return 0.0

    @property
    def fat_percentage(self) -> float:
        """Calculate fat percentage of calories"""
        if self.calories > 0:
            return (self.fat * 9) / self.calories * 100
        return 0.0

    def get_daily_value_percentage(self, nutrient: str, daily_value: float) -> float:
        """Calculate percentage of daily recommended value"""
        nutrient_value = getattr(self, nutrient.lower(), 0)
        if daily_value > 0:
            return (nutrient_value / daily_value) * 100
        return 0.0

    def __repr__(self):
        return f"<NutritionInfo(recipe_id={self.recipe_id}, calories={self.calories})>"


class ShoppingList(Base):
    """Shopping list model for users"""
    __tablename__ = "shopping_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(Float, default="My Shopping List")
    is_completed: Mapped[bool] = mapped_column(Float, default=False)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="shopping_lists")
    items: Mapped[List["ShoppingListItem"]] = relationship("ShoppingListItem", back_populates="shopping_list", cascade="all, delete-orphan")


class ShoppingListItem(Base):
    """Individual items in a shopping list"""
    __tablename__ = "shopping_list_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    shopping_list_id: Mapped[int] = mapped_column(Integer, ForeignKey("shopping_lists.id"), nullable=False)
    ingredient_id: Mapped[int] = mapped_column(Integer, ForeignKey("ingredients.id"), nullable=False)

    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(Float, nullable=False)
    is_purchased: Mapped[bool] = mapped_column(Float, default=False)

    # Relationships
    shopping_list: Mapped["ShoppingList"] = relationship("ShoppingList", back_populates="items")
    ingredient: Mapped["Ingredient"] = relationship("Ingredient")

    def __repr__(self):
        return f"<ShoppingListItem(ingredient_id={self.ingredient_id}, quantity={self.quantity} {self.unit})>"
