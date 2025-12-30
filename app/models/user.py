"""
User model for authentication and user management
"""

from sqlalchemy import String, Boolean, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import List, Optional
import enum

from app.database import Base


class UserRole(str, enum.Enum):
    """User role enumeration"""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"


class DietaryPreference(str, enum.Enum):
    """Dietary preference enumeration"""
    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    KETO = "keto"
    PALEO = "paleo"
    MEDITERRANEAN = "mediterranean"


class User(Base):
    """User model"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[UserRole] = mapped_column(String(20), default=UserRole.USER)

    # Profile information
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(255))

    # Dietary preferences
    dietary_preference: Mapped[DietaryPreference] = mapped_column(String(20), default=DietaryPreference.NONE)
    allergies: Mapped[Optional[str]] = mapped_column(Text)  # JSON string of allergies
    cuisine_preferences: Mapped[Optional[str]] = mapped_column(Text)  # JSON string of preferred cuisines

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    recipes: Mapped[List["Recipe"]] = relationship("Recipe", back_populates="author", cascade="all, delete-orphan")
    favorite_recipes: Mapped[List["Recipe"]] = relationship(
        "Recipe",
        secondary="user_favorite_recipes",
        back_populates="favorited_by"
    )
    shopping_lists: Mapped[List["ShoppingList"]] = relationship("ShoppingList", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class UserFavoriteRecipes(Base):
    """Association table for user favorite recipes"""
    __tablename__ = "user_favorite_recipes"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recipe_id: Mapped[int] = mapped_column(Integer, primary_key=True)
