"""
Database configuration and connection management
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import create_engine
from typing import AsyncGenerator
import os
from loguru import logger

from app.config import settings


# SQLAlchemy Base class for models
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# Database engines
# For synchronous operations (migrations, etc.)
sync_engine = create_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://"),
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# For asynchronous operations (FastAPI)
async_engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://").replace(
        "sqlite:///", "sqlite+aiosqlite:///"
    ),
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Session factories
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """Get synchronous database session"""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


async def create_tables():
    """Create all database tables"""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def drop_tables():
    """Drop all database tables (for testing/cleanup)"""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


async def init_database():
    """Initialize database with default data"""
    from app.models.user import User
    from app.models.recipe import Recipe
    from app.models.ingredient import Ingredient
    from app.models.nutrition import NutritionInfo

    # Create tables
    await create_tables()

    # Add default data if needed
    # This would include default ingredients, basic recipes, etc.
    logger.info("Database initialized with default data")


# Database health check
async def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
