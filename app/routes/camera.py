"""
Camera API Routes - Ingredient Detection
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image

from app.services.camera_service import IngredientDetector
from app.config import settings


router = APIRouter()


@router.post("/detect-ingredients", response_model=List[Dict[str, Any]])
async def detect_ingredients_from_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Detect ingredients from uploaded image

    - **file**: Image file (JPEG, PNG, WebP)
    - **confidence_threshold**: Minimum confidence score (0.0-1.0)
    """
    try:
        # Validate file type
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
            )

        # Validate file size
        file_content = await file.read()
        if len(file_content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes"
            )

        # Initialize detector
        detector = IngredientDetector()

        # Detect ingredients
        detections = detector.detect_ingredients_from_image(
            file_content,
            confidence_threshold
        )

        if not detections:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No ingredients detected",
                    "detections": [],
                    "suggestions": [
                        "Try taking a clearer photo",
                        "Ensure good lighting",
                        "Focus on individual ingredients",
                        "Remove background clutter"
                    ]
                }
            )

        return detections

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting ingredients: {str(e)}")


@router.post("/detect-ingredients-base64", response_model=List[Dict[str, Any]])
async def detect_ingredients_from_base64(
    image_data: Dict[str, str],
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Detect ingredients from base64 encoded image

    - **image_data**: Dictionary with 'image' key containing base64 string
    - **confidence_threshold**: Minimum confidence score (0.0-1.0)
    """
    try:
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")

        base64_string = image_data['image']

        # Initialize detector
        detector = IngredientDetector()

        # Detect ingredients
        detections = detector.detect_ingredients_from_base64(
            base64_string,
            confidence_threshold
        )

        if not detections:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No ingredients detected",
                    "detections": [],
                    "suggestions": [
                        "Try taking a clearer photo",
                        "Ensure good lighting",
                        "Focus on individual ingredients",
                        "Remove background clutter"
                    ]
                }
            )

        return detections

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting ingredients: {str(e)}")


@router.get("/requirements")
async def get_camera_requirements():
    """Get camera and detection requirements"""
    detector = IngredientDetector()

    return {
        "camera_requirements": detector.get_camera_requirements(),
        "supported_formats": settings.ALLOWED_IMAGE_TYPES,
        "max_file_size": settings.MAX_UPLOAD_SIZE,
        "recommended_settings": {
            "resolution": "1920x1080 or higher",
            "lighting": "Well lit, natural light preferred",
            "angle": "45-degree angle, avoid extreme angles",
            "distance": "6-12 inches from subject",
            "background": "Plain, uncluttered background"
        },
        "best_practices": [
            "Take photos of individual ingredients when possible",
            "Ensure ingredients are clearly visible",
            "Avoid shadows and glare",
            "Clean camera lens before use",
            "Hold camera steady to avoid blur"
        ]
    }


@router.post("/analyze-image")
async def analyze_image_quality(
    file: UploadFile = File(...)
):
    """
    Analyze image quality for ingredient detection

    Returns suggestions for better image capture
    """
    try:
        # Validate file type
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_content = await file.read()

        # Basic image analysis
        try:
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size

            analysis = {
                "dimensions": f"{width}x{height}",
                "format": image.format,
                "mode": image.mode,
                "file_size_mb": len(file_content) / (1024 * 1024)
            }

        except Exception:
            analysis = {
                "error": "Could not analyze image format",
                "file_size_mb": len(file_content) / (1024 * 1024)
            }

        # Quality assessment
        quality_score = 0
        suggestions = []

        if len(file_content) < 1024 * 1024:  # Less than 1MB
            quality_score += 20
        else:
            suggestions.append("File size is large, consider compressing")

        # Basic suggestions
        suggestions.extend([
            "Ensure good lighting for better detection",
            "Focus camera on ingredients clearly",
            "Avoid blurry or shaky images",
            "Try different angles if detection fails"
        ])

        return {
            "quality_score": quality_score,
            "analysis": analysis,
            "suggestions": suggestions,
            "recommendations": [
                "Use natural lighting when possible",
                "Keep camera steady",
                "Focus on one ingredient at a time",
                "Ensure ingredients are well-separated"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@router.get("/supported-ingredients")
async def get_supported_ingredients():
    """Get list of ingredients the system can detect"""
    detector = IngredientDetector()

    # Get class names if model is loaded
    if detector.class_names:
        ingredients_by_category = {}
        for ingredient in detector.class_names:
            category = detector._categorize_ingredient(ingredient)
            if category not in ingredients_by_category:
                ingredients_by_category[category] = []
            ingredients_by_category[category].append(ingredient)

        return {
            "total_ingredients": len(detector.class_names),
            "categories": ingredients_by_category,
            "detection_method": "AI Model" if detector.is_loaded else "Color Analysis",
            "accuracy_note": "Detection accuracy varies by lighting and image quality"
        }
    else:
        # Fallback categories for basic detection
        return {
            "total_ingredients": 0,
            "categories": {
                "fruits": ["apple", "banana", "orange", "tomato"],
                "vegetables": ["lettuce", "carrot", "potato", "onion"],
                "proteins": ["chicken", "beef", "fish"],
                "dairy": ["milk", "cheese"]
            },
            "detection_method": "Basic Color Analysis",
            "accuracy_note": "Limited detection capabilities without AI model"
        }


@router.post("/batch-detect")
async def batch_detect_ingredients(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Detect ingredients from multiple images

    - **files**: Multiple image files
    - **confidence_threshold**: Minimum confidence score
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

        detector = IngredientDetector()
        all_detections = []

        for file in files:
            # Validate file
            if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
                continue

            file_content = await file.read()

            if len(file_content) > settings.MAX_UPLOAD_SIZE:
                continue

            # Detect ingredients
            detections = detector.detect_ingredients_from_image(
                file_content,
                confidence_threshold
            )

            all_detections.append({
                "filename": file.filename,
                "detections": detections,
                "detection_count": len(detections)
            })

        # Aggregate results
        total_detections = sum(item["detection_count"] for item in all_detections)
        unique_ingredients = set()

        for item in all_detections:
            for detection in item["detections"]:
                unique_ingredients.add(detection["name"])

        return {
            "batch_summary": {
                "total_images": len(files),
                "successful_detections": len([d for d in all_detections if d["detection_count"] > 0]),
                "total_detections": total_detections,
                "unique_ingredients": len(unique_ingredients),
                "unique_ingredient_list": list(unique_ingredients)
            },
            "results": all_detections
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch detection: {str(e)}")
