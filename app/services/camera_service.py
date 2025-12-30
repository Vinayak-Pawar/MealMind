"""
Camera Service - Ingredient Detection using Computer Vision
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, camera detection will use fallback methods")

from app.config import settings


class IngredientDetector:
    """Computer vision service for ingredient detection"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ingredient detector"""
        self.model_path = model_path or settings.INGREDIENT_MODEL_PATH
        self.model = None
        self.class_names = []
        self.is_loaded = False

        # Load the model
        self.load_model()

        # Initialize OpenCV Haar cascades for basic object detection
        self.cascade_paths = {
            'face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'eye': cv2.data.haarcascades + 'haarcascade_eye.xml'
        }

    def load_model(self) -> bool:
        """Load the computer vision model"""
        try:
            if os.path.exists(self.model_path) and TENSORFLOW_AVAILABLE:
                self.model = tf.keras.models.load_model(self.model_path)
                # Load class names (this would be saved with the model)
                class_names_path = self.model_path.replace('.h5', '_classes.pkl').replace('.pb', '_classes.pkl')
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'rb') as f:
                        self.class_names = pickle.load(f)
                else:
                    # Default common ingredients
                    self.class_names = [
                        'apple', 'banana', 'orange', 'tomato', 'potato', 'carrot',
                        'onion', 'garlic', 'lettuce', 'spinach', 'broccoli', 'cucumber',
                        'chicken', 'beef', 'fish', 'eggs', 'milk', 'cheese',
                        'rice', 'pasta', 'bread', 'flour', 'sugar', 'salt'
                    ]

                self.is_loaded = True
                logger.info("Ingredient detection model loaded successfully")
                return True
            else:
                logger.info("Using basic computer vision methods (no ML model)")
                return False
        except Exception as e:
            logger.error(f"Error loading ingredient detection model: {e}")
            return False

    def detect_ingredients_from_image(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect ingredients from image data"""
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Invalid image data")

            # Use ML model if available
            if self.is_loaded and TENSORFLOW_AVAILABLE:
                return self._detect_with_ml_model(image, confidence_threshold)
            else:
                return self._detect_with_basic_cv(image)

        except Exception as e:
            logger.error(f"Error detecting ingredients: {e}")
            return []

    def _detect_with_ml_model(self, image: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect ingredients using ML model"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Make prediction
            predictions = self.model.predict(processed_image)[0]

            # Get top predictions above threshold
            detected_ingredients = []
            for i, confidence in enumerate(predictions):
                if confidence >= confidence_threshold and i < len(self.class_names):
                    detected_ingredients.append({
                        'name': self.class_names[i],
                        'confidence': float(confidence),
                        'category': self._categorize_ingredient(self.class_names[i])
                    })

            # Sort by confidence
            detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)

            return detected_ingredients[:10]  # Return top 10

        except Exception as e:
            logger.error(f"Error in ML-based detection: {e}")
            return self._detect_with_basic_cv(image)

    def _detect_with_basic_cv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detection using basic computer vision techniques"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Basic color-based detection for common ingredients
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            detections = []

            # Detect red objects (tomatoes, apples, etc.)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            if cv2.countNonZero(mask_red) > 1000:  # Minimum area threshold
                detections.append({
                    'name': 'tomato',
                    'confidence': 0.7,
                    'category': 'vegetables'
                })

            # Detect yellow objects (bananas, lemons, etc.)
            lower_yellow = np.array([20, 50, 50])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            if cv2.countNonZero(mask_yellow) > 1000:
                detections.append({
                    'name': 'banana',
                    'confidence': 0.6,
                    'category': 'fruits'
                })

            # Detect green objects (lettuce, cucumbers, etc.)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            if cv2.countNonZero(mask_green) > 2000:
                detections.append({
                    'name': 'lettuce',
                    'confidence': 0.5,
                    'category': 'vegetables'
                })

            return detections

        except Exception as e:
            logger.error(f"Error in basic CV detection: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ML model"""
        try:
            # Resize to model input size (assuming 224x224)
            resized = cv2.resize(image, (224, 224))

            # Convert to RGB if needed
            if resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize
            resized = resized.astype(np.float32) / 255.0

            # Add batch dimension
            return np.expand_dims(resized, axis=0)

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return a dummy batch
            return np.zeros((1, 224, 224, 3), dtype=np.float32)

    def _categorize_ingredient(self, ingredient_name: str) -> str:
        """Categorize ingredient based on name"""
        fruit_keywords = ['apple', 'banana', 'orange', 'lemon', 'grape', 'strawberry', 'blueberry']
        vegetable_keywords = ['tomato', 'potato', 'carrot', 'onion', 'garlic', 'lettuce', 'spinach', 'broccoli', 'cucumber']
        protein_keywords = ['chicken', 'beef', 'fish', 'eggs', 'tofu', 'beans']
        dairy_keywords = ['milk', 'cheese', 'butter', 'yogurt']
        grain_keywords = ['rice', 'pasta', 'bread', 'flour']

        name_lower = ingredient_name.lower()

        if any(keyword in name_lower for keyword in fruit_keywords):
            return 'fruits'
        elif any(keyword in name_lower for keyword in vegetable_keywords):
            return 'vegetables'
        elif any(keyword in name_lower for keyword in protein_keywords):
            return 'protein'
        elif any(keyword in name_lower for keyword in dairy_keywords):
            return 'dairy'
        elif any(keyword in name_lower for keyword in grain_keywords):
            return 'grains'
        else:
            return 'other'

    def detect_ingredients_from_base64(
        self,
        base64_image: str,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect ingredients from base64 encoded image"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
            return self.detect_ingredients_from_image(image_data, confidence_threshold)
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return []

    def get_camera_requirements(self) -> Dict[str, Any]:
        """Get camera and detection requirements"""
        return {
            'min_resolution': '640x480',
            'recommended_resolution': '1920x1080',
            'supported_formats': ['jpeg', 'png', 'webp'],
            'max_file_size': settings.MAX_UPLOAD_SIZE,
            'features': {
                'real_time_detection': False,  # Would need WebRTC for real-time
                'batch_processing': True,
                'confidence_thresholds': True
            }
        }

    def train_model(self, training_data_path: str) -> bool:
        """Train the ingredient detection model (placeholder for future implementation)"""
        logger.info(f"Training model with data from: {training_data_path}")
        # This would implement actual model training with ingredient images
        # For now, this is a placeholder
        return False

    def save_detected_image(self, image_data: bytes, detections: List[Dict], output_path: str) -> bool:
        """Save image with detection bounding boxes (for debugging/visualization)"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Draw detections on image
            for detection in detections:
                # For now, just add text labels (would need bounding boxes for full implementation)
                text = f"{detection['name']}: {detection['confidence']:.2f}"
                cv2.putText(image, text, (10, 30 + detections.index(detection) * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imwrite(output_path, image)
            return True

        except Exception as e:
            logger.error(f"Error saving detected image: {e}")
            return False
