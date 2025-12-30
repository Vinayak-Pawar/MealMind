"""
Image Generation Service - AI-powered recipe images using Nano-Banana
"""

import os
import json
import time
import base64
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import aiohttp
from loguru import logger
from PIL import Image
import io

from app.config import settings


class RecipeImageGenerator:
    """Service for generating AI-powered recipe images using Nano-Banana"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """Initialize the image generator"""
        self.api_key = api_key or settings.NANO_BANANA_API_KEY
        self.api_url = api_url or "https://api.nano-banana.com/v1/images/generations"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def generate_recipe_image(
        self,
        recipe_title: str,
        ingredients: List[str],
        style: str = "photorealistic",
        mood: str = "appetizing",
        additional_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a recipe image using Nano-Banana AI

        Args:
            recipe_title: Name of the recipe
            ingredients: List of key ingredients
            style: Image style (photorealistic, artistic, cartoon, etc.)
            mood: Image mood/tone
            additional_prompt: Additional prompt instructions

        Returns:
            Dictionary with image generation status and metadata
        """
        try:
            # Create image generation prompt
            prompt = self._create_image_prompt(
                recipe_title, ingredients, style, mood, additional_prompt
            )

            # Generate image
            image_result = await self._generate_image_with_nano_banana(prompt)

            if image_result['success']:
                # Save image locally
                saved_image = await self._save_generated_image(
                    image_result['image_data'],
                    recipe_title
                )

                return {
                    'success': True,
                    'image_url': saved_image['url'],
                    'image_path': saved_image['path'],
                    'prompt_used': prompt,
                    'metadata': {
                        'recipe_title': recipe_title,
                        'style': style,
                        'mood': mood,
                        'generated_at': datetime.now().isoformat(),
                        'generator': 'nano_banana',
                        'resolution': image_result.get('resolution', '512x512')
                    }
                }
            else:
                return {
                    'success': False,
                    'error': image_result.get('error', 'Unknown error'),
                    'image_url': None
                }

        except Exception as e:
            logger.error(f"Error generating recipe image: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_url': None
            }

    def _create_image_prompt(
        self,
        recipe_title: str,
        ingredients: List[str],
        style: str,
        mood: str,
        additional_prompt: str
    ) -> str:
        """Create a detailed prompt for image generation"""

        # Base prompt structure
        base_prompt = f"A beautifully plated {recipe_title}"

        # Add ingredient details
        key_ingredients = ingredients[:5]  # Use top 5 ingredients
        ingredient_text = ", ".join(key_ingredients)
        ingredient_prompt = f" featuring {ingredient_text}"

        # Style specifications
        style_prompts = {
            'photorealistic': "professional food photography, highly detailed, realistic lighting, appetizing presentation",
            'artistic': "artistic food illustration, creative composition, vibrant colors, artistic lighting",
            'minimalist': "minimalist food presentation, clean background, elegant simplicity, soft lighting",
            'rustic': "rustic homemade style, natural ingredients visible, cozy kitchen setting, warm lighting",
            'modern': "modern culinary presentation, sleek plating, contemporary kitchen, dramatic lighting",
            'vintage': "vintage style food photography, retro kitchen aesthetic, nostalgic presentation"
        }

        style_text = style_prompts.get(style, style_prompts['photorealistic'])

        # Mood specifications
        mood_prompts = {
            'appetizing': "mouth-watering, fresh, delicious appearance",
            'healthy': "fresh, nutritious, wholesome appearance, vibrant colors",
            'comforting': "warm, comforting, hearty appearance, inviting",
            'elegant': "sophisticated, elegant presentation, refined",
            'fun': "playful, fun presentation, colorful and engaging",
            'exotic': "exotic, adventurous appearance, intriguing flavors"
        }

        mood_text = mood_prompts.get(mood, mood_prompts['appetizing'])

        # Combine all elements
        full_prompt = f"{base_prompt}{ingredient_prompt}, {style_text}, {mood_text}"

        if additional_prompt:
            full_prompt += f", {additional_prompt}"

        # Add quality enhancers
        full_prompt += ", high resolution, professional quality, food photography, culinary art"

        return full_prompt

    async def _generate_image_with_nano_banana(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Nano-Banana API"""
        if not self.api_key:
            return {
                'success': False,
                'error': 'Nano-Banana API key not configured'
            }

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'prompt': prompt,
                'n': 1,
                'size': '512x512',  # Standard size
                'response_format': 'b64_json'
            }

            async with self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:

                if response.status == 200:
                    result = await response.json()

                    if 'data' in result and len(result['data']) > 0:
                        image_data = result['data'][0]['b64_json']

                        return {
                            'success': True,
                            'image_data': image_data,
                            'resolution': payload['size']
                        }
                    else:
                        return {
                            'success': False,
                            'error': 'No image data in response'
                        }
                else:
                    error_text = await response.text()
                    logger.error(f"Nano-Banana API error: {response.status} - {error_text}")
                    return {
                        'success': False,
                        'error': f'API error: {response.status}'
                    }

        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Request timeout'
            }
        except Exception as e:
            logger.error(f"Error calling Nano-Banana API: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _save_generated_image(self, base64_data: str, recipe_title: str) -> Dict[str, str]:
        """Save the generated image to disk"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_data)

            # Create PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Generate filename
            safe_title = "".join(c for c in recipe_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            timestamp = int(time.time())
            filename = f"{safe_title}_{timestamp}.png"

            # Save path
            save_path = os.path.join(settings.UPLOAD_DIR, 'generated_images', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save image
            image.save(save_path, 'PNG')

            return {
                'url': f'/uploads/generated_images/{filename}',
                'path': save_path
            }

        except Exception as e:
            logger.error(f"Error saving generated image: {e}")
            raise

    async def generate_image_variations(
        self,
        original_image_url: str,
        variations: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate variations of an existing recipe image"""
        try:
            # This would load the original image and create variations
            # For now, return placeholder
            variations_list = []

            for i in range(variations):
                # In a real implementation, this would modify the prompt slightly
                # and generate new images
                variation = {
                    'success': True,
                    'image_url': f'{original_image_url}_variation_{i+1}',
                    'variation_type': f'variation_{i+1}'
                }
                variations_list.append(variation)

            return variations_list

        except Exception as e:
            logger.error(f"Error generating image variations: {e}")
            return []

    def get_supported_styles(self) -> List[str]:
        """Get list of supported image styles"""
        return [
            'photorealistic',
            'artistic',
            'minimalist',
            'rustic',
            'modern',
            'vintage',
            'watercolor',
            'sketch',
            '3d_render'
        ]

    def get_supported_moods(self) -> List[str]:
        """Get list of supported image moods"""
        return [
            'appetizing',
            'healthy',
            'comforting',
            'elegant',
            'fun',
            'exotic',
            'cozy',
            'luxurious',
            'fresh',
            'hearty'
        ]

    def get_image_requirements(self) -> Dict[str, Any]:
        """Get image generation requirements and capabilities"""
        return {
            'max_prompt_length': 1000,
            'supported_resolutions': ['256x256', '512x512', '1024x1024'],
            'supported_formats': ['png', 'jpg'],
            'max_variations': 10,
            'generation_time': '10-30 seconds',
            'rate_limits': {
                'requests_per_minute': 10,
                'requests_per_hour': 100
            },
            'features': {
                'style_variations': True,
                'mood_control': True,
                'ingredient_focus': True,
                'batch_generation': True
            },
            'apis_required': ['nano_banana']
        }

    async def get_generation_status(self, generation_id: str) -> Dict[str, Any]:
        """Get the status of an image generation job"""
        # In a real implementation, this would check a job queue or database
        return {
            'generation_id': generation_id,
            'status': 'completed',  # or 'processing', 'failed'
            'progress': 100,
            'estimated_time_remaining': 0
        }

    async def validate_api_connection(self) -> bool:
        """Validate Nano-Banana API connection"""
        if not self.api_key:
            return False

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            # Simple test request (models endpoint or similar)
            test_url = "https://api.nano-banana.com/v1/models"  # Adjust based on actual API

            async with self.session.get(test_url, headers=headers, timeout=10) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"API validation error: {e}")
            return False
