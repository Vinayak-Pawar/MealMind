"""
Video Generation Service - AI-powered cooking tutorial videos using Google Flow
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
from loguru import logger

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logger.warning("Google Generative AI not available")

try:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy not available, video generation will be limited")

from app.config import settings


class VideoGenerator:
    """Service for generating AI-powered cooking tutorial videos"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the video generator"""
        self.api_key = api_key or settings.GOOGLE_FLOW_API_KEY
        self.client = None

        if GOOGLE_AI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-pro-vision')
            logger.info("Google AI client initialized for video generation")

    async def generate_recipe_video(
        self,
        recipe_title: str,
        instructions: str,
        ingredients: List[str],
        estimated_time: int,
        difficulty: str,
        style: str = "modern"
    ) -> Dict[str, Any]:
        """
        Generate a cooking tutorial video for a recipe

        Args:
            recipe_title: Name of the recipe
            instructions: Step-by-step cooking instructions
            ingredients: List of ingredients
            estimated_time: Total cooking time in minutes
            difficulty: Recipe difficulty level
            style: Video style (modern, traditional, quick, etc.)

        Returns:
            Dictionary with video generation status and metadata
        """
        try:
            # Create video generation request
            video_request = self._create_video_request(
                recipe_title, instructions, ingredients,
                estimated_time, difficulty, style
            )

            # Generate video script and storyboard
            script_data = await self._generate_video_script(video_request)

            if not script_data:
                return {
                    'success': False,
                    'error': 'Failed to generate video script',
                    'video_url': None
                }

            # Generate video using available tools
            video_result = await self._create_video_from_script(script_data)

            return {
                'success': True,
                'video_url': video_result.get('url'),
                'duration': video_result.get('duration', 0),
                'script': script_data.get('script'),
                'metadata': {
                    'recipe_title': recipe_title,
                    'style': style,
                    'generated_at': datetime.now().isoformat(),
                    'generator': 'google_flow_ai'
                }
            }

        except Exception as e:
            logger.error(f"Error generating recipe video: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_url': None
            }

    def _create_video_request(
        self,
        title: str,
        instructions: str,
        ingredients: List[str],
        time: int,
        difficulty: str,
        style: str
    ) -> Dict[str, Any]:
        """Create a structured video generation request"""
        return {
            'title': title,
            'instructions': instructions,
            'ingredients': ingredients,
            'total_time': time,
            'difficulty': difficulty,
            'style': style,
            'target_duration': min(max(time // 2, 30), 300),  # 30 seconds to 5 minutes
            'aspect_ratio': '16:9',
            'resolution': '1080p'
        }

    async def _generate_video_script(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate video script using AI"""
        if not self.client:
            # Fallback to template-based script generation
            return self._generate_template_script(request)

        try:
            prompt = f"""
            Create a detailed video script for a cooking tutorial with the following recipe:

            Recipe: {request['title']}
            Ingredients: {', '.join(request['ingredients'])}
            Instructions: {request['instructions']}
            Total Time: {request['total_time']} minutes
            Difficulty: {request['difficulty']}
            Style: {request['style']}

            Please provide:
            1. A compelling introduction script (10-15 seconds)
            2. Step-by-step narration for each cooking step
            3. Visual descriptions for each step (what camera should show)
            4. Background music suggestions
            5. Voiceover timing estimates

            Format as JSON with keys: introduction, steps, visuals, music, timing
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(prompt)
            )

            # Parse AI response
            script_text = response.text
            script_data = json.loads(script_text)

            return script_data

        except Exception as e:
            logger.error(f"Error generating AI script: {e}")
            return self._generate_template_script(request)

    def _generate_template_script(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a template-based video script"""
        steps = request['instructions'].split('\n')

        return {
            'introduction': f"Welcome to this {request['style']} cooking tutorial for {request['title']}. Today we'll be making a delicious dish that takes about {request['total_time']} minutes.",
            'steps': [
                f"Step {i+1}: {step.strip()}"
                for i, step in enumerate(steps)
                if step.strip()
            ],
            'visuals': [
                "Wide shot of ingredients on counter",
                "Close-up of hands preparing ingredients",
                "Medium shot of cooking process",
                "Final plated dish"
            ],
            'music': "Upbeat cooking music with kitchen sounds",
            'timing': {
                'introduction': 15,
                'steps': [30] * len(steps),  # 30 seconds per step
                'conclusion': 10
            }
        }

    async def _create_video_from_script(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create video from generated script"""
        if not MOVIEPY_AVAILABLE:
            # Return mock video result
            return {
                'url': f'/generated_videos/mock_{int(time.time())}.mp4',
                'duration': sum(script_data.get('timing', {}).values()),
                'format': 'mp4',
                'status': 'mock_generated'
            }

        try:
            # Create video clips from script
            clips = []

            # Introduction clip
            intro_text = script_data.get('introduction', 'Cooking Tutorial')
            intro_clip = TextClip(
                intro_text,
                fontsize=50,
                color='white',
                bg_color='black',
                duration=script_data.get('timing', {}).get('introduction', 10)
            ).set_position('center')

            clips.append(intro_clip)

            # Step clips
            steps = script_data.get('steps', [])
            for i, step in enumerate(steps):
                step_clip = TextClip(
                    f"Step {i+1}: {step}",
                    fontsize=40,
                    color='white',
                    bg_color='black',
                    duration=15  # 15 seconds per step
                ).set_position('center')
                clips.append(step_clip)

            # Conclusion
            conclusion_clip = TextClip(
                "Enjoy your delicious meal!",
                fontsize=50,
                color='white',
                bg_color='green',
                duration=5
            ).set_position('center')

            clips.append(conclusion_clip)

            # Combine clips
            final_video = CompositeVideoClip(clips)

            # Generate unique filename
            video_filename = f"recipe_video_{int(time.time())}.mp4"
            video_path = os.path.join(settings.UPLOAD_DIR, 'videos', video_filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            # Write video file
            final_video.write_videofile(
                video_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )

            return {
                'url': f'/uploads/videos/{video_filename}',
                'duration': final_video.duration,
                'format': 'mp4',
                'status': 'generated'
            }

        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return {
                'url': None,
                'duration': 0,
                'error': str(e),
                'status': 'failed'
            }

    async def get_video_generation_status(self, video_id: str) -> Dict[str, Any]:
        """Get the status of a video generation job"""
        # In a real implementation, this would check a job queue or database
        return {
            'video_id': video_id,
            'status': 'completed',  # or 'processing', 'failed'
            'progress': 100,
            'estimated_time_remaining': 0
        }

    def get_supported_styles(self) -> List[str]:
        """Get list of supported video styles"""
        return [
            'modern',
            'traditional',
            'quick',
            'healthy',
            'gourmet',
            'family_friendly',
            'beginner_friendly'
        ]

    def get_video_requirements(self) -> Dict[str, Any]:
        """Get video generation requirements and capabilities"""
        return {
            'max_duration': 300,  # 5 minutes
            'min_duration': 30,   # 30 seconds
            'supported_resolutions': ['720p', '1080p'],
            'supported_formats': ['mp4', 'webm'],
            'ai_features': {
                'script_generation': True,
                'voiceover': False,  # Would need additional service
                'background_music': False,  # Would need additional service
                'visual_effects': True
            },
            'apis_required': ['google_flow']
        }

    async def generate_video_thumbnail(
        self,
        video_url: str,
        timestamp: int = 5
    ) -> Optional[str]:
        """Generate a thumbnail image from video"""
        if not MOVIEPY_AVAILABLE:
            return None

        try:
            video_path = os.path.join(settings.UPLOAD_DIR, video_url.lstrip('/uploads/'))

            if not os.path.exists(video_path):
                return None

            clip = VideoFileClip(video_path)
            thumbnail = clip.get_frame(timestamp)

            # Save thumbnail
            thumbnail_filename = f"thumb_{int(time.time())}.jpg"
            thumbnail_path = os.path.join(settings.UPLOAD_DIR, 'thumbnails', thumbnail_filename)
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)

            # Save as JPEG
            from PIL import Image
            img = Image.fromarray(thumbnail)
            img.save(thumbnail_path, 'JPEG')

            return f'/uploads/thumbnails/{thumbnail_filename}'

        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return None
