"""
T2I Integration for Story System

This module provides the integration layer between the Story Engine and T2I Engine,
enabling automatic scene image generation during story progression.
"""

from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SceneImageResult:
    """Result from scene image generation"""
    image_url: str
    prompt: str
    negative_prompt: str
    generation_time: float
    seed: Optional[int] = None
    width: int = 768
    height: int = 768


class StoryT2IIntegration:
    """
    Integration layer for Story-driven T2I generation

    This class manages automatic scene image generation based on story context,
    determining when to generate images and constructing appropriate prompts.
    """

    def __init__(self, t2i_engine=None, prompt_generator=None):
        """
        Initialize T2I integration

        Args:
            t2i_engine: T2I engine instance (will be lazy-loaded if None)
            prompt_generator: Story prompt generator instance (will be lazy-loaded if None)
        """
        self._t2i_engine = t2i_engine
        self._prompt_generator = prompt_generator

        # Trigger conditions for image generation
        self.auto_trigger_keywords = [
            "進入", "到達", "來到", "看見", "發現",  # Chinese
            "enter", "arrive", "reach", "see", "discover"  # English
        ]

        # Generation settings
        self.default_width = 768
        self.default_height = 768
        self.default_steps = 25
        self.default_cfg_scale = 7.0

    @property
    def t2i_engine(self):
        """Lazy load T2I engine"""
        if self._t2i_engine is None:
            from core.t2i.engine import get_t2i_engine
            self._t2i_engine = get_t2i_engine()
        return self._t2i_engine

    @property
    def prompt_generator(self):
        """Lazy load prompt generator"""
        if self._prompt_generator is None:
            from core.t2i.story_prompt_generator import StoryPromptGenerator
            self._prompt_generator = StoryPromptGenerator()
        return self._prompt_generator

    def _should_generate(self, scene_context: Dict[str, Any], narrative_text: str) -> bool:
        """
        Determine if an image should be generated for this scene

        Args:
            scene_context: Current scene context (location, time, atmosphere, etc.)
            narrative_text: The narrative text from story engine

        Returns:
            True if image should be generated
        """
        # Always generate on scene transitions
        if scene_context.get("scene_transition", False):
            return True

        # Check if narrative contains trigger keywords
        text_lower = narrative_text.lower()
        for keyword in self.auto_trigger_keywords:
            if keyword in text_lower:
                return True

        # Generate on major events
        if scene_context.get("is_major_event", False):
            return True

        return False

    async def generate_scene_image(
        self,
        scene_context: Dict[str, Any],
        narrative_text: str = "",
        force: bool = False
    ) -> Optional[SceneImageResult]:
        """
        Generate scene image based on story context

        Args:
            scene_context: Scene context dictionary containing:
                - location: Current location name
                - time: Time of day (morning/afternoon/evening/night)
                - atmosphere: Scene atmosphere (peaceful/tense/mysterious/etc)
                - characters: List of characters present
                - scene_transition: Whether this is a scene transition
                - is_major_event: Whether this is a major story event
            narrative_text: The narrative text to check for triggers
            force: Force generation regardless of trigger conditions

        Returns:
            SceneImageResult if image was generated, None otherwise
        """
        try:
            # Check if we should generate
            if not force and not self._should_generate(scene_context, narrative_text):
                logger.debug("Scene image generation skipped - no triggers matched")
                return None

            # Generate prompt from scene context
            prompt_data = await self.prompt_generator.generate_from_scene(scene_context)

            logger.info(f"Generating scene image: {prompt_data.positive[:100]}...")

            # Call T2I engine
            result = await self.t2i_engine.generate(
                prompt=prompt_data.positive,
                negative_prompt=prompt_data.negative,
                width=self.default_width,
                height=self.default_height,
                num_inference_steps=self.default_steps,
                guidance_scale=self.default_cfg_scale
            )

            # Convert to SceneImageResult
            scene_image = SceneImageResult(
                image_url=result.get("url", result.get("image_url", "")),
                prompt=prompt_data.positive,
                negative_prompt=prompt_data.negative,
                generation_time=result.get("generation_time", result.get("time", 0)),
                seed=result.get("seed"),
                width=result.get("width", self.default_width),
                height=result.get("height", self.default_height)
            )

            logger.info(f"Scene image generated in {scene_image.generation_time:.2f}s")
            return scene_image

        except Exception as e:
            logger.error(f"Failed to generate scene image: {e}", exc_info=True)
            return None

    async def regenerate_with_modifications(
        self,
        base_scene_image: SceneImageResult,
        modifications: Dict[str, Any]
    ) -> Optional[SceneImageResult]:
        """
        Regenerate scene image with modifications

        Args:
            base_scene_image: Previous scene image result
            modifications: Modifications to apply (e.g., {"atmosphere": "darker", "add_character": "warrior"})

        Returns:
            New SceneImageResult
        """
        try:
            # Modify prompt based on changes
            modified_prompt = await self.prompt_generator.modify_prompt(
                base_prompt=base_scene_image.prompt,
                modifications=modifications
            )

            # Regenerate with same seed for consistency (if available)
            result = await self.t2i_engine.generate(
                prompt=modified_prompt.positive,
                negative_prompt=modified_prompt.negative,
                width=base_scene_image.width,
                height=base_scene_image.height,
                num_inference_steps=self.default_steps,
                guidance_scale=self.default_cfg_scale,
                seed=base_scene_image.seed  # Use same seed for similar composition
            )

            return SceneImageResult(
                image_url=result.get("url", result.get("image_url", "")),
                prompt=modified_prompt.positive,
                negative_prompt=modified_prompt.negative,
                generation_time=result.get("generation_time", result.get("time", 0)),
                seed=result.get("seed"),
                width=result.get("width", base_scene_image.width),
                height=result.get("height", base_scene_image.height)
            )

        except Exception as e:
            logger.error(f"Failed to regenerate scene image: {e}", exc_info=True)
            return None


# Singleton instance
_t2i_integration_instance: Optional[StoryT2IIntegration] = None


def get_t2i_integration() -> StoryT2IIntegration:
    """Get or create singleton T2I integration instance"""
    global _t2i_integration_instance
    if _t2i_integration_instance is None:
        _t2i_integration_instance = StoryT2IIntegration()
    return _t2i_integration_instance
