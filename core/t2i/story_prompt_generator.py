"""
Story-specific T2I Prompt Generator

This module converts story scene context into optimized T2I prompts
for generating anime-style scene illustrations.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class T2IPrompt:
    """T2I prompt data"""
    positive: str
    negative: str


class StoryPromptGenerator:
    """
    Generates T2I prompts from story scene context

    This class translates story context (location, atmosphere, characters, time)
    into optimized prompts for anime-style image generation.
    """

    def __init__(self):
        # Base quality tags for anime style
        self.base_quality_tags = [
            "anime style",
            "high quality",
            "detailed background",
            "masterpiece",
            "best quality"
        ]

        # Negative prompt defaults
        self.base_negative_tags = [
            "low quality",
            "blurry",
            "distorted",
            "ugly",
            "bad anatomy",
            "bad hands",
            "text",
            "watermark",
            "signature",
            "nsfw",
            "explicit"
        ]

        # Time of day lighting mappings
        self.time_lighting = {
            "morning": "soft morning light, golden hour, warm lighting",
            "afternoon": "bright daylight, clear sky, natural lighting",
            "evening": "sunset glow, orange sky, warm evening light",
            "night": "moonlight, starry sky, dark atmosphere, night scene",
            "dawn": "dawn light, purple and orange sky, early morning",
            "dusk": "twilight, dusk, fading light"
        }

        # Atmosphere descriptors
        self.atmosphere_tags = {
            "peaceful": "calm, serene, peaceful atmosphere, tranquil",
            "tense": "tense atmosphere, dramatic lighting, ominous",
            "mysterious": "mysterious, foggy, shadowy, enigmatic",
            "dangerous": "dangerous, threatening, dark, ominous clouds",
            "joyful": "bright, cheerful, vibrant colors, happy atmosphere",
            "sad": "melancholic, rain, gray sky, somber mood",
            "epic": "epic, grand scale, dramatic, cinematic",
            "horror": "horror, dark, eerie, creepy atmosphere"
        }

        # Location type templates
        self.location_templates = {
            "forest": "forest background, trees, nature, wilderness",
            "city": "city background, buildings, urban environment",
            "dungeon": "dungeon interior, stone walls, torches, underground",
            "castle": "castle interior, grand hall, medieval architecture",
            "tavern": "tavern interior, wooden furniture, warm lighting, medieval inn",
            "mountain": "mountain landscape, peaks, rocky terrain",
            "beach": "beach, ocean, sand, coastal scenery",
            "village": "village, houses, rural setting, countryside"
        }

    async def generate_from_scene(self, scene_context: Dict[str, Any]) -> T2IPrompt:
        """
        Generate T2I prompt from scene context

        Args:
            scene_context: Scene context containing:
                - location: Location name/type
                - time: Time of day
                - atmosphere: Scene atmosphere
                - characters: List of character names (optional)
                - weather: Weather condition (optional)
                - special_elements: List of special elements to include (optional)

        Returns:
            T2IPrompt with positive and negative prompts
        """
        # Extract context elements
        location = scene_context.get("location", "unknown place")
        time_of_day = scene_context.get("time", "daytime")
        atmosphere = scene_context.get("atmosphere", "neutral")
        weather = scene_context.get("weather", "")
        special_elements = scene_context.get("special_elements", [])

        # Build positive prompt components
        prompt_parts = []

        # 1. Location description
        location_desc = self._get_location_description(location)
        prompt_parts.append(location_desc)

        # 2. Time and lighting
        lighting = self.time_lighting.get(time_of_day.lower(), "natural lighting")
        prompt_parts.append(lighting)

        # 3. Atmosphere
        atmosphere_desc = self.atmosphere_tags.get(atmosphere.lower(), atmosphere)
        prompt_parts.append(atmosphere_desc)

        # 4. Weather (if specified)
        if weather:
            prompt_parts.append(weather)

        # 5. Special elements
        if special_elements:
            prompt_parts.extend(special_elements)

        # 6. Quality tags
        prompt_parts.extend(self.base_quality_tags)

        # Combine positive prompt
        positive_prompt = ", ".join(prompt_parts)

        # Build negative prompt
        negative_prompt = ", ".join(self.base_negative_tags)

        logger.debug(f"Generated prompt - Positive: {positive_prompt[:100]}...")

        return T2IPrompt(
            positive=positive_prompt,
            negative=negative_prompt
        )

    def _get_location_description(self, location: str) -> str:
        """
        Get location description from location name

        Args:
            location: Location name or type

        Returns:
            Location description for prompt
        """
        location_lower = location.lower()

        # Check for known location types
        for loc_type, template in self.location_templates.items():
            if loc_type in location_lower:
                return template

        # Return generic location description
        return f"{location} background, detailed environment"

    async def modify_prompt(
        self,
        base_prompt: str,
        modifications: Dict[str, Any]
    ) -> T2IPrompt:
        """
        Modify existing prompt with changes

        Args:
            base_prompt: Original prompt text
            modifications: Dictionary of modifications:
                - atmosphere: Change atmosphere
                - add_elements: List of elements to add
                - remove_elements: List of elements to remove
                - time: Change time of day

        Returns:
            Modified T2IPrompt
        """
        # Parse base prompt into parts
        prompt_parts = [p.strip() for p in base_prompt.split(",")]

        # Apply modifications
        if "atmosphere" in modifications:
            # Remove old atmosphere tags
            prompt_parts = [p for p in prompt_parts if not any(
                atm in p.lower() for atm in self.atmosphere_tags.keys()
            )]
            # Add new atmosphere
            new_atmosphere = self.atmosphere_tags.get(
                modifications["atmosphere"].lower(),
                modifications["atmosphere"]
            )
            prompt_parts.insert(2, new_atmosphere)

        if "time" in modifications:
            # Remove old time/lighting tags
            prompt_parts = [p for p in prompt_parts if not any(
                time_tag in p.lower() for time_tag in ["light", "lighting", "sky", "morning", "evening", "night"]
            )]
            # Add new time lighting
            new_lighting = self.time_lighting.get(
                modifications["time"].lower(),
                "natural lighting"
            )
            prompt_parts.insert(1, new_lighting)

        if "add_elements" in modifications:
            # Add new elements before quality tags
            quality_index = len(prompt_parts) - len(self.base_quality_tags)
            for element in modifications["add_elements"]:
                prompt_parts.insert(quality_index, element)

        if "remove_elements" in modifications:
            # Remove specified elements
            remove_set = set(e.lower() for e in modifications["remove_elements"])
            prompt_parts = [p for p in prompt_parts if p.lower() not in remove_set]

        # Rebuild prompt
        positive_prompt = ", ".join(prompt_parts)
        negative_prompt = ", ".join(self.base_negative_tags)

        return T2IPrompt(
            positive=positive_prompt,
            negative=negative_prompt
        )

    def add_character_to_prompt(
        self,
        base_prompt: str,
        character_description: str
    ) -> T2IPrompt:
        """
        Add character to existing scene prompt

        Args:
            base_prompt: Base scene prompt
            character_description: Character description to add

        Returns:
            Modified prompt with character
        """
        # Insert character description at the beginning
        positive_prompt = f"{character_description}, {base_prompt}"

        return T2IPrompt(
            positive=positive_prompt,
            negative=", ".join(self.base_negative_tags)
        )

    async def generate_character_portrait(
        self,
        character_name: str,
        appearance_desc: str,
        visual_style: Optional[str] = None
    ) -> T2IPrompt:
        """
        Generate T2I prompt for a character portrait (sprite)

        Args:
            character_name: Name of the character
            appearance_desc: Physical appearance description
            visual_style: Optional visual style hints

        Returns:
            T2IPrompt with optimized character sprite tags
        """
        # Character-specific portrait tags
        portrait_tags = [
            "full body",
            "standing",
            "facing viewer",
            "simple white background",
            "transparent background",
            "character concept art",
            "high quality anime style",
            "solo"
        ]

        prompt_parts = []

        # 1. Subject and basic appearance
        if character_name:
            # Avoid using name directly if it's a generic prompt,
            # but for anime characters it sometimes helps with consistency.
            prompt_parts.append(f"1character, {character_name}")
        else:
            prompt_parts.append("1character")

        prompt_parts.append(appearance_desc)

        # 2. Portrait specific constraints
        prompt_parts.extend(portrait_tags)

        # 3. Quality and style tags
        if visual_style:
            prompt_parts.append(visual_style)

        prompt_parts.extend(self.base_quality_tags)

        # Combine positive prompt
        positive_prompt = ", ".join(prompt_parts)

        # Build negative prompt - extra emphasis on no backgrounds
        negative_parts = list(self.base_negative_tags)
        negative_parts.extend([
            "background",
            "landscape",
            "complex background",
            "cityscape",
            "forest",
            "nature",
            "scenery",
            "multiple characters",
            "group",
            "words",
            "border",
            "frame"
        ])
        negative_prompt = ", ".join(negative_parts)

        return T2IPrompt(
            positive=positive_prompt,
            negative=negative_prompt
        )
