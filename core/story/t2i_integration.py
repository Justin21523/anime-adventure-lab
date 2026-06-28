"""
T2I Integration for Story System

This module provides the integration layer between the Story Engine and T2I Engine,
enabling automatic scene image generation during story progression.
"""

from typing import Optional, Dict, Any, List, Tuple
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

    def _apply_world_style(
        self, world_id: str, positive_prompt: str, negative_prompt: str
    ) -> Tuple[str, str, Optional[str], List[Dict[str, Any]]]:
        """Apply world visual style (LoRA / prefix / negatives)"""
        world_id = str(world_id or "default").strip() or "default"
        lora_configs: List[Dict[str, Any]] = []
        model_id: Optional[str] = None

        try:
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(world_id)
            if worldpack:
                visual = worldpack.visual
                if getattr(visual, "prompt_prefix", "") and str(visual.prompt_prefix).strip():
                    positive_prompt = ", ".join(
                        [str(visual.prompt_prefix).strip(), positive_prompt.strip()]
                    )
                if getattr(visual, "negative_prompt", "") and str(visual.negative_prompt).strip():
                    negative_prompt = ", ".join(
                        [negative_prompt.strip(), str(visual.negative_prompt).strip()]
                    )
                if getattr(visual, "base_model", None):
                    model_id = str(visual.base_model)
                if getattr(visual, "default_loras", None):
                    lora_configs = [
                        {"lora_id": l.lora_id, "weight": float(getattr(l, "weight", 0.8))}
                        for l in visual.default_loras
                        if getattr(l, "lora_id", None)
                    ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("World visual style skipped: %s", exc)

        return positive_prompt, negative_prompt, model_id, lora_configs

    def _process_t2i_result(self, result: Dict[str, Any], positive_prompt: str, negative_prompt: str) -> SceneImageResult:
        """Process T2I engine result into SceneImageResult"""
        images_base64 = result.get("images") or []
        metadata = result.get("metadata", {}) or {}
        output_paths = metadata.get("output_paths") or []

        image_url = ""
        if output_paths:
            try:
                from urllib.parse import quote
                from pathlib import Path
                from core.shared_cache import get_shared_cache

                cache = get_shared_cache()
                root = Path(cache.get_path("OUTPUT_DIR")).resolve()
                rel = Path(str(output_paths[0])).resolve().relative_to(root)
                image_url = f"/api/v1/t2i/file?path={quote(str(rel))}"
            except Exception:
                image_url = ""
        if (not image_url) and images_base64:
            image_url = f"data:image/png;base64,{images_base64[0]}"
        if not image_url:
            image_url = result.get("url") or result.get("image_url") or ""

        return SceneImageResult(
            image_url=image_url,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            generation_time=float(metadata.get("generation_time", 0.0) or 0.0),
            seed=metadata.get("seed"),
            width=int(metadata.get("parameters", {}).get("width", self.default_width) or self.default_width),
            height=int(metadata.get("parameters", {}).get("height", self.default_height) or self.default_height),
        )

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

            positive_prompt, negative_prompt, model_id, lora_configs = self._apply_world_style(
                scene_context.get("world_id", "default"),
                prompt_data.positive,
                prompt_data.negative
            )

            # Apply runtime preset defaults (best-effort)
            width = int(self.default_width)
            height = int(self.default_height)
            steps = int(self.default_steps)
            guidance_scale = float(self.default_cfg_scale)
            preset_opt: Dict[str, Any] = {}
            try:
                preset_id = str(scene_context.get("runtime_preset_id") or "").strip()
                if preset_id:
                    from core.runtime.catalog import get_runtime_preset

                    preset = get_runtime_preset(preset_id) or {}
                    t2i = preset.get("t2i") if isinstance(preset.get("t2i"), dict) else {}

                    width = int(t2i.get("default_width", width) or width)
                    height = int(t2i.get("default_height", height) or height)
                    steps = int(t2i.get("default_steps", steps) or steps)
                    guidance_scale = float(t2i.get("default_guidance_scale", guidance_scale) or guidance_scale)

                    max_w = int(t2i.get("max_width", width) or width)
                    max_h = int(t2i.get("max_height", height) or height)
                    max_steps = int(t2i.get("max_steps", steps) or steps)
                    width = max(256, min(width, max_w))
                    height = max(256, min(height, max_h))
                    steps = max(1, min(steps, max_steps))

                    # If world didn't force base_model, allow preset hint.
                    if not model_id and t2i.get("model_id"):
                        model_id = str(t2i.get("model_id"))

                    preset_opt = {
                        "enable_attention_slicing": bool(t2i.get("enable_attention_slicing", True)),
                        "enable_vae_slicing": bool(t2i.get("enable_vae_slicing", True)),
                        "enable_vae_tiling": bool(t2i.get("enable_vae_tiling", False)),
                        "enable_cpu_offload": bool(t2i.get("enable_cpu_offload", False)),
                        "enable_sequential_cpu_offload": bool(t2i.get("enable_sequential_cpu_offload", False)),
                    }
            except Exception as exc:  # noqa: BLE001
                logger.debug("Runtime preset skipped: %s", exc)

            logger.info(f"Generating scene image: {positive_prompt[:100]}...")

            request_payload: Dict[str, Any] = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
            }
            if preset_opt:
                request_payload.update(preset_opt)
            if model_id:
                request_payload["model_id"] = model_id
            if lora_configs:
                request_payload["lora_configs"] = lora_configs

            result = await self.t2i_engine.txt2img(request_payload)
            return self._process_t2i_result(result, positive_prompt, negative_prompt)

        except Exception as e:
            logger.error(f"Failed to generate scene image: {e}", exc_info=True)
            return None

    async def generate_character_portrait_image(
        self,
        character_name: str,
        appearance_desc: str,
        world_id: str = "default",
        visual_style: Optional[str] = None
    ) -> Optional[SceneImageResult]:
        """
        Generate character portrait image (sprite)

        Args:
            character_name: Name of the character
            appearance_desc: Physical appearance description
            world_id: World ID for style consistency
            visual_style: Optional visual style hints

        Returns:
            SceneImageResult if image was generated, None otherwise
        """
        try:
            # Generate prompt for character portrait
            prompt_data = await self.prompt_generator.generate_character_portrait(
                character_name=character_name,
                appearance_desc=appearance_desc,
                visual_style=visual_style
            )

            positive_prompt, negative_prompt, model_id, lora_configs = self._apply_world_style(
                world_id,
                prompt_data.positive,
                prompt_data.negative
            )

            logger.info(f"Generating character portrait for {character_name}: {positive_prompt[:100]}...")

            request_payload: Dict[str, Any] = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": 512,
                "height": 768,
                "num_inference_steps": self.default_steps,
                "guidance_scale": self.default_cfg_scale,
            }
            if model_id:
                request_payload["model_id"] = model_id
            if lora_configs:
                request_payload["lora_configs"] = lora_configs

            result = await self.t2i_engine.txt2img(request_payload)
            return self._process_t2i_result(result, positive_prompt, negative_prompt)

        except Exception as e:
            logger.error(f"Failed to generate character portrait: {e}", exc_info=True)
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
            request_payload: Dict[str, Any] = {
                "prompt": modified_prompt.positive,
                "negative_prompt": modified_prompt.negative,
                "width": base_scene_image.width,
                "height": base_scene_image.height,
                "num_inference_steps": self.default_steps,
                "guidance_scale": self.default_cfg_scale,
            }
            if base_scene_image.seed is not None:
                request_payload["seed"] = base_scene_image.seed

            result = await self.t2i_engine.txt2img(request_payload)

            images_base64 = result.get("images") or []
            metadata = result.get("metadata", {}) or {}
            output_paths = metadata.get("output_paths") or []

            image_url = ""
            if output_paths:
                try:
                    from urllib.parse import quote
                    from pathlib import Path
                    from core.shared_cache import get_shared_cache

                    cache = get_shared_cache()
                    root = Path(cache.get_path("OUTPUT_DIR")).resolve()
                    rel = Path(str(output_paths[0])).resolve().relative_to(root)
                    image_url = f"/api/v1/t2i/file?path={quote(str(rel))}"
                except Exception:
                    image_url = ""
            if (not image_url) and images_base64:
                image_url = f"data:image/png;base64,{images_base64[0]}"
            if not image_url:
                image_url = result.get("url") or result.get("image_url") or ""

            return SceneImageResult(
                image_url=image_url,
                prompt=modified_prompt.positive,
                negative_prompt=modified_prompt.negative,
                generation_time=float(metadata.get("generation_time", 0.0) or 0.0),
                seed=metadata.get("seed", base_scene_image.seed),
                width=int(metadata.get("parameters", {}).get("width", base_scene_image.width) or base_scene_image.width),
                height=int(metadata.get("parameters", {}).get("height", base_scene_image.height) or base_scene_image.height),
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
