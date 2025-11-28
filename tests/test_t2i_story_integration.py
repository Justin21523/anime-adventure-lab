"""
Tests for T2I Story Integration

IMPORTANT: All tests use mocks to avoid GPU usage during testing.
Never load real models in these tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.story.t2i_integration import StoryT2IIntegration, SceneImageResult
from core.t2i.story_prompt_generator import StoryPromptGenerator, T2IPrompt


# Mock Fixtures ---------------------------------------------------------------

@pytest.fixture
def mock_t2i_engine():
    """Mock T2I Engine to avoid GPU usage"""
    mock = AsyncMock()
    mock.generate.return_value = {
        "url": "http://localhost/mock/forest_night.png",
        "image_url": "http://localhost/mock/forest_night.png",
        "time": 2.5,
        "generation_time": 2.5,
        "seed": 42,
        "width": 768,
        "height": 768
    }
    return mock


@pytest.fixture
def mock_prompt_generator():
    """Mock prompt generator"""
    mock = AsyncMock()
    mock.generate_from_scene.return_value = T2IPrompt(
        positive="dark forest, night, mysterious atmosphere, anime style, high quality",
        negative="low quality, blurry, distorted, ugly, nsfw"
    )
    mock.modify_prompt.return_value = T2IPrompt(
        positive="dark forest, dawn light, mysterious atmosphere, anime style, high quality",
        negative="low quality, blurry, distorted, ugly, nsfw"
    )
    return mock


@pytest.fixture
def t2i_integration(mock_t2i_engine, mock_prompt_generator):
    """T2I integration instance with mocked dependencies"""
    integration = StoryT2IIntegration(
        t2i_engine=mock_t2i_engine,
        prompt_generator=mock_prompt_generator
    )
    return integration


# Scene Context Fixtures ------------------------------------------------------

@pytest.fixture
def basic_scene_context():
    """Basic scene context"""
    return {
        "location": "dark forest",
        "time": "night",
        "atmosphere": "mysterious",
        "characters": ["hero", "guide"],
        "scene_transition": False,
        "is_major_event": False,
        "weather": ""
    }


@pytest.fixture
def scene_transition_context():
    """Scene context with transition flag"""
    return {
        "location": "ancient castle",
        "time": "evening",
        "atmosphere": "tense",
        "characters": ["warrior"],
        "scene_transition": True,
        "is_major_event": False,
        "weather": "rain"
    }


@pytest.fixture
def major_event_context():
    """Scene context for major event"""
    return {
        "location": "dragon lair",
        "time": "afternoon",
        "atmosphere": "dangerous",
        "characters": ["hero", "dragon"],
        "scene_transition": False,
        "is_major_event": True,
        "weather": "clear"
    }


# Trigger Detection Tests -----------------------------------------------------

class TestTriggerDetection:
    """Test scene image generation trigger logic"""

    def test_scene_transition_triggers(self, t2i_integration, scene_transition_context):
        """Scene transitions should trigger image generation"""
        narrative = "You continue your journey."
        assert t2i_integration._should_generate(scene_transition_context, narrative) is True

    def test_major_event_triggers(self, t2i_integration, major_event_context):
        """Major events should trigger image generation"""
        narrative = "The battle continues."
        assert t2i_integration._should_generate(major_event_context, narrative) is True

    def test_keyword_triggers_chinese(self, t2i_integration, basic_scene_context):
        """Chinese keywords should trigger generation"""
        narrative = "你進入了黑暗的森林深處。"
        assert t2i_integration._should_generate(basic_scene_context, narrative) is True

    def test_keyword_triggers_english(self, t2i_integration, basic_scene_context):
        """English keywords should trigger generation"""
        narrative = "You enter the dark forest."
        assert t2i_integration._should_generate(basic_scene_context, narrative) is True

    def test_no_trigger_normal_action(self, t2i_integration, basic_scene_context):
        """Normal actions without triggers should not generate"""
        narrative = "You examine your surroundings carefully."
        assert t2i_integration._should_generate(basic_scene_context, narrative) is False


# Image Generation Tests ------------------------------------------------------

class TestSceneImageGeneration:
    """Test scene image generation with mocked T2I engine"""

    @pytest.mark.asyncio
    async def test_generate_with_scene_transition(
        self,
        t2i_integration,
        mock_t2i_engine,
        mock_prompt_generator,
        scene_transition_context
    ):
        """Test generation triggered by scene transition"""
        narrative = "You arrive at the ancient castle gates."

        result = await t2i_integration.generate_scene_image(
            scene_context=scene_transition_context,
            narrative_text=narrative,
            force=False
        )

        # Verify result
        assert result is not None
        assert isinstance(result, SceneImageResult)
        assert result.image_url == "http://localhost/mock/forest_night.png"
        assert result.generation_time == 2.5
        assert result.seed == 42
        assert result.width == 768
        assert result.height == 768

        # Verify prompt generator was called
        mock_prompt_generator.generate_from_scene.assert_called_once()
        call_args = mock_prompt_generator.generate_from_scene.call_args[0][0]
        assert call_args["location"] == "ancient castle"
        assert call_args["time"] == "evening"
        assert call_args["atmosphere"] == "tense"

        # Verify T2I engine was called
        mock_t2i_engine.generate.assert_called_once()
        engine_call = mock_t2i_engine.generate.call_args[1]
        assert "dark forest, night" in engine_call["prompt"]
        assert engine_call["width"] == 768
        assert engine_call["height"] == 768

    @pytest.mark.asyncio
    async def test_skip_generation_no_trigger(
        self,
        t2i_integration,
        mock_t2i_engine,
        basic_scene_context
    ):
        """Test that generation is skipped when no triggers match"""
        narrative = "You examine the area."

        result = await t2i_integration.generate_scene_image(
            scene_context=basic_scene_context,
            narrative_text=narrative,
            force=False
        )

        # Should return None (skipped)
        assert result is None

        # T2I engine should NOT be called
        mock_t2i_engine.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_generation(
        self,
        t2i_integration,
        mock_t2i_engine,
        basic_scene_context
    ):
        """Test forced generation regardless of triggers"""
        narrative = "You wait quietly."

        result = await t2i_integration.generate_scene_image(
            scene_context=basic_scene_context,
            narrative_text=narrative,
            force=True  # Force generation
        )

        # Should generate despite no triggers
        assert result is not None
        mock_t2i_engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_error_handling(
        self,
        t2i_integration,
        mock_t2i_engine,
        scene_transition_context
    ):
        """Test that errors are handled gracefully"""
        # Make T2I engine raise exception
        mock_t2i_engine.generate.side_effect = Exception("Mock T2I failure")

        narrative = "You enter the castle."

        result = await t2i_integration.generate_scene_image(
            scene_context=scene_transition_context,
            narrative_text=narrative,
            force=False
        )

        # Should return None on error, not raise
        assert result is None


# Prompt Generator Tests ------------------------------------------------------

class TestStoryPromptGenerator:
    """Test story prompt generation logic"""

    @pytest.mark.asyncio
    async def test_generate_basic_prompt(self):
        """Test basic prompt generation from scene context"""
        generator = StoryPromptGenerator()

        scene_context = {
            "location": "dark forest",
            "time": "night",
            "atmosphere": "mysterious",
            "characters": [],
            "weather": ""
        }

        prompt = await generator.generate_from_scene(scene_context)

        assert isinstance(prompt, T2IPrompt)
        assert "forest" in prompt.positive.lower()
        assert "night" in prompt.positive.lower()
        assert "mysterious" in prompt.positive.lower()
        assert "anime style" in prompt.positive.lower()
        assert "low quality" in prompt.negative.lower()

    @pytest.mark.asyncio
    async def test_generate_with_weather(self):
        """Test prompt includes weather conditions"""
        generator = StoryPromptGenerator()

        scene_context = {
            "location": "mountain peak",
            "time": "morning",
            "atmosphere": "epic",
            "weather": "heavy snow",
            "characters": []
        }

        prompt = await generator.generate_from_scene(scene_context)

        assert "mountain" in prompt.positive.lower()
        assert "morning" in prompt.positive.lower()
        assert "heavy snow" in prompt.positive.lower()

    @pytest.mark.asyncio
    async def test_generate_with_special_elements(self):
        """Test prompt includes special elements"""
        generator = StoryPromptGenerator()

        scene_context = {
            "location": "throne room",
            "time": "afternoon",
            "atmosphere": "tense",
            "special_elements": ["golden throne", "magical artifacts", "stained glass windows"],
            "characters": []
        }

        prompt = await generator.generate_from_scene(scene_context)

        assert "golden throne" in prompt.positive
        assert "magical artifacts" in prompt.positive
        assert "stained glass windows" in prompt.positive

    @pytest.mark.asyncio
    async def test_modify_prompt_atmosphere(self):
        """Test modifying prompt atmosphere"""
        generator = StoryPromptGenerator()

        base_prompt = "dark forest, night, mysterious, anime style, high quality"
        modifications = {"atmosphere": "peaceful"}

        modified = await generator.modify_prompt(base_prompt, modifications)

        assert "peaceful" in modified.positive.lower()
        assert "mysterious" not in modified.positive.lower()

    @pytest.mark.asyncio
    async def test_modify_prompt_time(self):
        """Test modifying prompt time of day"""
        generator = StoryPromptGenerator()

        base_prompt = "dark forest, night scene, moonlight, mysterious, anime style"
        modifications = {"time": "morning"}

        modified = await generator.modify_prompt(base_prompt, modifications)

        assert "morning" in modified.positive.lower()
        # Old time-related tags should be removed
        assert "night" not in modified.positive.lower()
        assert "moonlight" not in modified.positive.lower()

    @pytest.mark.asyncio
    async def test_add_character_to_prompt(self):
        """Test adding character to scene prompt"""
        generator = StoryPromptGenerator()

        base_prompt = "castle interior, evening light, tense, anime style"
        character_desc = "female warrior in armor, red cape, determined expression"

        modified = generator.add_character_to_prompt(base_prompt, character_desc)

        # Character should be at the beginning
        assert modified.positive.startswith(character_desc)
        assert "castle interior" in modified.positive


# Regeneration Tests ----------------------------------------------------------

class TestSceneRegeneration:
    """Test scene image regeneration with modifications"""

    @pytest.mark.asyncio
    async def test_regenerate_with_modifications(
        self,
        t2i_integration,
        mock_t2i_engine,
        mock_prompt_generator
    ):
        """Test regenerating scene with modifications"""
        base_scene = SceneImageResult(
            image_url="http://localhost/mock/original.png",
            prompt="dark forest, night, mysterious",
            negative_prompt="low quality",
            generation_time=2.0,
            seed=42,
            width=768,
            height=768
        )

        modifications = {
            "atmosphere": "peaceful",
            "time": "morning"
        }

        result = await t2i_integration.regenerate_with_modifications(
            base_scene_image=base_scene,
            modifications=modifications
        )

        # Verify prompt generator was called with modifications
        mock_prompt_generator.modify_prompt.assert_called_once()

        # Verify T2I engine was called with same seed for consistency
        mock_t2i_engine.generate.assert_called_once()
        engine_call = mock_t2i_engine.generate.call_args[1]
        assert engine_call["seed"] == 42  # Same seed as base
        assert engine_call["width"] == 768
        assert engine_call["height"] == 768

        # Verify result
        assert result is not None
        assert isinstance(result, SceneImageResult)


# Integration Tests with Story Router ----------------------------------------

class TestStoryRouterIntegration:
    """Test integration between story router and T2I system"""

    @pytest.mark.asyncio
    async def test_scene_image_in_turn_response(self):
        """Test that scene_image is included in StoryTurnResponse"""
        from schemas.story import StoryTurnResponse, SceneImage

        # Create sample response with scene image
        response = StoryTurnResponse(
            session_id="test-session",
            turn_count=1,
            narrative="You enter the dark forest at nightfall.",
            choices=[{"text": "Continue deeper", "choice_id": "deeper"}],
            stats={"hp": 100, "mp": 50},
            inventory=["sword", "potion"],
            scene_id="forest_01",
            scene_image=SceneImage(
                image_url="http://localhost/mock/scene.png",
                prompt="dark forest, night, mysterious",
                negative_prompt="low quality",
                generation_time=2.5,
                seed=42,
                width=768,
                height=768
            )
        )

        # Verify schema
        assert response.scene_image is not None
        assert response.scene_image.image_url == "http://localhost/mock/scene.png"
        assert response.scene_image.generation_time == 2.5

    @pytest.mark.asyncio
    async def test_optional_scene_image(self):
        """Test that scene_image is optional in StoryTurnResponse"""
        from schemas.story import StoryTurnResponse

        # Create response without scene image
        response = StoryTurnResponse(
            session_id="test-session",
            turn_count=1,
            narrative="You examine the area.",
            choices=[],
            stats={"hp": 100},
            inventory=[],
        )

        # Should be valid with None scene_image
        assert response.scene_image is None


# Singleton Tests -------------------------------------------------------------

def test_t2i_integration_singleton():
    """Test that get_t2i_integration returns singleton"""
    from core.story.t2i_integration import get_t2i_integration

    instance1 = get_t2i_integration()
    instance2 = get_t2i_integration()

    assert instance1 is instance2
