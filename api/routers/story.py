# api/routers/story.py
"""
Story Engine Router (separate from game.py)
"""

import logging
from fastapi import APIRouter, HTTPException
from core.story.engine import get_story_engine
from schemas.game import GamePersonaInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/story/personas")
async def list_story_personas():
    """List available story personas"""
    try:
        story_engine = get_story_engine()
        personas = story_engine.get_personas()

        return {
            persona_id: GamePersonaInfo(**persona_data)
            for persona_id, persona_data in personas.items()
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to list personas: {str(e)}")


@router.get("/story/templates")
async def list_story_templates():
    """List available story templates"""
    try:
        # Mock story templates
        return {
            "templates": [
                {
                    "id": "hero_journey",
                    "name": "英雄旅程",
                    "description": "經典的英雄成長故事結構",
                    "themes": ["成長", "冒險", "友誼"],
                    "difficulty": "normal",
                },
                {
                    "id": "mystery_solve",
                    "name": "推理解謎",
                    "description": "偵探推理類故事模板",
                    "themes": ["推理", "懸疑", "真相"],
                    "difficulty": "hard",
                },
            ]
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to list templates: {str(e)}")
