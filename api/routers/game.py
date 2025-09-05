# api/routers/game.py
"""
Text Adventure Game Router
Story-driven interactive game endpoints
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict

from core.story.engine import get_story_engine
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from schemas.game import (
    NewGameRequest,
    GameStepRequest,
    GameResponse,
    GameSessionSummary,
    GamePersonaInfo,
    GameParameters,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/game/new", response_model=GameResponse)
async def new_game_session(request: NewGameRequest):
    """Create new text adventure game session"""

    try:
        story_engine = get_story_engine()

        # Extract parameters with defaults
        params = request.parameters or GameParameters()  # type: ignore

        # Create new game session
        game_state = story_engine.create_session(
            player_name=request.player_name,
            persona_id=params.persona_id,
            setting=params.setting,
            difficulty=params.difficulty,
        )

        # Generate opening scene
        opening_response = story_engine.process_turn(
            session_id=game_state.session_id,
            player_input=f"開始遊戲，玩家名稱：{request.player_name}",
            choice_id=None,
        )

        return GameResponse(  # type: ignore
            session_id=game_state.session_id,
            turn_count=game_state.turn_count,
            scene=game_state.current_scene,
            narration=opening_response.narration,
            dialogues=opening_response.dialogues,
            choices=[
                {
                    "id": choice.id,
                    "text": choice.text,
                    "description": choice.description,
                }
                for choice in opening_response.choices
            ],
            parameters=params,
            game_state={
                "inventory": game_state.inventory,
                "stats": game_state.stats,
                "flags": {k: v for k, v in game_state.flags.items() if v},
            },
            status="active",
        )

    except GameError as e:
        logger.error(f"Game creation failed: {e}")
        raise HTTPException(400, f"Failed to create game: {e.message}")

    except Exception as e:
        logger.error(f"Unexpected error in new game: {e}", exc_info=True)
        raise HTTPException(500, "Failed to create new game session")


@router.post("/game/step", response_model=GameResponse)
async def game_step(request: GameStepRequest):
    """Take an action in the game"""

    try:
        story_engine = get_story_engine()

        # Process the turn
        turn_response = story_engine.process_turn(
            session_id=request.session_id,
            player_input=request.action,
            choice_id=request.choice_id,
        )

        # Get updated game state
        game_state = story_engine.get_session(request.session_id)

        # Use request parameters or create defaults
        params = request.parameters or GameParameters()  # type: ignore

        return GameResponse(  # type: ignore
            session_id=request.session_id,
            turn_count=game_state.turn_count,
            scene=game_state.current_scene,
            narration=turn_response.narration,
            dialogues=turn_response.dialogues,
            choices=[
                {
                    "id": choice.id,
                    "text": choice.text,
                    "description": choice.description,
                }
                for choice in turn_response.choices
            ],
            parameters=params,
            game_state={
                "inventory": game_state.inventory,
                "stats": game_state.stats,
                "flags": {k: v for k, v in game_state.flags.items() if v},
                "relationships": game_state.relationships,
            },
            status="active",
        )

    except SessionNotFoundError as e:
        raise HTTPException(404, f"Game session not found: {request.session_id}")

    except InvalidChoiceError as e:
        raise HTTPException(400, f"Invalid choice: {e.message}")

    except GameError as e:
        logger.error(f"Game step failed: {e}")
        raise HTTPException(500, f"Game action failed: {e.message}")

    except Exception as e:
        logger.error(f"Unexpected error in game step: {e}", exc_info=True)
        raise HTTPException(500, "Game action processing failed")
