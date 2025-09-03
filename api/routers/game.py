# api/routers/game.py
"""
Text Adventure Game API
Story-driven interactive game endpoints
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

from core.story.engine import get_story_engine
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from schemas.game import (
    NewGameRequest,
    GameStepRequest,
    GameResponse,
    GameSessionSummary,
    GamePersonaInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/game/new", response_model=GameResponse)
async def new_game_session(request: NewGameRequest):
    """
    Create new text adventure game session

    - **player_name**: Player character name
    - **persona_id**: Game master persona (default: "default")
    - **setting**: Game world setting (fantasy/sci-fi/modern)
    - **difficulty**: Game difficulty (easy/normal/hard)
    """

    try:
        story_engine = get_story_engine()

        # Create new game session
        game_state = story_engine.create_session(
            player_name=request.player_name,
            persona_id=request.persona_id,
            setting=request.setting,
            difficulty=request.difficulty,
        )

        # Generate opening scene
        opening_response = story_engine.process_turn(
            session_id=game_state.session_id,
            player_input=f"開始遊戲，玩家名稱：{request.player_name}",
            choice_id=None,
        )

        return GameResponse(
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
    """
    Take an action in the game

    - **session_id**: Game session identifier
    - **action**: Player action description
    - **choice_id**: Optional predefined choice ID
    """

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

        return GameResponse(
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


@router.get("/game/{session_id}", response_model=GameSessionSummary)
async def get_game_session(session_id: str):
    """Get current game session status"""

    try:
        story_engine = get_story_engine()
        summary = story_engine.get_session_summary(session_id)

        return GameSessionSummary(**summary)

    except SessionNotFoundError:
        raise HTTPException(404, f"Game session not found: {session_id}")

    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(500, "Failed to retrieve session")


@router.get("/game/sessions", response_model=List[GameSessionSummary])
async def list_game_sessions():
    """List all game sessions (active and saved)"""

    try:
        story_engine = get_story_engine()
        sessions = story_engine.list_sessions()

        return [GameSessionSummary(**session) for session in sessions]

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(500, "Failed to list game sessions")


@router.delete("/game/{session_id}")
async def delete_game_session(session_id: str):
    """Delete game session"""

    try:
        story_engine = get_story_engine()

        if story_engine.delete_session(session_id):
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(404, f"Session {session_id} not found")

    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(500, "Failed to delete session")


@router.post("/game/{session_id}/save")
async def save_game_session(session_id: str):
    """Save game session to persistent storage"""

    try:
        story_engine = get_story_engine()

        if story_engine.save_session(session_id):
            return {"message": f"Session {session_id} saved successfully"}
        else:
            raise HTTPException(404, f"Session {session_id} not found")

    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}")
        raise HTTPException(500, "Failed to save session")


@router.post("/game/{session_id}/load")
async def load_game_session(session_id: str):
    """Load game session from persistent storage"""

    try:
        story_engine = get_story_engine()

        if story_engine.load_session(session_id):
            summary = story_engine.get_session_summary(session_id)
            return {
                "message": f"Session {session_id} loaded successfully",
                "session": summary,
            }
        else:
            raise HTTPException(404, f"Saved session {session_id} not found")

    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        raise HTTPException(500, "Failed to load session")


@router.get("/game/personas", response_model=Dict[str, GamePersonaInfo])
async def list_game_personas():
    """List available game personas"""

    try:
        story_engine = get_story_engine()
        personas = story_engine.get_personas()

        # Convert to response format
        return {
            persona_id: GamePersonaInfo(**persona_data)
            for persona_id, persona_data in personas.items()
        }

    except Exception as e:
        logger.error(f"Failed to list personas: {e}")
        raise HTTPException(500, "Failed to retrieve personas")


@router.get("/game/{session_id}/inventory")
async def get_player_inventory(session_id: str):
    """Get player's current inventory"""

    try:
        story_engine = get_story_engine()
        game_state = story_engine.get_session(session_id)

        return {
            "session_id": session_id,
            "inventory": game_state.inventory,
            "inventory_count": len(game_state.inventory),
            "stats": game_state.stats,
        }

    except SessionNotFoundError:
        raise HTTPException(404, f"Session {session_id} not found")

    except Exception as e:
        logger.error(f"Failed to get inventory for {session_id}: {e}")
        raise HTTPException(500, "Failed to retrieve inventory")
