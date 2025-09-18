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
    GameStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/game/new", response_model=GameResponse)
async def new_game_session(request: NewGameRequest):
    """Create new text adventure game session"""

    try:
        story_engine = get_story_engine()

        # Extract parameters with defaults
        params = request.parameters or GameParameters()

        # Create new game session
        session = story_engine.create_session(
            player_name=request.player_name,
            persona_id=params.persona_id,
            setting=params.setting,
            difficulty=params.difficulty,
        )

        # Generate opening scene
        opening_response = await story_engine.process_turn(
            session_id=session.session_id,
            player_input=f"開始遊戲，玩家名稱：{request.player_name}",
            choice_id=None,
        )

        return GameResponse(  # type: ignore
            session_id=session.session_id,
            turn_count=opening_response["turn_count"],
            narrative=opening_response["narrative"],
            choices=opening_response["choices"],
            stats=opening_response["stats"],
            inventory=opening_response["inventory"],
            scene_id=opening_response["scene_id"],
            flags=opening_response.get("flags", {}),
            success=True,
            message="遊戲會話創建成功",
        )

    except GameError as e:
        logger.error(f"Game error in new_game_session: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in new_game_session: {e}")
        raise HTTPException(status_code=500, detail="內部伺服器錯誤")


@router.post("/game/step", response_model=GameResponse)
async def game_step(request: GameStepRequest):
    """Take an action in the game"""

    try:
        story_engine = get_story_engine()

        # Process the turn
        response = await story_engine.process_turn(
            session_id=request.session_id,
            player_input=request.player_input,
            choice_id=request.choice_id,
        )

        return GameResponse(  # type: ignore
            session_id=response["session_id"],
            turn_count=response["turn_count"],
            narrative=response["narrative"],
            choices=response["choices"],
            stats=response["stats"],
            inventory=response["inventory"],
            scene_id=response["scene_id"],
            flags=response.get("flags", {}),
            choice_result=response.get("choice_result"),
            success=True,
            message="回合處理成功",
        )

    except (SessionNotFoundError, InvalidChoiceError) as e:
        logger.error(f"Game error in game_step: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except GameError as e:
        logger.error(f"Game error in game_step: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in game_step: {e}")
        raise HTTPException(status_code=500, detail="內部伺服器錯誤")


@router.get("/game/sessions", response_model=List[GameSessionSummary])
async def list_game_sessions(active_only: bool = True):
    """List game sessions"""
    try:
        story_engine = get_story_engine()
        sessions = story_engine.list_sessions(active_only=active_only)

        return [GameSessionSummary(**session) for session in sessions]

    except Exception as e:
        logger.error(f"Error in list_game_sessions: {e}")
        raise HTTPException(status_code=500, detail="無法獲取遊戲會話列表")


@router.get("/game/session/{session_id}", response_model=GameSessionSummary)
async def get_game_session(session_id: str):
    """Get game session details"""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session_summary(session_id)

        return GameSessionSummary(**session)

    except SessionNotFoundError as e:
        logger.error(f"Session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_game_session: {e}")
        raise HTTPException(status_code=500, detail="無法獲取遊戲會話")


@router.delete("/game/session/{session_id}")
async def end_game_session(session_id: str):
    """End a game session"""
    try:
        story_engine = get_story_engine()
        story_engine.end_session(session_id)

        return {"success": True, "message": f"遊戲會話 {session_id} 已結束"}

    except SessionNotFoundError as e:
        logger.error(f"Session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in end_game_session: {e}")
        raise HTTPException(status_code=500, detail="無法結束遊戲會話")


@router.get("/game/personas", response_model=List[GamePersonaInfo])
async def list_personas():
    """List available game personas"""
    try:
        story_engine = get_story_engine()
        personas = story_engine.list_personas()

        return [GamePersonaInfo(**persona) for persona in personas]

    except Exception as e:
        logger.error(f"Error in list_personas: {e}")
        raise HTTPException(status_code=500, detail="無法獲取角色列表")


@router.get("/game/stats/{session_id}", response_model=GameStatsResponse)
async def get_game_stats(session_id: str):
    """Get detailed game statistics"""
    try:
        story_engine = get_story_engine()
        session = story_engine.get_session(session_id)

        return GameStatsResponse(  # type: ignore
            session_id=session.session_id,
            player_name=session.player_name,
            stats=session.stats.to_dict(),
            inventory=session.inventory,
            turn_count=session.turn_count,
            flags=session.current_state.flags,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
        )

    except SessionNotFoundError as e:
        logger.error(f"Session not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_game_stats: {e}")
        raise HTTPException(status_code=500, detail="無法獲取遊戲統計")
