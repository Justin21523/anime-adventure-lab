# api/routers/game.py
from fastapi import APIRouter, HTTPException
from api.schemas import GameStartRequest, GameStepRequest, GameResponse
import uuid
from typing import Dict, List, Any, Optional

router = APIRouter(prefix="/game", tags=["Game"])

# NOTE: In-memory sessions for now; consider moving to core.story engine later.
game_sessions: Dict[str, dict] = {}


@router.post("/new", response_model=GameResponse)
async def start_new_game(request: GameStartRequest):
    """Start a new text adventure game"""
    try:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "persona": request.persona,
            "setting": request.setting,
            "difficulty": request.difficulty,
            "status": "active",
            "turn": 1,
            "inventory": ["basic clothes", "small pouch"],
            "health": 100,
            "history": [],
        }
        scene = (
            f"Welcome to the {request.setting} adventure! "
            f"You are a traveler who has just arrived at a mysterious village. "
            f"The {request.persona} persona guides your journey. What would you like to do?"
        )
        choices = [
            "Explore the village center",
            "Talk to the local villagers",
            "Check the nearby forest",
            "Rest at the tavern",
        ]
        session["current_scene"] = scene
        session["available_choices"] = choices
        game_sessions[session_id] = session

        return GameResponse(
            session_id=session_id,
            scene=scene,
            choices=choices,
            status="active",
            inventory=session["inventory"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step", response_model=GameResponse)
async def game_step(request: GameStepRequest):
    """Take an action in the game"""
    try:
        if request.session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")

        session = game_sessions[request.session_id]
        session["turn"] += 1
        session["history"].append(
            {
                "turn": session["turn"] - 1,
                "action": request.action,
            }
        )

        # Simple response generation based on action
        if "explore" in request.action.lower():
            scene = f"You explore the area and discover interesting landmarks. Your {session['persona']} intuition guides you to notice hidden details."
            choices = [
                "Continue exploring",
                "Return to safety",
                "Investigate further",
                "Rest",
            ]
        elif "talk" in request.action.lower():
            scene = "You approach the villagers. They seem friendly but speak in whispers about strange occurrences."
            choices = [
                "Ask about the occurrences",
                "Offer help",
                "Leave politely",
                "Buy supplies",
            ]
        elif "forest" in request.action.lower():
            scene = "The forest is dark and mysterious. You hear strange sounds in the distance."
            choices = ["Enter deeper", "Turn back", "Listen carefully", "Light a torch"]
        else:
            scene = f"You decide to {request.action}. The world responds to your choice in unexpected ways."
            choices = ["Continue", "Reconsider", "Try something else", "Take a break"]

        session["current_scene"] = scene
        session["available_choices"] = choices

        return GameResponse(
            session_id=request.session_id,
            scene=scene,
            choices=choices,
            status=session["status"],
            inventory=session["inventory"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
