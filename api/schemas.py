# api/schemas.py - Unified Pydantic models
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


# === Common Types ===
class ModelType(str, Enum):
    sd15 = "runwayml/stable-diffusion-v1-5"
    sdxl = "stabilityai/stable-diffusion-xl-base-1.0"


class ControlNetType(str, Enum):
    pose = "pose"
    depth = "depth"
    canny = "canny"
    lineart = "lineart"


# === T2I Schemas ===
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(768, ge=256, le=2048)
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    model: ModelType = ModelType.sd15
    lora_ids: List[str] = []
    lora_scales: List[float] = []
    safety_check: bool = True


class T2IResponse(BaseModel):
    image_path: str
    metadata_path: str
    seed: int
    elapsed_ms: int
    model_used: str


# === ControlNet Schemas ===
class ControlNetRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    image_path: Optional[str] = None  # condition image
    image_base64: Optional[str] = None
    controlnet_type: ControlNetType = ControlNetType.pose
    strength: float = Field(1.0, ge=0.0, le=2.0)
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(768, ge=256, le=2048)
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    lora_ids: List[str] = []


class ControlNetResponse(BaseModel):
    image_path: str
    metadata_path: str
    condition_summary: str
    seed: int
    elapsed_ms: int


# === LoRA Schemas ===
class LoRAInfo(BaseModel):
    id: str
    name: str
    model_type: str  # sd15/sdxl
    rank: int
    path: str
    loaded: bool = False
    weight: float = 1.0


class LoRAListResponse(BaseModel):
    loras: List[LoRAInfo]


class LoRALoadRequest(BaseModel):
    lora_id: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


# === Chat Schemas ===
class ChatMessage(BaseModel):
    role: str  # user/assistant
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_length: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    model: str = "qwen-chat"


class ChatResponse(BaseModel):
    message: str
    model_used: str
    usage: Dict[str, int] = {}


# === VQA Schemas ===
class VQARequest(BaseModel):
    question: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    max_length: int = Field(128, ge=1, le=512)
    model: str = "llava-1.5"


class VQAResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    model_used: str


# === Caption Schemas ===
class CaptionRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    max_length: int = Field(50, ge=1, le=200)
    num_beams: int = Field(3, ge=1, le=10)


class CaptionResponse(BaseModel):
    caption: str
    confidence: float
    model_used: str


# === RAG Schemas ===
class RAGRequest(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = True


class RAGSource(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class RAGResponse(BaseModel):
    answer: str
    sources: List[RAGSource]
    model_used: str


# === Game Schemas ===
class GameStartRequest(BaseModel):
    persona: str = "default"
    setting: str = "fantasy"
    difficulty: str = "normal"


class GameStepRequest(BaseModel):
    session_id: str
    action: str


class GameResponse(BaseModel):
    session_id: str
    scene: str
    choices: List[str]
    status: str  # active/paused/ended
    inventory: List[str] = []


# === Batch Schemas ===
class BatchJobRequest(BaseModel):
    job_type: str  # t2i/chat/caption
    tasks: List[Dict[str, Any]]
    callback_url: Optional[str] = None


class BatchJobStatus(BaseModel):
    job_id: str
    status: str  # queued/running/completed/failed
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    created_at: str
    updated_at: str
