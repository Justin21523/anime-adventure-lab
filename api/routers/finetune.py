from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class LoraJob(BaseModel):
    config_path: str
    run_id: str
    notes: str | None = None

@router.post("/finetune/lora")
def submit_lora(job: LoraJob):
    return {"accepted": True, "job_id": job.run_id}
