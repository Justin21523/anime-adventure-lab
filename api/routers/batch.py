from fastapi import APIRouter
router = APIRouter()

@router.get("/batch/status/{job_id}")
def status(job_id: str): return {"job_id": job_id, "status": "stub"}
