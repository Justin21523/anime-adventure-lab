# workers/tasks.py
from celery import current_task
from celery.exceptions import WorkerLostError
import torch
import os
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import uuid


from workers.celery_app import celery_app, redis_client
from core.vlm.engine import VLMEngine
from core.llm.adapter import LLMAdapter
from core.monitoring.logger import structured_logger

# Initialize engines (will be loaded lazily)
vlm_engine = None
llm_adapter = None


def get_vlm_engine():
    global vlm_engine
    if vlm_engine is None:
        vlm_engine = VLMEngine()
    return vlm_engine


def get_llm_adapter():
    global llm_adapter
    if llm_adapter is None:
        llm_adapter = LLMAdapter()
    return llm_adapter


def update_task_progress(
    task_id: str, processed: int, total: int, results: Optional[Dict] = None
):
    """Update task progress in Redis"""
    progress_data = {
        "processed_items": processed,
        "total_items": total,
        "progress_percent": (processed / total) * 100 if total > 0 else 0,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if results:
        progress_data["partial_results"] = results

    redis_client.setex(f"task_progress:{task_id}", 3600, json.dumps(progress_data))


@celery_app.task(bind=True, name="batch_caption_task")
def batch_caption_task(
    self, job_id: str, image_paths: List[str], config: Dict[str, Any]
):
    """Batch image captioning task"""
    try:
        task_id = self.request.id
        total_items = len(image_paths)
        processed_items = 0
        results = []
        failed_items = []

        # Update initial progress
        update_task_progress(task_id, 0, total_items)

        # Get VLM engine
        engine = get_vlm_engine()

        # Load caption model if not already loaded
        if not engine.caption_model:
            engine.load_caption_model()

        for i, image_path in enumerate(image_paths):
            try:
                # Caption single image
                result = engine.caption_image(
                    image_path=image_path,
                    max_length=config.get("max_length", 50),
                    num_beams=config.get("num_beams", 3),
                )

                results.append(
                    {
                        "image_path": image_path,
                        "caption": result["caption"],
                        "confidence": result.get("confidence", 0.0),
                        "index": i,
                    }
                )

                processed_items += 1

                # Update progress every 10 items or at the end
                if (i + 1) % 10 == 0 or (i + 1) == total_items:
                    update_task_progress(
                        task_id,
                        processed_items,
                        total_items,
                        {"latest_results": results[-10:]},
                    )

            except Exception as e:
                failed_items.append(
                    {"image_path": image_path, "error": str(e), "index": i}
                )
                structured_logger.error(f"Failed to caption {image_path}: {e}")

        # Save results to file
        AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
        results_dir = (
            Path(AI_CACHE_ROOT) / "outputs" / "multi-modal-lab" / "batch_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"caption_results_{job_id}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_items": total_items,
                    "processed_items": processed_items,
                    "failed_items": len(failed_items),
                    "results": results,
                    "failures": failed_items,
                    "config": config,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "job_id": job_id,
            "status": "completed",
            "total_items": total_items,
            "processed_items": processed_items,
            "failed_items": len(failed_items),
            "results_file": str(results_file),
        }

    except Exception as e:
        structured_logger.error(
            f"Batch caption task failed: {e}\n{traceback.format_exc()}"
        )
        raise


@celery_app.task(bind=True, name="batch_vqa_task")
def batch_vqa_task(
    self, job_id: str, inputs: List[Dict[str, str]], config: Dict[str, Any]
):
    """Batch Visual Question Answering task"""
    try:
        task_id = self.request.id
        total_items = len(inputs)
        processed_items = 0
        results = []
        failed_items = []

        update_task_progress(task_id, 0, total_items)

        engine = get_vlm_engine()

        # Load VQA model if not already loaded
        if not engine.vqa_model:
            engine.load_vqa_model()

        for i, item in enumerate(inputs):
            try:
                image_path = item["image_path"]
                question = item["question"]

                result = engine.visual_question_answering(
                    image_path=image_path,
                    question=question,
                    max_length=config.get("max_length", 100),
                )

                results.append(
                    {
                        "image_path": image_path,
                        "question": question,
                        "answer": result["answer"],
                        "confidence": result.get("confidence", 0.0),
                        "index": i,
                    }
                )

                processed_items += 1

                if (i + 1) % 5 == 0 or (i + 1) == total_items:
                    update_task_progress(
                        task_id,
                        processed_items,
                        total_items,
                        {"latest_results": results[-5:]},
                    )

            except Exception as e:
                failed_items.append(
                    {
                        "image_path": item.get("image_path", ""),
                        "question": item.get("question", ""),
                        "error": str(e),
                        "index": i,
                    }
                )
                structured_logger.error(f"Failed VQA for item {i}: {e}")

        # Save results
        AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
        results_dir = (
            Path(AI_CACHE_ROOT) / "outputs" / "multi-modal-lab" / "batch_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"vqa_results_{job_id}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_items": total_items,
                    "processed_items": processed_items,
                    "failed_items": len(failed_items),
                    "results": results,
                    "failures": failed_items,
                    "config": config,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "job_id": job_id,
            "status": "completed",
            "total_items": total_items,
            "processed_items": processed_items,
            "failed_items": len(failed_items),
            "results_file": str(results_file),
        }

    except Exception as e:
        structured_logger.error(f"Batch VQA task failed: {e}\n{traceback.format_exc()}")
        raise


@celery_app.task(bind=True, name="batch_chat_task")
def batch_chat_task(
    self, job_id: str, messages_batch: List[List[Dict]], config: Dict[str, Any]
):
    """Batch chat completion task"""
    try:
        task_id = self.request.id
        total_items = len(messages_batch)
        processed_items = 0
        results = []
        failed_items = []

        update_task_progress(task_id, 0, total_items)

        adapter = get_llm_adapter()

        for i, messages in enumerate(messages_batch):
            try:
                result = adapter.chat_completion(
                    messages=messages,
                    max_length=config.get("max_length", 512),
                    temperature=config.get("temperature", 0.7),
                )

                results.append(
                    {
                        "messages": messages,
                        "response": result["message"],
                        "model_used": result.get("model_used", "unknown"),
                        "index": i,
                    }
                )

                processed_items += 1

                if (i + 1) % 5 == 0 or (i + 1) == total_items:
                    update_task_progress(
                        task_id,
                        processed_items,
                        total_items,
                        {"latest_results": results[-5:]},
                    )

            except Exception as e:
                failed_items.append({"messages": messages, "error": str(e), "index": i})
                structured_logger.error(f"Failed chat completion for item {i}: {e}")

        # Save results
        AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
        results_dir = (
            Path(AI_CACHE_ROOT) / "outputs" / "multi-modal-lab" / "batch_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"chat_results_{job_id}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_items": total_items,
                    "processed_items": processed_items,
                    "failed_items": len(failed_items),
                    "results": results,
                    "failures": failed_items,
                    "config": config,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "job_id": job_id,
            "status": "completed",
            "total_items": total_items,
            "processed_items": processed_items,
            "failed_items": len(failed_items),
            "results_file": str(results_file),
        }

    except Exception as e:
        structured_logger.error(
            f"Batch chat task failed: {e}\n{traceback.format_exc()}"
        )
        raise


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status from Celery and Redis"""
    try:
        # Get task state from Celery
        task = celery_app.AsyncResult(task_id)

        # Get progress from Redis
        progress_key = f"task_progress:{task_id}"
        progress_data = redis_client.get(progress_key)

        status_info = {"status": task.state, "task_id": task_id}

        if progress_data:
            progress = json.loads(progress_data)
            status_info.update(progress)

        if task.state == "SUCCESS":
            status_info.update(task.result)
        elif task.state == "FAILURE":
            status_info["error"] = str(task.info)

        return status_info

    except Exception as e:
        return {"status": "UNKNOWN", "error": str(e), "task_id": task_id}


def cancel_task(task_id: str) -> bool:
    """Cancel a Celery task"""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception as e:
        structured_logger.error(f"Failed to cancel task {task_id}: {e}")
        return False
