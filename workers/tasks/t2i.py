# workers/tasks/t2i.py
from workers.celery_app import celery_app
from core.t2i import get_t2i_pipeline, save_image_to_cache
import torch


@celery_app.task
def generate_image_async(request_data: dict):
    """Async image generation task"""

    pipeline = get_t2i_pipeline(
        request_data.get("model", "runwayml/stable-diffusion-v1-5")
    )

    result = pipeline(
        prompt=request_data["prompt"],
        negative_prompt=request_data.get("negative_prompt", ""),
        width=request_data.get("width", 768),
        height=request_data.get("height", 768),
        num_inference_steps=request_data.get("steps", 25),
        guidance_scale=request_data.get("guidance_scale", 7.5),
        generator=torch.Generator().manual_seed(request_data.get("seed", 42)),
    )

    image_path, metadata_path = save_image_to_cache(result.images[0], request_data)

    return {
        "image_path": image_path,
        "metadata_path": metadata_path,
        "seed": request_data.get("seed", 42),
    }
