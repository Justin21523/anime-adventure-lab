# workers/tasks/t2i.py
from workers.celery_app import celery_app
from core.t2i.engine import T2IEngine
from core.shared_cache import get_shared_cache


@celery_app.task
def generate_image_async(request_data: dict):
    """Async image generation task"""

    cache = get_shared_cache()
    engine = T2IEngine(
        cache_root=cache.cache_root,
        device="auto",
        config={
            "default_model": request_data.get("model", "runwayml/stable-diffusion-v1-5"),
            "mock_generation": request_data.get("mock", False),
        },
    )
    result = engine.txt2img(
        {
            "prompt": request_data["prompt"],
            "negative_prompt": request_data.get("negative_prompt", ""),
            "width": request_data.get("width", 768),
            "height": request_data.get("height", 768),
            "num_inference_steps": request_data.get("steps", 25),
            "guidance_scale": request_data.get("guidance_scale", 7.5),
            "seed": request_data.get("seed", 42),
        }
    )

    metadata = result.get("metadata", {})
    paths = metadata.get("output_paths") or []

    return {
        "image_path": paths[0] if paths else "",
        "metadata": metadata,
        "seed": request_data.get("seed", 42),
    }
