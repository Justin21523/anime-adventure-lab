import os, pathlib, torch
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v; pathlib.Path(v).mkdir(parents=True, exist_ok=True)
print("[cache]", AI_CACHE_ROOT, "| GPU:", torch.cuda.is_available())
