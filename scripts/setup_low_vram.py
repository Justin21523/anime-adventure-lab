#!/usr/bin/env python3
# scripts/setup_low_vram.py
"""
Low VRAM optimization setup for SagaForge
Optimizes system for 4GB-8GB GPU memory
"""

import os
import yaml
import torch
from pathlib import Path


def detect_gpu_memory():
    """Detect available GPU memory"""
    if not torch.cuda.is_available():
        return 0

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return gpu_memory


def create_optimized_config(gpu_memory_gb: float):
    """Create optimized configuration based on GPU memory"""

    if gpu_memory_gb < 6:
        profile = "low_vram"
        config = {
            "memory": {
                "enable_4bit": True,
                "enable_8bit": False,
                "cpu_offload": True,
                "sequential_cpu_offload": True,
                "attention_slicing": True,
                "vae_slicing": True,
                "enable_xformers": True,
                "max_gpu_memory": gpu_memory_gb - 1.0,
                "reserved_gpu_memory": 0.5,
            },
            "inference": {
                "max_batch_size": 1,
                "pipeline_sequential": True,
                "use_deterministic": False,
            },
            "models": {
                "llm": {
                    "quantization": "4bit",
                    "max_context_length": 1024,
                    "use_flash_attention": False,
                },
                "diffusion": {"num_inference_steps": 15, "guidance_scale": 7.0},
            },
        }
    elif gpu_memory_gb < 12:
        profile = "balanced"
        config = {
            "memory": {
                "enable_4bit": False,
                "enable_8bit": True,
                "cpu_offload": False,
                "sequential_cpu_offload": False,
                "attention_slicing": True,
                "vae_slicing": True,
                "enable_xformers": True,
                "max_gpu_memory": gpu_memory_gb - 2.0,
                "reserved_gpu_memory": 1.0,
            },
            "inference": {
                "max_batch_size": 2,
                "pipeline_sequential": False,
                "use_deterministic": False,
            },
            "models": {
                "llm": {
                    "quantization": "8bit",
                    "max_context_length": 2048,
                    "use_flash_attention": True,
                },
                "diffusion": {"num_inference_steps": 20, "guidance_scale": 7.5},
            },
        }
    else:
        profile = "high_performance"
        config = {
            "memory": {
                "enable_4bit": False,
                "enable_8bit": False,
                "cpu_offload": False,
                "sequential_cpu_offload": False,
                "attention_slicing": False,
                "vae_slicing": False,
                "enable_xformers": True,
                "max_gpu_memory": gpu_memory_gb - 2.0,
                "reserved_gpu_memory": 2.0,
            },
            "inference": {
                "max_batch_size": 4,
                "pipeline_sequential": False,
                "use_deterministic": False,
            },
            "models": {
                "llm": {
                    "quantization": "none",
                    "max_context_length": 4096,
                    "use_flash_attention": True,
                },
                "diffusion": {"num_inference_steps": 25, "guidance_scale": 7.5},
            },
        }

    return profile, config


def setup_environment_variables():
    """Set up optimal environment variables"""

    # PyTorch optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # Memory management
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    # HuggingFace optimizations
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    print("✅ Environment variables configured for low VRAM")


def create_env_file(profile: str, gpu_memory: float):
    """Create optimized .env file"""

    env_content = f"""# SagaForge Environment Configuration - {profile.upper()} Profile
# Generated for {gpu_memory:.1f}GB GPU

# Shared Cache
AI_CACHE_ROOT=../ai_warehouse/cache

# Performance Profile
PERFORMANCE_PROFILE={profile}

# GPU Settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Memory Optimization
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1

# HuggingFace Settings
HF_HUB_DISABLE_TELEMETRY=1
TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# API Settings
API_CORS_ORIGINS=http://localhost:3000,http://localhost:7860
API_MAX_WORKERS=1

# Redis (optional)
REDIS_URL=redis://localhost:6379/1

# Development
DEBUG=false
LOG_LEVEL=INFO
"""

    with open(".env", "w") as f:
        f.write(env_content)

    print(f"✅ Created .env file with {profile} profile")


def apply_torch_optimizations():
    """Apply PyTorch-level optimizations"""

    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Enable memory efficiency
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print("✅ Applied PyTorch CUDA optimizations")
    else:
        print("⚠️ CUDA not available, skipping GPU optimizations")


def create_requirements_minimal():
    """Create minimal requirements for low VRAM setup"""

    requirements = """# Minimal requirements for low VRAM setup
torch>=2.0.0
torchvision
torchaudio

# Core ML libraries
transformers>=4.35.0
diffusers>=0.24.0
accelerate>=0.25.0
peft>=0.7.0

# Quantization
bitsandbytes>=0.41.0

# Memory efficiency
xformers>=0.0.22

# Fast inference
optimum>=1.15.0

# API
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
celery[redis]>=5.3.0
redis>=5.0.0

# Basic dependencies
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0
pydantic>=2.5.0
"""

    with open("requirements-low-vram.txt", "w") as f:
        f.write(requirements)

    print("✅ Created requirements-low-vram.txt")


def main():
    """Main setup function"""
    print("🚀 SagaForge Low VRAM Optimization Setup")
    print("=" * 50)

    # Detect GPU
    gpu_memory = detect_gpu_memory()
    print(f"Detected GPU memory: {gpu_memory:.1f}GB")

    if gpu_memory == 0:
        print("⚠️ No GPU detected, configuring for CPU-only mode")
        profile = "cpu_only"
        config = {
            "memory": {"cpu_offload": True, "device": "cpu"},
            "inference": {"max_batch_size": 1},
        }
    else:
        # Create optimized config
        profile, config = create_optimized_config(gpu_memory)
        print(f"Selected profile: {profile}")

    # Apply optimizations
    setup_environment_variables()
    create_env_file(profile, gpu_memory)
    apply_torch_optimizations()
    create_requirements_minimal()

    # Save performance config
    config_path = Path("configs/performance_optimized.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Saved optimized config to {config_path}")

    print("\n" + "=" * 50)
    print("🎉 Low VRAM optimization setup complete!")

    print(f"\nRecommendations for {profile} profile:")
    if profile == "low_vram":
        print("- Use 4-bit quantization for LLMs")
        print("- Enable CPU offloading for large models")
        print("- Reduce batch size to 1")
        print("- Use attention slicing for diffusion models")
        print("- Consider using smaller base models (SD 1.5 vs SDXL)")
    elif profile == "balanced":
        print("- Use 8-bit quantization for good speed/quality balance")
        print("- Enable xformers for memory efficiency")
        print("- Batch size of 2 should work well")
        print("- Monitor GPU memory usage during inference")
    else:
        print("- Full precision models available")
        print("- Higher batch sizes for faster generation")
        print("- Consider torch.compile for additional speedup")

    print(f"\nNext steps:")
    print("1. Install optimized requirements: pip install -r requirements-low-vram.txt")
    print(
        "2. Start with performance config: python -c 'from core.performance import *; test_setup()'"
    )
    print("3. Run smoke test: python scripts/test_stage9_performance.py")
    print("4. Monitor performance: python frontend/gradio/performance_monitor.py")


if __name__ == "__main__":
    main()
