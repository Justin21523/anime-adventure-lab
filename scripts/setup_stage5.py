# scripts/setup_stage5.py
"""
SagaForge Stage 5 Setup Script
Sets up T2I pipeline dependencies and validates environment
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\n🔥 Checking CUDA...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split("\n")
            for line in lines:
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version: ")[1].split()[0]
                    print(f"✓ CUDA Version: {cuda_version}")
                    break
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - will use CPU mode")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - will use CPU mode")
        return False


def install_packages():
    """Install required packages"""
    print("\n📦 Installing Stage 5 dependencies...")

    # Core packages for Stage 5
    packages = [
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.30.0",
        "transformers>=4.35.0",
        "accelerate>=0.20.0",
        "peft>=0.5.0",
        "safetensors>=0.4.0",
        "xformers",  # For memory optimization
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "pillow>=9.0.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
    ]

    # Install with pip
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                check=True,
                capture_output=True,
            )
            print(f"✓ {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False

    return True


def setup_directories():
    """Setup required directories"""
    print("\n📁 Setting up directories...")

    # Get cache root from environment
    cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    cache_path = Path(cache_root)

    directories = [
        cache_path / "hf",
        cache_path / "hf/transformers",
        cache_path / "hf/datasets",
        cache_path / "hf/hub",
        cache_path / "torch",
        cache_path / "models/sd",
        cache_path / "models/sdxl",
        cache_path / "models/controlnet",
        cache_path / "models/lora",
        cache_path / "outputs/saga-forge",
        Path("configs/presets"),
        Path("logs"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

    print(f"✓ Cache root: {cache_root}")
    return True


def create_env_file():
    """Create .env file if not exists"""
    print("\n⚙️  Setting up environment...")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        if env_example.exists():
            # Copy from example
            content = env_example.read_text(encoding="utf-8")
            env_file.write_text(content, encoding="utf-8")
            print("✓ Created .env from .env.example")
        else:
            # Create minimal .env
            cache_root = Path("../ai_warehouse/cache").resolve()
            content = f"""# SagaForge Stage 5 Environment
AI_CACHE_ROOT={cache_root}
CUDA_VISIBLE_DEVICES=0
API_CORS_ORIGINS=http://localhost:3000,http://localhost:7860
"""
            env_file.write_text(content, encoding="utf-8")
            print("✓ Created minimal .env file")
    else:
        print("✓ .env file already exists")


def validate_installation():
    """Validate installation"""
    print("\n🔍 Validating installation...")

    try:
        # Test imports
        import torch

        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        import diffusers

        print(f"✓ Diffusers {diffusers.__version__}")

        import transformers

        print(f"✓ Transformers {transformers.__version__}")

        import fastapi

        print(f"✓ FastAPI {fastapi.__version__}")

        # Test shared cache
        sys.path.append(str(Path(__file__).parent.parent))
        from core.shared_cache import setup_cache

        cache_root, app_dirs = setup_cache()
        print(f"✓ Shared cache initialized: {cache_root}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 SagaForge Stage 5 Setup")
    print("=" * 40)
    print("Setting up T2I Pipeline dependencies")
    print("=" * 40)

    # Step 1: Check Python
    if not check_python_version():
        return False

    # Step 2: Check CUDA
    cuda_available = check_cuda()

    # Step 3: Setup directories
    if not setup_directories():
        return False

    # Step 4: Create .env
    create_env_file()

    # Step 5: Install packages
    if not install_packages():
        return False

    # Step 6: Validate
    if not validate_installation():
        return False

    # Success!
    print("\n" + "=" * 40)
    print("🎉 Stage 5 setup complete!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Start API server:")
    print("   python api/main.py")
    print("\n2. Run smoke tests:")
    print("   python scripts/smoke_test_t2i.py")
    print("\n3. Open API docs:")
    print("   http://localhost:8000/docs")
    print("\n4. Test image generation:")
    print("   curl -X POST http://localhost:8000/api/v1/t2i/generate \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"prompt":"anime girl","seed":42}\'')

    if not cuda_available:
        print("\n⚠️  Running in CPU mode - generation will be slow")
        print("   Consider setting up CUDA for better performance")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
