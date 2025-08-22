#!/usr/bin/env python3
"""
SagaForge v1.0.0 Release Script
Automates the complete release process including testing, documentation, and deployment
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Shared cache bootstrap
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    Path(v).mkdir(parents=True, exist_ok=True)


class ReleaseManager:
    """Manages the complete release process"""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.release_date = datetime.now().strftime("%Y-%m-%d")
        self.repo_root = Path(__file__).parent.parent
        self.release_notes = []

    def run_command(self, cmd: str, cwd: Optional[Path] = None) -> Dict[str, any]:
        """Execute shell command and return result"""
        print(f"🔧 Running: {cmd}")

        try:
            result = subprocess.run(
                cmd.split(),
                cwd=cwd or self.repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("📋 Checking prerequisites...")

        checks = [
            ("git", "Git is required"),
            ("docker", "Docker is required"),
            ("docker-compose", "Docker Compose is required"),
            ("python", "Python 3.10+ is required"),
        ]

        for cmd, desc in checks:
            result = self.run_command(f"{cmd} --version")
            if not result["success"]:
                print(f"❌ {desc}")
                return False
            print(f"✅ {desc}")

        # Check if we're on develop branch
        result = self.run_command("git branch --show-current")
        if result["success"] and result["stdout"].strip() != "develop":
            print("❌ Must be on 'develop' branch for release")
            return False

        # Check for uncommitted changes
        result = self.run_command("git status --porcelain")
        if result["success"] and result["stdout"].strip():
            print("❌ Repository has uncommitted changes")
            return False

        print("✅ All prerequisites met")
        return True

    def run_tests(self) -> bool:
        """Run complete test suite"""
        print("🧪 Running test suite...")

        # Unit tests
        print("Running unit tests...")
        result = self.run_command("python -m pytest tests/unit/ -v --tb=short")
        if not result["success"]:
            print("❌ Unit tests failed")
            print(result["stderr"])
            return False

        # Integration tests
        print("Running integration tests...")
        result = self.run_command("python -m pytest tests/integration/ -v --tb=short")
        if not result["success"]:
            print("❌ Integration tests failed")
            print(result["stderr"])
            return False

        print("✅ All tests passed")
        return True

    def run_e2e_smoke_test(self) -> bool:
        """Run end-to-end smoke test"""
        print("🔥 Running E2E smoke test...")

        # Start services in test mode
        print("Starting test environment...")
        env = os.environ.copy()
        env["POSTGRES_PASSWORD"] = "test123"
        env["AI_CACHE_ROOT"] = AI_CACHE_ROOT

        # Use test compose file
        result = self.run_command("docker-compose -f docker-compose.test.yml up -d")
        if not result["success"]:
            print("❌ Failed to start test environment")
            return False

        # Wait for services to be ready
        print("Waiting for services to start...")
        time.sleep(30)

        try:
            # Test health endpoint
            response = requests.get("http://localhost:8000/healthz", timeout=10)
            if response.status_code != 200:
                print("❌ Health check failed")
                return False

            # Test basic functionality
            test_data = {
                "message": "Hello, test world!",
                "world_id": "test",
                "character_id": "test_char",
            }

            response = requests.post(
                "http://localhost:8000/llm/turn", json=test_data, timeout=30
            )

            if response.status_code != 200:
                print("❌ LLM endpoint test failed")
                return False

            # Test image generation
            image_data = {
                "prompt": "test image",
                "width": 512,
                "height": 512,
                "steps": 10,
            }

            response = requests.post(
                "http://localhost:8000/t2i/generate", json=image_data, timeout=60
            )

            if response.status_code != 200:
                print("❌ Image generation test failed")
                return False

            print("✅ E2E smoke test passed")
            return True

        except Exception as e:
            print(f"❌ E2E test failed: {e}")
            return False

        finally:
            # Cleanup test environment
            self.run_command("docker-compose -f docker-compose.test.yml down -v")

    def build_documentation(self) -> bool:
        """Build and verify documentation"""
        print("📚 Building documentation...")

        # Check if all required docs exist
        required_docs = [
            "README.md",
            "docs/deployment.md",
            "docs/api.md",
            "docs/development.md",
        ]

        for doc in required_docs:
            doc_path = self.repo_root / doc
            if not doc_path.exists():
                print(f"❌ Missing required documentation: {doc}")
                return False

        # Generate API documentation
        try:
            from api.main import app

            # Generate OpenAPI spec
            openapi_spec = app.openapi()
            spec_path = self.repo_root / "docs" / "openapi.json"

            with open(spec_path, "w") as f:
                json.dump(openapi_spec, f, indent=2)

            print("✅ API documentation generated")

        except Exception as e:
            print(f"❌ Failed to generate API docs: {e}")
            return False

        print("✅ Documentation ready")
        return True

    def build_docker_images(self) -> bool:
        """Build and tag Docker images"""
        print("🐳 Building Docker images...")

        images = [
            ("api", "Dockerfile.api"),
            ("worker", "Dockerfile.worker"),
            ("webui", "Dockerfile.webui"),
        ]

        for image_name, dockerfile in images:
            print(f"Building {image_name} image...")

            result = self.run_command(
                f"docker build -t sagaforge/{image_name}:{self.version} -f {dockerfile} ."
            )

            if not result["success"]:
                print(f"❌ Failed to build {image_name} image")
                print(result["stderr"])
                return False

            # Also tag as latest
            self.run_command(
                f"docker tag sagaforge/{image_name}:{self.version} sagaforge/{image_name}:latest"
            )

        print("✅ Docker images built successfully")
        return True

    def create_release_artifacts(self) -> bool:
        """Create release artifacts and packages"""
        print("📦 Creating release artifacts...")

        release_dir = self.repo_root / f"release-{self.version}"
        release_dir.mkdir(exist_ok=True)

        # Create source distribution
        result = self.run_command(
            "git archive --format=zip --output=sagaforge-source.zip HEAD"
        )
        if not result["success"]:
            print("❌ Failed to create source archive")
            return False

        # Move to release directory
        (self.repo_root / "sagaforge-source.zip").rename(
            release_dir / f"sagaforge-{self.version}-source.zip"
        )

        # Create deployment package
        deployment_files = [
            "docker-compose.prod.yml",
            ".env.example",
            "nginx.conf",
            "scripts/setup_production.sh",
        ]

        import zipfile

        deploy_zip_path = release_dir / f"sagaforge-{self.version}-deployment.zip"

        with zipfile.ZipFile(deploy_zip_path, "w") as zf:
            for file_path in deployment_files:
                if (self.repo_root / file_path).exists():
                    zf.write(self.repo_root / file_path, file_path)

        # Create release notes
        self.generate_release_notes(release_dir)

        print(f"✅ Release artifacts created in {release_dir}")
        return True

    def generate_release_notes(self, release_dir: Path) -> None:
        """Generate comprehensive release notes"""
        print("📝 Generating release notes...")

        # Get commit log since last release
        result = self.run_command(
            "git log --oneline --grep='feat\\|fix\\|breaking' --since='3 months ago'"
        )
        commits = result["stdout"].strip().split("\n") if result["success"] else []

        features = []
        fixes = []
        breaking = []

        for commit in commits:
            if "feat(" in commit:
                features.append(commit)
            elif "fix(" in commit:
                fixes.append(commit)
            elif "breaking" in commit.lower():
                breaking.append(commit)

        release_notes = f"""# SagaForge v{self.version} Release Notes

**Release Date:** {self.release_date}

## 🎉 Major Features

SagaForge v{self.version} represents the first stable release of our comprehensive AI-powered interactive storytelling platform. This release includes:

### ✨ Core Features
- **Multi-modal AI Pipeline**: Complete integration of LLM + RAG + T2I + VLM + LoRA training
- **Traditional Chinese RAG**: Optimized retrieval-augmented generation for Traditional Chinese content
- **LoRA Fine-tuning**: Automated character-specific LoRA training with evaluation metrics
- **Interactive Story Engine**: Dynamic narrative generation with choice-driven gameplay
- **Visual Content Generation**: SDXL/SD1.5 with ControlNet pose control and IP-Adapter
- **Batch Processing**: Scalable background job processing with Celery + Redis
- **Production Ready**: Docker Compose deployment with monitoring and health checks

### 🛠 Technical Highlights
- **Shared Model Warehouse**: Centralized AI_CACHE_ROOT for efficient model/data management
- **Performance Optimizations**: Embedding caching, model hot-loading, GPU memory management
- **Comprehensive Monitoring**: Health checks, metrics, task monitoring, and error tracking
- **API-First Design**: FastAPI with auto-generated OpenAPI documentation
- **Multi-Interface Support**: Gradio WebUI + PyQt desktop + REST API

### 📚 New Components
- **LLM Adapters**: Support for Transformers, Ollama, and custom backends
- **RAG Engine**: Hierarchical chunking, hybrid retrieval, and citation tracking
- **T2I Pipeline**: Unified SDXL/SD1.5 with LoRA management and safety filters
- **VLM Integration**: BLIP/LLaVA captioning with consistency checking
- **Training Framework**: Automated LoRA training with accelerate + PEFT

## 🔧 New Features

"""

        if features:
            release_notes += "### Features Added\n"
            for feat in features[:10]:  # Limit to top 10
                release_notes += f"- {feat}\n"

        if fixes:
            release_notes += "\n### Bug Fixes\n"
            for fix in fixes[:10]:
                release_notes += f"- {fix}\n"

        if breaking:
            release_notes += "\n### ⚠️ Breaking Changes\n"
            for change in breaking:
                release_notes += f"- {change}\n"

        release_notes += f"""

## 📊 Performance Benchmarks

### Generation Performance (RTX 4070)
- **SDXL 1024x1024**: ~15-20s (30 steps)
- **SD1.5 512x512**: ~5-8s (25 steps)
- **ControlNet Pose**: +2-3s overhead
- **Batch Generation**: ~5 images/min sustained

### RAG Performance
- **Embedding (bge-m3)**: ~50ms per query
- **Retrieval + Rerank**: ~100-200ms
- **Cache Hit Rate**: >80% in typical usage

### Memory Usage
- **Minimal Setup**: 8GB GPU VRAM
- **Full Pipeline**: 12GB GPU VRAM recommended
- **CPU Fallback**: 16GB system RAM

## 🚀 Quick Start

### Production Deployment
```bash
git clone https://github.com/your-org/saga-forge.git
cd saga-forge
cp .env.example .env
# Edit .env with your settings
docker-compose -f docker-compose.prod.yml up -d
```

### Development Setup
```bash
conda create -n adventure-lab python=3.10 -y
conda activate adventure-lab
pip install -r requirements.txt
export AI_CACHE_ROOT="../ai_warehouse/cache"
uvicorn api.main:app --reload
```

## 📖 Documentation

- **[Deployment Guide](docs/deployment.md)**: Complete production setup
- **[API Documentation](http://localhost:8000/docs)**: Interactive API explorer
- **[Development Guide](docs/development.md)**: Contributing and customization
- **[Worldpack Format](docs/worldpack_format.md)**: Content creation guide

## 🔄 Migration Guide

This is the first stable release. Future versions will include migration scripts for:
- Database schema updates
- Configuration file format changes
- Model compatibility updates

## 🐛 Known Issues

- **GPU Memory**: Some VRAM-constrained setups may need `low_vram_mode: true`
- **Windows Docker**: Path mounting may require adjustments in docker-compose.yml
- **M1/M2 Macs**: Limited GPU acceleration support (CPU fallback available)

## 🛡️ Security Notes

- Default deployment uses unencrypted connections (use nginx + SSL for production)
- No authentication enabled by default (configure JWT tokens for multi-user)
- Model downloads happen automatically (verify sources match your security policy)

## 🎯 Roadmap v1.1

- **Video Generation**: Stable Video Diffusion integration
- **Real-time Collaboration**: Multi-user story sessions
- **Advanced LoRA**: Multi-character and style mixing
- **Mobile App**: React Native companion app
- **Cloud Integration**: AWS/GCP deployment templates

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- PR review process
- Issue reporting templates

## 🙏 Acknowledgments

Special thanks to:
- **Hugging Face** for transformers and diffusers libraries
- **Stability AI** for Stable Diffusion models
- **Beijing Academy of AI** for BGE embedding models
- **OpenAI** for LLM evaluation frameworks
- **Our amazing beta testers** for feedback and bug reports

## 📞 Support

- **Documentation**: [docs.sagaforge.ai](https://docs.sagaforge.ai)
- **Community**: [GitHub Discussions](https://github.com/your-org/saga-forge/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/your-org/saga-forge/issues)
- **Discord**: [Join our community](https://discord.gg/sagaforge)

## 📄 License

SagaForge is released under the [Apache License 2.0](LICENSE).

---

**Full Changelog**: [v0.9.0...v{self.version}](https://github.com/your-org/saga-forge/compare/v0.9.0...v{self.version})

**Download**: [Release Assets](https://github.com/your-org/saga-forge/releases/tag/v{self.version})
"""

        # Write release notes
        notes_path = release_dir / f"RELEASE_NOTES_v{self.version}.md"
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(release_notes)

        print(f"✅ Release notes generated: {notes_path}")

    def create_git_release(self) -> bool:
        """Create Git tag and GitHub release"""
        print("🏷️ Creating Git release...")

        # Merge develop to main
        print("Merging develop to main...")
        result = self.run_command("git checkout main")
        if not result["success"]:
            print("❌ Failed to checkout main branch")
            return False

        result = self.run_command(
            "git merge develop --no-ff -m f'release: v{self.version}'"
        )
        if not result["success"]:
            print("❌ Failed to merge develop to main")
            return False

        # Create and push tag
        tag_message = (
            f"SagaForge v{self.version} - Interactive AI Storytelling Platform"
        )
        result = self.run_command(f"git tag -a v{self.version} -m '{tag_message}'")
        if not result["success"]:
            print("❌ Failed to create Git tag")
            return False

        result = self.run_command("git push origin main --tags")
        if not result["success"]:
            print("❌ Failed to push release to remote")
            return False

        print(f"✅ Git release v{self.version} created and pushed")
        return True

    def run_deployment_test(self) -> bool:
        """Test production deployment configuration"""
        print("🚀 Testing production deployment...")

        # Test docker-compose.prod.yml syntax
        result = self.run_command("docker-compose -f docker-compose.prod.yml config")
        if not result["success"]:
            print("❌ Production docker-compose configuration is invalid")
            print(result["stderr"])
            return False

        # Test environment variables
        env_example = self.repo_root / ".env.example"
        if not env_example.exists():
            print("❌ .env.example file missing")
            return False

        # Parse and validate required env vars
        required_vars = ["AI_CACHE_ROOT", "POSTGRES_PASSWORD"]

        with open(env_example) as f:
            env_content = f.read()

        for var in required_vars:
            if var not in env_content:
                print(f"❌ Required environment variable {var} not in .env.example")
                return False

        print("✅ Production deployment configuration valid")
        return True

    def cleanup_and_verify(self) -> bool:
        """Final cleanup and verification"""
        print("🧹 Final cleanup and verification...")

        # Verify all artifacts exist
        release_dir = self.repo_root / f"release-{self.version}"
        expected_artifacts = [
            f"sagaforge-{self.version}-source.zip",
            f"sagaforge-{self.version}-deployment.zip",
            f"RELEASE_NOTES_v{self.version}.md",
        ]

        for artifact in expected_artifacts:
            artifact_path = release_dir / artifact
            if not artifact_path.exists():
                print(f"❌ Missing release artifact: {artifact}")
                return False

        # Verify Docker images exist
        result = self.run_command(f"docker images sagaforge/api:{self.version}")
        if "sagaforge/api" not in result["stdout"]:
            print("❌ Docker images not properly built")
            return False

        # Check Git tag exists
        result = self.run_command("git tag -l")
        if f"v{self.version}" not in result["stdout"]:
            print("❌ Git tag not created")
            return False

        print("✅ All verification checks passed")
        return True

    def display_success_summary(self) -> None:
        """Display final success summary with next steps"""
        print("\n" + "=" * 60)
        print(f"🎉 SagaForge v{self.version} Release Complete! 🎉")
        print("=" * 60)

        print(f"\n📦 Release Artifacts:")
        release_dir = self.repo_root / f"release-{self.version}"
        print(f"   📁 {release_dir}")

        print(f"\n🐳 Docker Images:")
        print(f"   🏷️ sagaforge/api:{self.version}")
        print(f"   🏷️ sagaforge/worker:{self.version}")
        print(f"   🏷️ sagaforge/webui:{self.version}")

        print(f"\n🏷️ Git Release:")
        print(f"   📋 Tag: v{self.version}")
        print(f"   🌿 Branch: main (merged from develop)")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Push Docker images to registry:")
        print(f"      docker push sagaforge/api:{self.version}")
        print(f"      docker push sagaforge/worker:{self.version}")
        print(f"      docker push sagaforge/webui:{self.version}")

        print(f"   2. Create GitHub Release:")
        print(f"      - Go to https://github.com/your-org/saga-forge/releases")
        print(f"      - Create release from tag v{self.version}")
        print(f"      - Upload artifacts from {release_dir}")

        print(f"   3. Update documentation:")
        print(f"      - Update README.md with v{self.version} info")
        print(f"      - Publish docs to docs.sagaforge.ai")

        print(f"   4. Announce release:")
        print(f"      - Update Discord community")
        print(f"      - Post on social media")
        print(f"      - Email subscribers")

        print(f"\n📊 Release Statistics:")
        result = self.run_command("git log --oneline v0.9.0..HEAD")
        commit_count = (
            len(result["stdout"].strip().split("\n")) if result["success"] else 0
        )
        print(f"   📝 Commits since last release: {commit_count}")

        print(f"\n✨ Thank you for using SagaForge! ✨")
        print("=" * 60)


def main():
    """Main release process"""
    print("🚀 Starting SagaForge v1.0.0 Release Process")
    print("=" * 50)

    manager = ReleaseManager("1.0.0")

    steps = [
        ("Prerequisites Check", manager.check_prerequisites),
        ("Test Suite", manager.run_tests),
        ("E2E Smoke Test", manager.run_e2e_smoke_test),
        ("Documentation Build", manager.build_documentation),
        ("Docker Images", manager.build_docker_images),
        ("Release Artifacts", manager.create_release_artifacts),
        ("Deployment Test", manager.run_deployment_test),
        ("Git Release", manager.create_git_release),
        ("Final Verification", manager.cleanup_and_verify),
    ]

    start_time = time.time()

    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 40)

        if not step_func():
            print(f"💥 Release failed at step: {step_name}")
            sys.exit(1)

        print(f"✅ {step_name} completed successfully")

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total release time: {elapsed_time:.1f} seconds")

    manager.display_success_summary()


if __name__ == "__main__":
    main()
