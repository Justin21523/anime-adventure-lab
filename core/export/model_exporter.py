# core/export/model_exporter.py
"""
Model Export System
Export trained models to different formats and deployment targets
"""

import os
import json
import logging
import shutil
import zipfile
import tempfile
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from peft import PeftModel
import yaml

from ..config import get_config
from ..exceptions import ExportError, ValidationError
from ..utils import get_cache_manager, get_model_manager

logger = logging.getLogger(__name__)


class ModelExporter:
    """Model export and packaging system"""

    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.model_manager = get_model_manager()

        # Export formats supported
        self.supported_formats = {
            "diffusers": self._export_diffusers,
            "safetensors": self._export_safetensors,
            "checkpoint": self._export_checkpoint,
            "onnx": self._export_onnx,
            "huggingface": self._export_huggingface,
            "package": self._export_package,
            "torchscript": self._export_torchscript,
            "tflite": self._export_tflite,
        }

    def export_model(
        self,
        model_path: str,
        output_path: str,
        export_format: str = "diffusers",
        include_metadata: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export model to specified format"""

        if export_format not in self.supported_formats:
            raise ValidationError(f"Unsupported export format: {export_format}")

        # Validate model exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise ExportError(f"Model file not found: {model_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔄 Exporting model to {export_format} format")

        try:
            # Call appropriate export function
            export_func = self.supported_formats[export_format]
            result = export_func(model_path, output_path, **kwargs)

            # Add metadata if requested
            if include_metadata:
                self._add_metadata(model_path, output_path, export_format, result)

            result.update(
                {
                    "export_format": export_format,
                    "output_path": str(output_path),
                    "success": True,
                    "exported_at": time.time(),
                }
            )

            logger.info(f"✅ Model exported successfully: {output_path}")
            return result

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            raise ExportError(f"Model export failed: {e}")

    def _export_diffusers(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as Diffusers format"""
        try:
            # Determine model type based on structure
            if (model_path / "unet").exists():
                # Already a diffusers model, just copy
                shutil.copytree(model_path, output_path, dirs_exist_ok=True)

                return {
                    "format": "diffusers",
                    "components": [
                        "unet",
                        "vae",
                        "text_encoder",
                        "tokenizer",
                        "scheduler",
                    ],
                    "method": "copy",
                }

            elif (model_path / "adapter_model.safetensors").exists():
                # LoRA weights, need base model
                base_model = kwargs.get("base_model", "runwayml/stable-diffusion-v1-5")

                # Load base pipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

                # Load LoRA weights
                pipeline.load_lora_weights(str(model_path))

                # Optionally fuse LoRA weights into pipeline
                if kwargs.get("fuse_lora", False):
                    pipeline.fuse_lora()

                # Save pipeline
                pipeline.save_pretrained(output_path)

                return {
                    "format": "diffusers",
                    "components": [
                        "unet",
                        "vae",
                        "text_encoder",
                        "tokenizer",
                        "scheduler",
                    ],
                    "base_model": base_model,
                    "fused_lora": kwargs.get("fuse_lora", False),
                }

            else:
                raise ExportError("Unknown model format for diffusers export")

        except Exception as e:
            raise ExportError(f"Diffusers export failed: {e}")

    def _export_safetensors(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as SafeTensors format"""
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file

            # Check if already in safetensors format
            if (model_path / "adapter_model.safetensors").exists():
                # Copy existing safetensors file
                shutil.copy2(model_path / "adapter_model.safetensors", output_path)

            elif (model_path / "pytorch_lora_weights.bin").exists():
                # Convert from PyTorch format
                weights = torch.load(
                    model_path / "pytorch_lora_weights.bin", map_location="cpu"
                )
                save_file(weights, output_path)

            elif model_path.suffix == ".bin":
                # Single PyTorch file
                weights = torch.load(model_path, map_location="cpu")
                if isinstance(weights, dict) and "state_dict" in weights:
                    weights = weights["state_dict"]
                save_file(weights, output_path)

            else:
                raise ExportError("No compatible weights found for SafeTensors export")

            file_size_mb = output_path.stat().st_size / 1024**2

            return {
                "format": "safetensors",
                "file_size_mb": file_size_mb,
                "secure": True,
            }

        except Exception as e:
            raise ExportError(f"SafeTensors export failed: {e}")

    def _export_checkpoint(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as checkpoint format"""
        try:
            base_model = kwargs.get("base_model", "runwayml/stable-diffusion-v1-5")

            # Load pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model, torch_dtype=torch.float16
            )

            # Load additional weights if LoRA
            if (model_path / "adapter_model.safetensors").exists():
                pipeline.load_lora_weights(str(model_path))
                pipeline.fuse_lora()

            # Convert to checkpoint format
            checkpoint = {
                "state_dict": {},
                "model_config": {
                    "base_model": base_model,
                    "resolution": kwargs.get("resolution", 512),
                    "channels": 4,
                    "sample_size": kwargs.get("resolution", 512) // 8,
                },
                "export_info": {
                    "format": "checkpoint",
                    "exported_at": time.time(),
                    "exporter_version": "1.0.0",
                },
            }

            # Extract state dict from UNet
            for name, param in pipeline.unet.named_parameters():
                checkpoint["state_dict"][f"model.diffusion_model.{name}"] = param.cpu()

            # Optionally include text encoder
            if kwargs.get("include_text_encoder", False):
                for name, param in pipeline.text_encoder.named_parameters():
                    checkpoint["state_dict"][
                        f"cond_stage_model.transformer.{name}"
                    ] = param.cpu()

            # Save checkpoint
            torch.save(checkpoint, output_path)

            file_size_mb = output_path.stat().st_size / 1024**2

            return {
                "format": "checkpoint",
                "file_size_mb": file_size_mb,
                "base_model": base_model,
                "includes_text_encoder": kwargs.get("include_text_encoder", False),
            }

        except Exception as e:
            raise ExportError(f"Checkpoint export failed: {e}")

    def _export_onnx(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as ONNX format"""
        try:
            import onnx

            base_model = kwargs.get("base_model", "runwayml/stable-diffusion-v1-5")
            batch_size = kwargs.get("batch_size", 1)
            resolution = kwargs.get("resolution", 512)

            # Load pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model, torch_dtype=torch.float16
            )

            # Load LoRA if applicable
            if (model_path / "adapter_model.safetensors").exists():
                pipeline.load_lora_weights(str(model_path))
                pipeline.fuse_lora()

            # Create output directory
            output_path.mkdir(exist_ok=True)

            # Export UNet to ONNX
            unet_path = output_path / "unet"
            unet_path.mkdir(exist_ok=True)

            # Create dummy inputs for tracing
            dummy_inputs = {
                "sample": torch.randn(batch_size, 4, resolution // 8, resolution // 8),
                "timestep": torch.tensor([1]),
                "encoder_hidden_states": torch.randn(batch_size, 77, 768),
            }

            # Export UNet
            torch.onnx.export(
                pipeline.unet,
                tuple(dummy_inputs.values()),
                unet_path / "model.onnx",
                input_names=list(dummy_inputs.keys()),
                output_names=["output"],
                dynamic_axes={
                    "sample": {0: "batch_size"},
                    "encoder_hidden_states": {0: "batch_size"},
                },
                opset_version=14,
                do_constant_folding=True,
            )

            # Save model config
            config = {
                "model_type": "stable_diffusion_unet",
                "base_model": base_model,
                "batch_size": batch_size,
                "resolution": resolution,
                "input_shapes": {k: list(v.shape) for k, v in dummy_inputs.items()},
            }

            with open(output_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            return {
                "format": "onnx",
                "components": ["unet"],
                "batch_size": batch_size,
                "resolution": resolution,
                "opset_version": 14,
            }

        except Exception as e:
            raise ExportError(f"ONNX export failed: {e}")

    def _export_huggingface(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export to Hugging Face Hub format"""
        try:
            # Create output directory structure
            output_path.mkdir(parents=True, exist_ok=True)

            # Copy model files
            if model_path.is_dir():
                # Copy all files from source directory
                for file in model_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, output_path)
            else:
                # Single file, copy it
                shutil.copy2(model_path, output_path)

            # Create or update adapter_config.json for LoRA models
            config_file = output_path / "adapter_config.json"
            if (
                not config_file.exists()
                and (output_path / "adapter_model.safetensors").exists()
            ):
                adapter_config = {
                    "base_model_name_or_path": kwargs.get(
                        "base_model", "runwayml/stable-diffusion-v1-5"
                    ),
                    "bias": "none",
                    "inference_mode": True,
                    "modules_to_save": None,
                    "peft_type": "LORA",
                    "r": kwargs.get("lora_rank", 16),
                    "lora_alpha": kwargs.get("lora_alpha", 32),
                    "lora_dropout": kwargs.get("lora_dropout", 0.1),
                    "target_modules": kwargs.get(
                        "target_modules", ["to_k", "to_q", "to_v", "to_out.0"]
                    ),
                    "task_type": "DIFFUSION_IMAGE_GENERATION",
                }

                with open(config_file, "w") as f:
                    json.dump(adapter_config, f, indent=2)

            # Create README.md
            readme_path = output_path / "README.md"
            self._create_model_readme(model_path, readme_path, **kwargs)

            # Create model card
            model_card_path = output_path / "MODEL_CARD.md"
            self._create_model_card(model_path, model_card_path, **kwargs)

            return {
                "format": "huggingface",
                "files_created": [f.name for f in output_path.iterdir()],
                "upload_ready": True,
                "hub_compatible": True,
            }

        except Exception as e:
            raise ExportError(f"Hugging Face export failed: {e}")

    def _export_package(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as complete package with metadata"""
        try:
            # Create package structure
            package_dir = output_path.with_suffix("")
            package_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files
            model_dir = package_dir / "model"
            model_dir.mkdir(exist_ok=True)

            if model_path.is_dir():
                shutil.copytree(model_path, model_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(model_path, model_dir)

            # Create package manifest
            manifest = {
                "package_info": {
                    "name": kwargs.get("model_name", model_path.name),
                    "version": kwargs.get("version", "1.0.0"),
                    "created_at": time.time(),
                    "package_format": "multi-modal-lab-v1",
                },
                "model_info": {
                    "type": kwargs.get("model_type", "diffusion"),
                    "base_model": kwargs.get("base_model", "unknown"),
                    "architecture": kwargs.get("architecture", "stable-diffusion"),
                    "precision": kwargs.get("precision", "fp16"),
                    "resolution": kwargs.get("resolution", 512),
                },
                "requirements": {
                    "python": ">=3.8",
                    "torch": ">=1.12.0",
                    "diffusers": ">=0.20.0",
                    "transformers": ">=4.20.0",
                },
                "usage": {
                    "example_prompt": kwargs.get(
                        "example_prompt", "a beautiful landscape"
                    ),
                    "recommended_settings": {
                        "guidance_scale": 7.5,
                        "num_inference_steps": 25,
                        "resolution": kwargs.get("resolution", 512),
                    },
                },
            }

            with open(package_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Create installation script
            install_script = f"""#!/bin/bash
# Installation script for {manifest['package_info']['name']}

echo "Installing {manifest['package_info']['name']}..."

# Check if AI_CACHE_ROOT is set
if [ -z "$AI_CACHE_ROOT" ]; then
    echo "Error: AI_CACHE_ROOT environment variable not set"
    exit 1
fi

# Create target directory
MODEL_ID="{kwargs.get('model_id', model_path.name)}"
TARGET_DIR="$AI_CACHE_ROOT/models/custom/$MODEL_ID"
mkdir -p "$TARGET_DIR"

# Copy model files
cp -r model/* "$TARGET_DIR/"

echo "Model installed to $TARGET_DIR"
echo "Usage: Load with model_id '$MODEL_ID'"
"""

            with open(package_dir / "install.sh", "w") as f:
                f.write(install_script)

            # Make install script executable
            (package_dir / "install.sh").chmod(0o755)

            # Create usage example
            usage_example = f"""# {manifest['package_info']['name']} Usage Example

from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipeline = StableDiffusionPipeline.from_pretrained(
    "{kwargs.get('base_model', 'runwayml/stable-diffusion-v1-5')}",
    torch_dtype=torch.float16
)

# Load LoRA weights (if applicable)
pipeline.load_lora_weights("path/to/model")

# Generate image
image = pipeline(
    "{kwargs.get('example_prompt', 'a beautiful landscape')}",
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

image.save("output.png")
"""

            with open(package_dir / "example.py", "w") as f:
                f.write(usage_example)

            # Create ZIP package
            zip_path = output_path.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)

            return {
                "format": "package",
                "package_dir": str(package_dir),
                "zip_file": str(zip_path),
                "files_included": len(list(package_dir.rglob("*"))),
                "zip_size_mb": zip_path.stat().st_size / 1024**2,
            }

        except Exception as e:
            raise ExportError(f"Package export failed: {e}")

    def _export_torchscript(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as TorchScript format"""
        try:
            base_model = kwargs.get("base_model", "runwayml/stable-diffusion-v1-5")

            # Load pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float32,  # TorchScript works better with fp32
            )

            # Load LoRA if applicable
            if (model_path / "adapter_model.safetensors").exists():
                pipeline.load_lora_weights(str(model_path))
                pipeline.fuse_lora()

            # Convert UNet to TorchScript
            unet = pipeline.unet
            unet.eval()

            # Create example inputs
            batch_size = kwargs.get("batch_size", 1)
            resolution = kwargs.get("resolution", 512)

            example_inputs = (
                torch.randn(batch_size, 4, resolution // 8, resolution // 8),
                torch.tensor([1]),
                torch.randn(batch_size, 77, 768),
            )

            # Trace the model
            traced_unet = torch.jit.trace(unet, example_inputs)

            # Save TorchScript model
            traced_unet.save(output_path)

            return {
                "format": "torchscript",
                "file_size_mb": output_path.stat().st_size / 1024**2,
                "optimized": True,
                "example_inputs_shape": [list(inp.shape) for inp in example_inputs],
            }

        except Exception as e:
            raise ExportError(f"TorchScript export failed: {e}")

    def _export_tflite(
        self, model_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Export as TensorFlow Lite format"""
        try:
            # This is a placeholder - actual TFLite conversion requires tf2onnx and additional tools
            raise ExportError("TensorFlow Lite export not yet implemented")

        except Exception as e:
            raise ExportError(f"TFLite export failed: {e}")

    def _add_metadata(
        self,
        model_path: Path,
        output_path: Path,
        export_format: str,
        export_result: Dict[str, Any],
    ):
        """Add metadata files to export"""
        try:
            if export_format in ["diffusers", "huggingface", "package"]:
                base_dir = output_path if output_path.is_dir() else output_path.parent
            else:
                base_dir = output_path.parent

            # Create metadata JSON
            metadata = {
                "export_info": {
                    "format": export_format,
                    "exported_at": time.time(),
                    "exporter_version": "1.0.0",
                    "source_path": str(model_path),
                },
                "model_info": {
                    "original_path": str(model_path),
                    "file_size_mb": self._get_path_size_mb(model_path),
                },
                "export_result": export_result,
            }

            with open(base_dir / "export_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ Failed to add metadata: {e}")

    def _create_model_card(self, model_path: Path, output_path: Path, **kwargs):
        """Create model card in Markdown format"""
        model_card = f"""# Model Card

## Model Information

- **Model Path**: {model_path}
- **Export Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Base Model**: {kwargs.get('base_model', 'Unknown')}
- **Model Type**: {kwargs.get('model_type', 'Diffusion Model')}

## Usage

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "{kwargs.get('base_model', 'runwayml/stable-diffusion-v1-5')}",
    torch_dtype=torch.float16
)

# Load this model (if LoRA)
pipeline.load_lora_weights("path/to/this/model")

# Generate image
image = pipeline(
    "your prompt here",
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]
```

## License

Please respect the license terms of the base model and any training data used.

## Disclaimer

This model is for research and creative purposes. Please use responsibly.
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(model_card)

    def _create_model_readme(self, model_path: Path, output_path: Path, **kwargs):
        """Create README for model"""
        readme_content = f"""# {kwargs.get('model_name', model_path.name)}

## Description

This model was exported from {model_path} using the Multi-Modal Lab export system.

## Model Details

- **Base Model**: {kwargs.get('base_model', 'Unknown')}
- **Model Type**: {kwargs.get('model_type', 'Diffusion Model')}
- **Resolution**: {kwargs.get('resolution', 512)}px
- **Precision**: {kwargs.get('precision', 'fp16')}

## Quick Start

```python
# Install requirements
pip install diffusers transformers torch

# Load and use the model
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("path/to/model")
image = pipeline("a beautiful sunset").images[0]
image.save("output.png")
```

## Configuration

Recommended settings:
- Guidance Scale: 7.5
- Inference Steps: 25-50
- Resolution: {kwargs.get('resolution', 512)}x{kwargs.get('resolution', 512)}

## Support

For issues and questions, please refer to the Multi-Modal Lab documentation.
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    def _get_path_size_mb(self, path: Path) -> float:
        """Get size of path in MB"""
        if path.is_file():
            return path.stat().st_size / 1024**2
        elif path.is_dir():
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            return total_size / 1024**2
        return 0.0

    def list_export_formats(self) -> Dict[str, Dict[str, Any]]:
        """List available export formats and their descriptions"""
        return {
            "diffusers": {
                "description": "Hugging Face Diffusers format (ready to use)",
                "file_type": "directory",
                "use_case": "Direct inference with Diffusers library",
            },
            "safetensors": {
                "description": "SafeTensors format (secure model weights)",
                "file_type": "file",
                "use_case": "Sharing model weights securely",
            },
            "checkpoint": {
                "description": "Traditional checkpoint format",
                "file_type": "file",
                "use_case": "Compatibility with older tools",
            },
            "onnx": {
                "description": "ONNX format for optimized inference",
                "file_type": "directory",
                "use_case": "Production deployment with ONNX Runtime",
            },
            "huggingface": {
                "description": "Hugging Face Hub ready format",
                "file_type": "directory",
                "use_case": "Upload to Hugging Face Hub",
            },
            "package": {
                "description": "Complete package with metadata and scripts",
                "file_type": "zip",
                "use_case": "Distribution and easy installation",
            },
            "torchscript": {
                "description": "TorchScript format for deployment",
                "file_type": "file",
                "use_case": "Production deployment without Python dependencies",
            },
            "tflite": {
                "description": "TensorFlow Lite format (experimental)",
                "file_type": "file",
                "use_case": "Mobile and embedded deployment",
            },
        }


# Global instance
_model_exporter = None


def get_model_exporter() -> ModelExporter:
    """Get global model exporter instance"""
    global _model_exporter
    if _model_exporter is None:
        _model_exporter = ModelExporter()
    return _model_exporter
