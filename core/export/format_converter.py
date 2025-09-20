# core/export/format_converter.py
"""
Format Conversion Utilities
Convert between different model formats and data standards
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import yaml
import shutil
import tempfile
from PIL import Image
import cv2

from ..config import get_config
from ..exceptions import ConversionError, ValidationError
from ..utils import get_cache_manager

logger = logging.getLogger(__name__)


class FormatConverter:
    """Convert between different model and data formats"""

    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()

        # Supported conversions (source_format, target_format) -> method
        self.conversions = {
            # Model format conversions
            ("pytorch", "safetensors"): self._pytorch_to_safetensors,
            ("safetensors", "pytorch"): self._safetensors_to_pytorch,
            ("diffusers", "checkpoint"): self._diffusers_to_checkpoint,
            ("checkpoint", "diffusers"): self._checkpoint_to_diffusers,
            ("lora", "full_model"): self._lora_to_full_model,
            ("onnx", "tensorrt"): self._onnx_to_tensorrt,
            # Data format conversions
            ("json", "yaml"): self._json_to_yaml,
            ("yaml", "json"): self._yaml_to_json,
            ("csv", "json"): self._csv_to_json,
            ("json", "csv"): self._json_to_csv,
            # Image format conversions
            ("image", "tensor"): self._image_to_tensor,
            ("tensor", "image"): self._tensor_to_image,
            ("image", "numpy"): self._image_to_numpy,
            ("numpy", "image"): self._numpy_to_image,
            # Video format conversions
            ("video", "frames"): self._video_to_frames,
            ("frames", "video"): self._frames_to_video,
            # Audio format conversions
            ("audio", "spectrogram"): self._audio_to_spectrogram,
            ("spectrogram", "audio"): self._spectrogram_to_audio,
            # Configuration conversions
            ("config", "env"): self._config_to_env,
            ("env", "config"): self._env_to_config,
        }

    def convert(
        self,
        input_path: str,
        output_path: str,
        source_format: str,
        target_format: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert from source format to target format"""

        conversion_key = (source_format, target_format)
        if conversion_key not in self.conversions:
            raise ValidationError(
                f"Conversion not supported: {source_format} -> {target_format}"
            )

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise ValidationError(f"Input path does not exist: {input_path}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔄 Converting {source_format} -> {target_format}")

        try:
            conversion_func = self.conversions[conversion_key]
            result = conversion_func(input_path, output_path, **kwargs)

            logger.info(f"✅ Conversion completed: {output_path}")
            return {
                "source_format": source_format,
                "target_format": target_format,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "success": True,
                **result,
            }

        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            raise ConversionError(f"Format conversion failed: {e}")

    # Model Format Conversions

    def _pytorch_to_safetensors(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert PyTorch weights to SafeTensors"""
        try:
            from safetensors.torch import save_file

            # Load PyTorch weights
            weights = torch.load(input_path, map_location="cpu")

            # Handle different PyTorch file structures
            if isinstance(weights, dict):
                if "state_dict" in weights:
                    state_dict = weights["state_dict"]
                elif "model" in weights:
                    state_dict = weights["model"]
                else:
                    state_dict = weights
            else:
                raise ConversionError("Unsupported PyTorch file structure")

            # Clean up state dict - remove any non-tensor entries
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    cleaned_state_dict[key] = value
                else:
                    logger.warning(f"Skipping non-tensor entry: {key}")

            # Save as SafeTensors
            save_file(cleaned_state_dict, output_path)

            return {
                "parameters": len(cleaned_state_dict),
                "file_size_mb": output_path.stat().st_size / 1024**2,
                "tensor_count": len(cleaned_state_dict),
            }

        except Exception as e:
            raise ConversionError(f"PyTorch to SafeTensors conversion failed: {e}")

    def _safetensors_to_pytorch(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert SafeTensors to PyTorch format"""
        try:
            from safetensors import safe_open

            # Load SafeTensors
            state_dict = {}
            with safe_open(input_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            # Create PyTorch checkpoint structure
            checkpoint = {
                "state_dict": state_dict,
                "meta": {
                    "converted_from": "safetensors",
                    "original_file": str(input_path),
                    "conversion_time": torch.tensor(0.0),  # Placeholder
                },
            }

            # Save as PyTorch
            torch.save(checkpoint, output_path)

            return {
                "parameters": len(state_dict),
                "file_size_mb": output_path.stat().st_size / 1024**2,
                "includes_metadata": True,
            }

        except Exception as e:
            raise ConversionError(f"SafeTensors to PyTorch conversion failed: {e}")

    def _diffusers_to_checkpoint(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert Diffusers pipeline to checkpoint format"""
        try:
            from diffusers import StableDiffusionPipeline

            # Load Diffusers pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                input_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Create checkpoint structure
            checkpoint = {
                "state_dict": {},
                "model_config": {
                    "attention_resolutions": "32,16,8",
                    "class_embed_type": None,
                    "conditioning_key": "crossattn",
                    "diffusion_steps": 1000,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "num_heads": 8,
                    "num_res_blocks": 2,
                    "resblock_updown": False,
                    "use_spatial_transformer": True,
                    "transformer_depth": 1,
                    "converted_from": "diffusers",
                },
            }

            # Extract UNet state dict
            component_count = 0
            for name, param in pipeline.unet.named_parameters():
                checkpoint_name = f"model.diffusion_model.{name}"
                checkpoint["state_dict"][checkpoint_name] = param.cpu()
                component_count += 1

            # Extract text encoder if requested
            if kwargs.get("include_text_encoder", False):
                for name, param in pipeline.text_encoder.named_parameters():
                    checkpoint_name = f"cond_stage_model.transformer.{name}"
                    checkpoint["state_dict"][checkpoint_name] = param.cpu()
                    component_count += 1

            # Extract VAE if requested
            if kwargs.get("include_vae", False):
                for name, param in pipeline.vae.named_parameters():
                    checkpoint_name = f"first_stage_model.{name}"
                    checkpoint["state_dict"][checkpoint_name] = param.cpu()
                    component_count += 1

            # Save checkpoint
            torch.save(checkpoint, output_path)

            return {
                "components": ["unet"]
                + (["text_encoder"] if kwargs.get("include_text_encoder") else [])
                + (["vae"] if kwargs.get("include_vae") else []),
                "parameters": len(checkpoint["state_dict"]),
                "file_size_mb": output_path.stat().st_size / 1024**2,
                "component_count": component_count,
            }

        except Exception as e:
            raise ConversionError(f"Diffusers to checkpoint conversion failed: {e}")

    def _checkpoint_to_diffusers(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert checkpoint to Diffusers format"""
        try:
            from diffusers import StableDiffusionPipeline

            # Load checkpoint
            checkpoint = torch.load(input_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Extract UNet state dict
            unet_state_dict = {}
            text_encoder_state_dict = {}
            vae_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith("model.diffusion_model."):
                    new_key = key.replace("model.diffusion_model.", "")
                    unet_state_dict[new_key] = value
                elif key.startswith("cond_stage_model.transformer."):
                    new_key = key.replace("cond_stage_model.transformer.", "")
                    text_encoder_state_dict[new_key] = value
                elif key.startswith("first_stage_model."):
                    new_key = key.replace("first_stage_model.", "")
                    vae_state_dict[new_key] = value

            # Load base pipeline
            base_model = kwargs.get("base_model", "runwayml/stable-diffusion-v1-5")
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Load converted weights
            components_loaded = []

            if unet_state_dict:
                pipeline.unet.load_state_dict(unet_state_dict, strict=False)
                components_loaded.append("unet")

            if text_encoder_state_dict and kwargs.get("load_text_encoder", True):
                try:
                    pipeline.text_encoder.load_state_dict(
                        text_encoder_state_dict, strict=False
                    )
                    components_loaded.append("text_encoder")
                except Exception as e:
                    logger.warning(f"Failed to load text encoder: {e}")

            if vae_state_dict and kwargs.get("load_vae", True):
                try:
                    pipeline.vae.load_state_dict(vae_state_dict, strict=False)
                    components_loaded.append("vae")
                except Exception as e:
                    logger.warning(f"Failed to load VAE: {e}")

            # Save as Diffusers
            pipeline.save_pretrained(output_path)

            return {
                "base_model": base_model,
                "components_loaded": components_loaded,
                "converted_parameters": len(unet_state_dict)
                + len(text_encoder_state_dict)
                + len(vae_state_dict),
            }

        except Exception as e:
            raise ConversionError(f"Checkpoint to Diffusers conversion failed: {e}")

    def _lora_to_full_model(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Merge LoRA weights into full model"""
        try:
            from diffusers import StableDiffusionPipeline

            base_model = kwargs.get("base_model")
            if not base_model:
                raise ConversionError(
                    "base_model parameter required for LoRA to full model conversion"
                )

            # Load base pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Load LoRA weights
            pipeline.load_lora_weights(str(input_path))

            # Fuse LoRA into the model
            pipeline.fuse_lora()

            # Save merged model
            pipeline.save_pretrained(output_path)

            return {
                "base_model": base_model,
                "lora_fused": True,
                "output_format": "diffusers",
                "lora_scale": kwargs.get("lora_scale", 1.0),
            }

        except Exception as e:
            raise ConversionError(f"LoRA to full model conversion failed: {e}")

    def _onnx_to_tensorrt(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert ONNX to TensorRT format"""
        try:
            # This requires TensorRT to be installed
            import tensorrt as trt

            # Create TensorRT logger
            logger_trt = trt.Logger(trt.Logger.WARNING)

            # Create builder
            builder = trt.Builder(logger_trt)
            config = builder.create_builder_config()

            # Set optimization profile
            profile = builder.create_optimization_profile()

            # Parse ONNX model
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger_trt)

            with open(input_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    errors = []
                    for error in range(parser.num_errors):
                        errors.append(parser.get_error(error))
                    raise ConversionError(f"Failed to parse ONNX file: {errors}")

            # Build engine
            engine = builder.build_engine(network, config)

            if engine is None:
                raise ConversionError("Failed to build TensorRT engine")

            # Save engine
            with open(output_path, "wb") as f:
                f.write(engine.serialize())

            return {
                "format": "tensorrt",
                "file_size_mb": output_path.stat().st_size / 1024**2,
                "optimized": True,
            }

        except ImportError:
            raise ConversionError(
                "TensorRT not installed. Please install tensorrt package."
            )
        except Exception as e:
            raise ConversionError(f"ONNX to TensorRT conversion failed: {e}")

    # Data Format Conversions

    def _json_to_yaml(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert JSON to YAML format"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data, f, default_flow_style=False, indent=2, allow_unicode=True
                )

            return {
                "converted_keys": len(data) if isinstance(data, dict) else 1,
                "data_type": type(data).__name__,
            }

        except Exception as e:
            raise ConversionError(f"JSON to YAML conversion failed: {e}")

    def _yaml_to_json(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert YAML to JSON format"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                "converted_keys": len(data) if isinstance(data, dict) else 1,
                "data_type": type(data).__name__,
            }

        except Exception as e:
            raise ConversionError(f"YAML to JSON conversion failed: {e}")

    def _csv_to_json(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert CSV to JSON format"""
        try:
            import pandas as pd

            # Read CSV
            df = pd.read_csv(input_path)

            # Convert to JSON
            if kwargs.get("records_format", True):
                # Array of objects format
                data = df.to_dict("records")
            else:
                # Columns format
                data = df.to_dict()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            return {
                "rows": len(df),
                "columns": len(df.columns),
                "format": (
                    "records" if kwargs.get("records_format", True) else "columns"
                ),
            }

        except Exception as e:
            raise ConversionError(f"CSV to JSON conversion failed: {e}")

    def _json_to_csv(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert JSON to CSV format"""
        try:
            import pandas as pd

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ConversionError(
                    "JSON data must be list or dict for CSV conversion"
                )

            # Save as CSV
            df.to_csv(output_path, index=False)

            return {"rows": len(df), "columns": len(df.columns)}

        except Exception as e:
            raise ConversionError(f"JSON to CSV conversion failed: {e}")

    # Image Format Conversions

    def _image_to_tensor(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert image to tensor format"""
        try:
            # Load image
            image = Image.open(input_path).convert("RGB")

            # Convert to tensor
            image_array = np.array(image).astype(np.float32) / 255.0

            # Normalize if requested
            if kwargs.get("normalize", True):
                mean = kwargs.get("mean", [0.5, 0.5, 0.5])
                std = kwargs.get("std", [0.5, 0.5, 0.5])
                image_array = (image_array - np.array(mean)) / np.array(std)

            # Convert to torch tensor
            tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # CHW format

            # Add batch dimension if requested
            if kwargs.get("add_batch_dim", False):
                tensor = tensor.unsqueeze(0)

            # Save tensor
            torch.save(tensor, output_path)

            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "normalized": kwargs.get("normalize", True),
                "original_size": image.size,
            }

        except Exception as e:
            raise ConversionError(f"Image to tensor conversion failed: {e}")

    def _tensor_to_image(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert tensor to image format"""
        try:
            # Load tensor
            tensor = torch.load(input_path, map_location="cpu")

            # Handle batch dimension
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # Remove batch dim

            # Convert from CHW to HWC
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)

            # Denormalize if needed
            if kwargs.get("denormalize", True):
                mean = kwargs.get("mean", [0.5, 0.5, 0.5])
                std = kwargs.get("std", [0.5, 0.5, 0.5])
                tensor = tensor * torch.tensor(std) + torch.tensor(mean)

            # Clamp and convert to uint8
            tensor = torch.clamp(tensor, 0, 1)
            image_array = (tensor * 255).numpy().astype(np.uint8)

            # Create PIL image
            image = Image.fromarray(image_array)

            # Save image
            format_name = output_path.suffix[1:].upper() or "PNG"
            image.save(output_path, format=format_name)

            return {
                "image_size": image.size,
                "mode": image.mode,
                "format": format_name,
                "file_size_kb": output_path.stat().st_size / 1024,
            }

        except Exception as e:
            raise ConversionError(f"Tensor to image conversion failed: {e}")

    def _image_to_numpy(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert image to NumPy array"""
        try:
            # Load image
            image = Image.open(input_path).convert("RGB")

            # Convert to numpy
            array = np.array(image)

            # Normalize if requested
            if kwargs.get("normalize", False):
                array = array.astype(np.float32) / 255.0

            # Save as numpy
            np.save(output_path, array)

            return {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "normalized": kwargs.get("normalize", False),
            }

        except Exception as e:
            raise ConversionError(f"Image to NumPy conversion failed: {e}")

    def _numpy_to_image(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert NumPy array to image"""
        try:
            # Load numpy array
            array = np.load(input_path)

            # Handle normalization
            if array.dtype == np.float32 or array.dtype == np.float64:
                if array.max() <= 1.0:
                    # Assume normalized, convert to 0-255
                    array = (array * 255).astype(np.uint8)
                else:
                    # Assume already in 0-255 range
                    array = array.astype(np.uint8)

            # Create PIL image
            image = Image.fromarray(array)

            # Save image
            format_name = output_path.suffix[1:].upper() or "PNG"
            image.save(output_path, format=format_name)

            return {"image_size": image.size, "mode": image.mode, "format": format_name}

        except Exception as e:
            raise ConversionError(f"NumPy to image conversion failed: {e}")

    # Video Format Conversions

    def _video_to_frames(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Extract frames from video"""
        try:
            import cv2

            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(str(input_path))

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Extract frames
            step = kwargs.get("frame_step", 1)
            max_frames = kwargs.get("max_frames", None)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % step == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Save frame
                    frame_filename = output_path / f"frame_{frame_count:06d}.png"
                    Image.fromarray(frame_rgb).save(frame_filename)

                    if max_frames and (frame_count // step) >= max_frames:
                        break

                frame_count += 1

            cap.release()

            extracted_frames = len(list(output_path.glob("frame_*.png")))

            return {
                "total_video_frames": total_frames,
                "extracted_frames": extracted_frames,
                "fps": fps,
                "frame_step": step,
            }

        except Exception as e:
            raise ConversionError(f"Video to frames conversion failed: {e}")

    def _frames_to_video(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Combine frames into video"""
        try:
            import cv2

            # Get frame files
            frame_files = sorted(list(input_path.glob("*.png"))) or sorted(
                list(input_path.glob("*.jpg"))
            )

            if not frame_files:
                raise ConversionError("No frame files found")

            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width, _ = first_frame.shape

            # Create video writer
            fps = kwargs.get("fps", 30)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Write frames
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)

            out.release()

            return {
                "frame_count": len(frame_files),
                "fps": fps,
                "resolution": (width, height),
                "duration_seconds": len(frame_files) / fps,
            }

        except Exception as e:
            raise ConversionError(f"Frames to video conversion failed: {e}")

    # Audio Format Conversions

    def _audio_to_spectrogram(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert audio to spectrogram"""
        try:
            import librosa
            import matplotlib.pyplot as plt

            # Load audio
            y, sr = librosa.load(input_path, sr=kwargs.get("sample_rate", None))

            # Compute spectrogram
            hop_length = kwargs.get("hop_length", 512)
            n_fft = kwargs.get("n_fft", 2048)

            D = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Save as image
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(
                S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz"
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title("Spectrogram")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Also save raw data
            raw_output = output_path.with_suffix(".npy")
            np.save(raw_output, S_db)

            return {
                "sample_rate": sr,
                "duration_seconds": len(y) / sr,
                "hop_length": hop_length,
                "n_fft": n_fft,
                "spectrogram_shape": S_db.shape,
            }

        except ImportError:
            raise ConversionError(
                "librosa not installed. Please install: pip install librosa"
            )
        except Exception as e:
            raise ConversionError(f"Audio to spectrogram conversion failed: {e}")

    def _spectrogram_to_audio(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert spectrogram back to audio"""
        try:
            import librosa
            import soundfile as sf

            # Load spectrogram data
            if input_path.suffix == ".npy":
                S_db = np.load(input_path)
            else:
                raise ConversionError(
                    "Spectrogram must be in .npy format for audio reconstruction"
                )

            # Convert back to amplitude
            S = librosa.db_to_amplitude(S_db)

            # Reconstruct audio using Griffin-Lim algorithm
            hop_length = kwargs.get("hop_length", 512)
            n_iter = kwargs.get("n_iter", 32)

            y = librosa.griffinlim(S, hop_length=hop_length, n_iter=n_iter)

            # Save audio
            sr = kwargs.get("sample_rate", 22050)
            sf.write(output_path, y, sr)

            return {
                "sample_rate": sr,
                "duration_seconds": len(y) / sr,
                "reconstruction_iterations": n_iter,
            }

        except ImportError:
            raise ConversionError("librosa and soundfile not installed")
        except Exception as e:
            raise ConversionError(f"Spectrogram to audio conversion failed: {e}")

    # Configuration Conversions

    def _config_to_env(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert config file to environment variables"""
        try:
            # Load config
            if input_path.suffix == ".json":
                with open(input_path, "r") as f:
                    config = json.load(f)
            elif input_path.suffix in [".yml", ".yaml"]:
                with open(input_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                raise ConversionError("Config file must be JSON or YAML")

            # Flatten config to environment variables
            env_vars = []

            def flatten_dict(d, prefix=""):
                for key, value in d.items():
                    env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                    if isinstance(value, dict):
                        flatten_dict(value, f"{env_key}_")
                    else:
                        env_vars.append(f"{env_key}={value}")

            flatten_dict(config)

            # Write to .env file
            with open(output_path, "w") as f:
                f.write("\n".join(env_vars))

            return {
                "variables_count": len(env_vars),
                "config_format": input_path.suffix,
            }

        except Exception as e:
            raise ConversionError(f"Config to env conversion failed: {e}")

    def _env_to_config(
        self, input_path: Path, output_path: Path, **kwargs
    ) -> Dict[str, Any]:
        """Convert environment variables to config file"""
        try:
            # Read .env file
            env_vars = {}
            with open(input_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            env_vars[key.strip()] = value.strip()

            # Convert to nested config structure
            config = {}
            for key, value in env_vars.items():
                # Split by underscore to create nested structure
                parts = key.lower().split("_")
                current = config

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Try to convert value to appropriate type
                try:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        value = float(value)
                except:
                    pass  # Keep as string

                current[parts[-1]] = value

            # Save as requested format
            if output_path.suffix == ".json":
                with open(output_path, "w") as f:
                    json.dump(config, f, indent=2)
            elif output_path.suffix in [".yml", ".yaml"]:
                with open(output_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                raise ConversionError("Output must be JSON or YAML")

            return {
                "variables_count": len(env_vars),
                "output_format": output_path.suffix,
                "nested_levels": self._count_nested_levels(config),
            }

        except Exception as e:
            raise ConversionError(f"Env to config conversion failed: {e}")

    def _count_nested_levels(self, d: dict, level: int = 0) -> int:
        """Count maximum nesting levels in dictionary"""
        if not isinstance(d, dict):
            return level

        max_level = level
        for value in d.values():
            if isinstance(value, dict):
                max_level = max(max_level, self._count_nested_levels(value, level + 1))

        return max_level

    def list_supported_conversions(self) -> List[Dict[str, str]]:
        """List all supported format conversions"""
        conversions = []

        for source, target in self.conversions.keys():
            conversions.append(
                {
                    "source": source,
                    "target": target,
                    "description": f"Convert {source} to {target}",
                    "category": self._get_conversion_category(source, target),
                }
            )

        return conversions

    def _get_conversion_category(self, source: str, target: str) -> str:
        """Get category for conversion"""
        model_formats = {
            "pytorch",
            "safetensors",
            "diffusers",
            "checkpoint",
            "lora",
            "onnx",
            "tensorrt",
        }
        data_formats = {"json", "yaml", "csv"}
        image_formats = {"image", "tensor", "numpy"}
        video_formats = {"video", "frames"}
        audio_formats = {"audio", "spectrogram"}
        config_formats = {"config", "env"}

        if source in model_formats or target in model_formats:
            return "model"
        elif source in data_formats or target in data_formats:
            return "data"
        elif source in image_formats or target in image_formats:
            return "image"
        elif source in video_formats or target in video_formats:
            return "video"
        elif source in audio_formats or target in audio_formats:
            return "audio"
        elif source in config_formats or target in config_formats:
            return "config"
        else:
            return "other"

    def get_conversion_info(
        self, source_format: str, target_format: str
    ) -> Dict[str, Any]:
        """Get information about a specific conversion"""
        conversion_key = (source_format, target_format)

        if conversion_key not in self.conversions:
            return {
                "supported": False,
                "error": f"Conversion {source_format} -> {target_format} not supported",
            }

        # Define conversion requirements and options
        conversion_info = {
            "supported": True,
            "category": self._get_conversion_category(source_format, target_format),
            "description": f"Convert {source_format} format to {target_format} format",
            "requirements": [],
            "optional_parameters": [],
            "output_type": "file",
        }

        # Add specific requirements and parameters
        if conversion_key == ("pytorch", "safetensors"):
            conversion_info["requirements"] = ["safetensors"]
            conversion_info["description"] = (
                "Convert PyTorch weights to secure SafeTensors format"
            )

        elif conversion_key == ("diffusers", "checkpoint"):
            conversion_info["optional_parameters"] = [
                "include_text_encoder",
                "include_vae",
            ]
            conversion_info["description"] = (
                "Convert Diffusers pipeline to traditional checkpoint"
            )

        elif conversion_key == ("lora", "full_model"):
            conversion_info["requirements"] = ["base_model"]
            conversion_info["description"] = "Merge LoRA weights into full model"

        elif conversion_key == ("onnx", "tensorrt"):
            conversion_info["requirements"] = ["tensorrt"]
            conversion_info["description"] = (
                "Optimize ONNX model for TensorRT inference"
            )

        elif conversion_key == ("image", "tensor"):
            conversion_info["optional_parameters"] = [
                "normalize",
                "mean",
                "std",
                "add_batch_dim",
            ]
            conversion_info["description"] = "Convert image to PyTorch tensor"

        elif conversion_key == ("video", "frames"):
            conversion_info["optional_parameters"] = ["frame_step", "max_frames"]
            conversion_info["output_type"] = "directory"
            conversion_info["requirements"] = ["opencv-python"]

        elif conversion_key == ("audio", "spectrogram"):
            conversion_info["optional_parameters"] = [
                "sample_rate",
                "hop_length",
                "n_fft",
            ]
            conversion_info["requirements"] = ["librosa", "matplotlib"]

        return conversion_info

    def batch_convert(self, conversions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform batch conversions"""
        results = []

        for i, conversion in enumerate(conversions):
            try:
                result = self.convert(**conversion)
                result["batch_index"] = i
                result["status"] = "success"
                results.append(result)

            except Exception as e:
                error_result = {
                    "batch_index": i,
                    "status": "failed",
                    "error": str(e),
                    "input_path": conversion.get("input_path"),
                    "output_path": conversion.get("output_path"),
                    "source_format": conversion.get("source_format"),
                    "target_format": conversion.get("target_format"),
                }
                results.append(error_result)
                logger.error(f"Batch conversion {i} failed: {e}")

        successful = len([r for r in results if r["status"] == "success"])
        failed = len(results) - successful

        logger.info(
            f"Batch conversion completed: {successful} successful, {failed} failed"
        )

        return results

    def validate_conversion_input(
        self, input_path: str, source_format: str
    ) -> Dict[str, Any]:
        """Validate input for conversion"""
        input_path = Path(input_path)

        validation = {"valid": True, "errors": [], "warnings": [], "file_info": {}}

        # Check if file exists
        if not input_path.exists():
            validation["valid"] = False
            validation["errors"].append(f"Input file does not exist: {input_path}")
            return validation

        # Get file info
        if input_path.is_file():
            validation["file_info"] = {
                "size_mb": input_path.stat().st_size / 1024**2,
                "extension": input_path.suffix,
                "is_directory": False,
            }
        elif input_path.is_dir():
            files = list(input_path.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            validation["file_info"] = {
                "size_mb": total_size / 1024**2,
                "file_count": len([f for f in files if f.is_file()]),
                "is_directory": True,
            }

        # Format-specific validation
        try:
            if source_format == "pytorch":
                # Try to load PyTorch file
                torch.load(input_path, map_location="cpu")

            elif source_format == "safetensors":
                from safetensors import safe_open

                with safe_open(input_path, framework="pt"):
                    pass

            elif source_format == "json":
                with open(input_path, "r") as f:
                    json.load(f)

            elif source_format in ["yaml", "yml"]:
                with open(input_path, "r") as f:
                    yaml.safe_load(f)

            elif source_format == "image":
                Image.open(input_path)

        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(
                f"Failed to validate {source_format} format: {e}"
            )

        # Size warnings
        if validation["file_info"].get("size_mb", 0) > 1000:  # 1GB
            validation["warnings"].append("Large file size may cause memory issues")

        return validation


# Global instance
_format_converter = None


def get_format_converter() -> FormatConverter:
    """Get global format converter instance"""
    global _format_converter
    if _format_converter is None:
        _format_converter = FormatConverter()
    return _format_converter
