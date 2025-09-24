# core/safety/validator.py
"""
Input Validation System
Validates API inputs, parameters, and file uploads
"""

import re
import logging
import io
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import mimetypes
from PIL import Image
import torch

from ..config import get_config
from ..exceptions import ValidationError, SafetyError

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation"""

    def __init__(self):
        self.config = get_config()

        # Validation rules
        self.max_prompt_length = getattr(self.config.safety, "max_prompt_length", 500)
        self.max_negative_prompt_length = getattr(
            self.config.safety, "max_negative_prompt_length", 300
        )
        self.compiled_patterns = self._compile_patterns()
        self.min_image_size = 64
        self.max_image_size = 2048
        self.allowed_image_formats = {"JPEG", "PNG", "WEBP", "BMP"}
        self.max_file_size_mb = 50

        # Text patterns
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # Script injection
            r"javascript:",  # JavaScript URLs
            r"data:text/html",  # HTML data URLs
            r"<?php",  # PHP code
            r"<%.*?%>",  # ASP/JSP code
        ]

        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.suspicious_patterns
        ]

    def _compile_patterns(self) -> List[Any]:
        """Compile regex patterns for validation"""
        patterns = []
        try:
            # 基本的敏感內容模式
            sensitive_patterns = [
                r"\b(?:nude|naked|nsfw|sexual)\b",
                r"\b(?:child|minor|underage)\b",
                # 加入更多模式...
            ]
            patterns = [
                re.compile(pattern, re.IGNORECASE) for pattern in sensitive_patterns
            ]
        except Exception as e:
            logger.warning(f"Pattern compilation failed: {e}")
        return patterns

    def validate_prompt(self, prompt: str, field_name: str = "prompt") -> str:
        """Validate text prompt"""
        if not prompt or not isinstance(prompt, str):
            raise ValidationError(
                "parameter_error", f"{field_name} must be a non-empty string"
            )

        # Length check
        max_length = (
            self.max_prompt_length
            if field_name == "prompt"
            else self.max_negative_prompt_length
        )
        if len(prompt) > max_length:
            raise ValidationError(
                "parameter_error",
                f"{field_name} exceeds maximum length of {max_length} characters",
            )

        # Check for suspicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                raise SafetyError(f"Suspicious content detected in {field_name}")

        # Basic sanitization
        cleaned_prompt = prompt.strip()

        # Remove null bytes and control characters
        cleaned_prompt = "".join(
            char for char in cleaned_prompt if ord(char) >= 32 or char in "\n\r\t"
        )

        return cleaned_prompt

    def validate_image_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Validate and clamp image dimensions"""
        try:
            width = int(width)
            height = int(height)
        except (ValueError, TypeError):
            raise ValidationError(
                "parameter_error", "Width and height must be integers"
            )

        # Size limits
        if width < self.min_image_size or height < self.min_image_size:
            raise ValidationError(
                "parameter_error",
                f"Image dimensions must be at least {self.min_image_size}x{self.min_image_size}",
            )

        if width > self.max_image_size or height > self.max_image_size:
            raise ValidationError(
                "parameter_error",
                f"Image dimensions must not exceed {self.max_image_size}x{self.max_image_size}",
            )

        # Must be divisible by 8 for most diffusion models
        if width % 8 != 0 or height % 8 != 0:
            width = (width // 8) * 8
            height = (height // 8) * 8
            logger.info(f"Adjusted dimensions to {width}x{height} (divisible by 8)")

        return width, height

    def validate_numeric_param(
        self,
        value: Union[int, float],
        param_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        param_type: type = float,
    ) -> Union[int, float]:
        """Validate numeric parameters"""
        try:
            if param_type == int:
                validated_value = int(value)
            else:
                validated_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                "parameter_error", f"{param_name} must be a {param_type.__name__}"
            )

        if min_val is not None and validated_value < min_val:
            raise ValidationError(
                "parameter_error", f"{param_name} must be >= {min_val}"
            )

        if max_val is not None and validated_value > max_val:
            raise ValidationError(
                "parameter_error", f"{param_name} must be <= {max_val}"
            )

        return validated_value

    def validate_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generation parameters - 修正截斷邏輯"""
        try:
            validated_params = params.copy()

            # 更嚴格的屬性存取和預設值
            max_resolution = getattr(self.config.safety, "max_resolution", 1024)
            max_steps = getattr(self.config.safety, "max_inference_steps", 50)
            max_guidance = getattr(self.config.safety, "max_guidance_scale", 20.0)

            # 確保截斷邏輯正確執行
            if "width" in params:
                original_width = params["width"]
                if original_width > max_resolution:
                    validated_params["width"] = max_resolution
                    logger.warning(
                        f"⚠️ Width clamped from {original_width} to {max_resolution}"
                    )

            if "height" in params:
                original_height = params["height"]
                if original_height > max_resolution:
                    validated_params["height"] = max_resolution
                    logger.warning(
                        f"⚠️ Height clamped from {original_height} to {max_resolution}"
                    )

            if "num_inference_steps" in params:
                original_steps = params["num_inference_steps"]
                if original_steps > max_steps:
                    validated_params["num_inference_steps"] = max_steps
                    logger.warning(
                        f"⚠️ Steps clamped from {original_steps} to {max_steps}"
                    )

            if "guidance_scale" in params:
                original_guidance = params["guidance_scale"]
                if original_guidance > max_guidance:
                    validated_params["guidance_scale"] = max_guidance
                    logger.warning(
                        f"⚠️ Guidance scale clamped from {original_guidance} to {max_guidance}"
                    )

            return validated_params

        except Exception as e:
            logger.error(f"❌ Parameter validation failed: {e}")
            raise ValidationError(
                "parameter_error", f"Parameter validation failed: {e}"
            )

    def validate_image_file(
        self, image_data: Union[bytes, Image.Image, str]
    ) -> Image.Image:
        """Validate uploaded image file"""
        try:
            # Handle different input types
            if isinstance(image_data, str):
                if image_data.startswith("data:image"):
                    # Base64 data URL
                    import base64

                    header, data = image_data.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # File path
                    image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                # Raw bytes
                if len(image_data) > self.max_file_size_mb * 1024 * 1024:
                    raise ValidationError(
                        "image_file_error",
                        f"Image file too large (max {self.max_file_size_mb}MB)",
                    )
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValidationError("image_file_error", "Invalid image data format")

            # Format validation
            if image.format not in self.allowed_image_formats:
                raise ValidationError(
                    "image_file_error", f"Unsupported image format: {image.format}"
                )

            # Size validation
            width, height = image.size
            self.validate_image_dimensions(width, height)

            # Convert to RGB if needed
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            return image

        except Exception as e:
            if isinstance(e, (ValidationError, SafetyError)):
                raise
            raise ValidationError("image_file_error", f"Failed to validate image: {e}")

    def validate_model_name(self, model_name: str) -> str:
        """Validate model name/ID"""
        if not model_name or not isinstance(model_name, str):
            raise ValidationError(
                "model_name_error", "Model name must be a non-empty string"
            )

        # Remove potentially dangerous characters
        cleaned_name = re.sub(r"[^\w\-\./]", "", model_name)

        if not cleaned_name:
            raise ValidationError("model_name_error", "Invalid model name format")

        # Length check
        if len(cleaned_name) > 200:
            raise ValidationError("model_name_error", "Model name too long")

        return cleaned_name

    def validate_lora_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LoRA-specific parameters"""
        validated = {}

        # LoRA scale/weight
        if "lora_scale" in params:
            validated["lora_scale"] = self.validate_numeric_param(
                params["lora_scale"], "lora_scale", min_val=0.0, max_val=2.0
            )

        # LoRA model path/name
        if "lora_path" in params:
            validated["lora_path"] = self.validate_model_name(params["lora_path"])

        # Training parameters
        if "rank" in params:
            validated["rank"] = self.validate_numeric_param(
                params["rank"], "rank", min_val=1, max_val=1024, param_type=int
            )

        if "alpha" in params:
            validated["alpha"] = self.validate_numeric_param(
                params["alpha"], "alpha", min_val=1, max_val=1024, param_type=int
            )

        if "learning_rate" in params:
            validated["learning_rate"] = self.validate_numeric_param(
                params["learning_rate"], "learning_rate", min_val=1e-6, max_val=1e-2
            )

        return validated

    def validate_chat_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chat/LLM parameters"""
        validated = {}

        # Message content
        if "message" in params:
            validated["message"] = self.validate_prompt(params["message"], "message")

        if "max_length" in params:
            validated["max_length"] = min(params["max_length"], 2048)

        # Max tokens
        if "max_tokens" in params:
            validated["max_tokens"] = self.validate_numeric_param(
                params["max_tokens"],
                "max_tokens",
                min_val=1,
                max_val=4000,
                param_type=int,
            )

        # Temperature
        if "temperature" in params:
            validated["temperature"] = self.validate_numeric_param(
                params["temperature"], "temperature", min_val=0.0, max_val=2.0
            )

        # Top-p
        if "top_p" in params:
            validated["top_p"] = self.validate_numeric_param(
                params["top_p"], "top_p", min_val=0.0, max_val=1.0
            )

        return validated

    def validate_batch_request(
        self, batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate batch generation request"""
        if not isinstance(batch_data, list):
            raise ValidationError("Batch data must be a list")

        if len(batch_data) == 0:
            raise ValidationError("Batch_data_error", "Batch data cannot be empty")

        if len(batch_data) > 100:  # Reasonable batch size limit
            raise ValidationError(
                "Batch_data_error", "Batch size too large (max 100 items)"
            )

        validated_batch = []
        for i, item in enumerate(batch_data):
            try:
                if not isinstance(item, dict):
                    raise ValidationError(
                        "Batch_data_error", f"Batch item {i} must be a dictionary"
                    )

                validated_item = self.validate_generation_params(item)
                validated_batch.append(validated_item)

            except Exception as e:
                logger.warning(f"⚠️ Skipping invalid batch item: {e}")
                raise ValidationError(
                    "validation_error", f"Validation failed for batch item {i}: {e}"
                )

        return validated_batch

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "unnamed"

        # Remove path separators and dangerous characters
        sanitized = re.sub(r"[^\w\-_\.]", "_", filename)

        # Remove leading dots
        sanitized = re.sub(r"^\.+", "", sanitized)

        # Limit length
        if len(sanitized) > 200:
            name, ext = (
                sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            )
            sanitized = name[:190] + ("." + ext if ext else "")

        return sanitized or "unnamed"


# Global instance
_input_validator = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance"""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator
