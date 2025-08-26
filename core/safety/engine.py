# core/safety/engine.py
"""Safety filtering and content moderation"""
import re
from typing import Dict, Any, List, Optional
from PIL import Image
from ..shared_cache import get_shared_cache
from ..config import get_config


class SafetyEngine:
    """Content safety and filtering engine"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.config = get_config()
        self._blocked_words = self._load_blocked_words()
        self._nsfw_model = None

    def _load_blocked_words(self) -> List[str]:
        """Load blocked words list"""
        # Default blocked words - in real implementation, load from config
        return [
            "violence",
            "harmful",
            "dangerous",
            "illegal",
            "explicit",
            "nsfw",
            "inappropriate",
        ]

    def check_prompt_safety(self, prompt: str) -> dict:
        """Check text prompt safety"""
        try:
            is_safe = True
            warnings = []

            if not self.config.safety.enable_text_filter:
                return {"is_safe": True, "warnings": []}

            # Check prompt length
            if len(prompt) > self.config.safety.max_prompt_length:
                warnings.append("Prompt too long")
                is_safe = False

            # Check blocked words
            prompt_lower = prompt.lower()
            for word in self._blocked_words:
                if word in prompt_lower:
                    warnings.append(f"Contains blocked word: {word}")
                    is_safe = False

            # Additional safety checks
            if re.search(r"\b(hack|exploit|attack|harm)\b", prompt_lower):
                warnings.append("Potentially harmful content detected")
                is_safe = False

            return {
                "is_safe": is_safe,
                "warnings": warnings,
                "filtered_prompt": (
                    self._filter_prompt(prompt) if not is_safe else prompt
                ),
            }

        except Exception as e:
            return {
                "is_safe": False,
                "warnings": [f"Safety check error: {str(e)}"],
                "filtered_prompt": "",
            }

    def check_image_safety(self, image) -> dict:
        """Check image content safety"""
        try:
            if not self.config.safety.enable_nsfw_filter:
                return {"is_safe": True, "processed_image": image}

            # Convert to PIL Image if needed
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif hasattr(image, "convert"):
                pil_image = image
            else:
                return {"is_safe": False, "warnings": ["Invalid image format"]}

            # Mock NSFW detection - in real implementation, use CLIP or specialized model
            is_safe = True
            warnings = []

            # Simple checks based on image properties
            width, height = pil_image.size
            if width * height > 2048 * 2048:
                warnings.append("Image resolution too high")

            return {
                "is_safe": is_safe,
                "warnings": warnings,
                "processed_image": pil_image,
                "confidence": 0.95,
            }

        except Exception as e:
            return {
                "is_safe": False,
                "warnings": [f"Image safety check error: {str(e)}"],
                "processed_image": None,
            }

    def _filter_prompt(self, prompt: str) -> str:
        """Filter and clean prompt content"""
        filtered = prompt
        for word in self._blocked_words:
            filtered = re.sub(
                rf"\b{word}\b", "[FILTERED]", filtered, flags=re.IGNORECASE
            )
        return filtered


class LicenseManager:
    """License and attribution management"""

    def __init__(self, cache_root: str):
        self.cache_root = cache_root
        self.validator = LicenseValidator()

    def register_upload(
        self, file_path: str, license_info: dict, uploader_id: str, safety_result: dict
    ):
        """Register file upload with license information"""
        try:
            # Validate license
            validation = self.validator.validate_license(license_info)

            # Store license information
            license_record = {
                "file_path": file_path,
                "license_info": license_info,
                "uploader_id": uploader_id,
                "safety_result": safety_result,
                "validation": validation,
                "registered_at": "2024-01-01T00:00:00",
            }

            # In real implementation, store in database
            return {"status": "registered", "record_id": str(hash(file_path))}

        except Exception as e:
            raise RuntimeError(f"License registration failed: {str(e)}")


class LicenseValidator:
    """License validation utilities"""

    def validate_license(self, license_info: dict) -> dict:
        """Validate license information"""
        required_fields = ["type", "source", "attribution"]
        missing_fields = [
            field for field in required_fields if field not in license_info
        ]

        valid_types = ["cc0", "cc-by", "cc-by-sa", "mit", "apache", "custom"]
        license_type = license_info.get("type", "").lower()

        return {
            "is_valid": len(missing_fields) == 0 and license_type in valid_types,
            "missing_fields": missing_fields,
            "license_type_valid": license_type in valid_types,
            "warnings": [],
        }


class AttributionManager:
    """Attribution tracking and watermark management"""

    def __init__(self, cache_root: str):
        self.cache_root = cache_root

    def add_watermark(self, image: Image.Image, attribution: str) -> Image.Image:
        """Add attribution watermark to image"""
        try:
            # Mock watermark - in real implementation, add text/logo overlay
            return image  # Return original for now
        except Exception as e:
            raise RuntimeError(f"Watermark addition failed: {str(e)}")


class ComplianceLogger:
    """Compliance and audit logging"""

    def __init__(self, cache_root: str):
        self.cache_root = cache_root
        self.log_dir = f"{cache_root}/logs/compliance"
        # Create log directory
        import os

        os.makedirs(self.log_dir, exist_ok=True)

    def log_upload(self, file_id: str, meta: dict, safety: dict):
        """Log file upload event"""
        try:
            log_entry = {
                "event_type": "upload",
                "file_id": file_id,
                "metadata": meta,
                "safety_result": safety,
                "timestamp": "2024-01-01T00:00:00",
            }
            # In real implementation, write to log file or database
        except Exception as e:
            print(f"Compliance logging error: {e}")

    def log_safety_violation(self, event: str, meta: dict, action: str):
        """Log safety violation event"""
        try:
            log_entry = {
                "event_type": "safety_violation",
                "event": event,
                "metadata": meta,
                "action_taken": action,
                "timestamp": "2024-01-01T00:00:00",
            }
            # In real implementation, write to log file or database
        except Exception as e:
            print(f"Safety violation logging error: {e}")
