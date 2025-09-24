# core/safety/content_filter.py
"""
Content Safety Filter
Implements NSFW detection, sensitive content filtering and safety validation
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import pipeline, AutoProcessor, AutoModel

from ..config import get_config
from ..exceptions import SafetyError, ValidationError
from ..utils.image import ImageProcessor

logger = logging.getLogger(__name__)


class ContentFilter:
    """Multi-layered content safety filter"""

    def __init__(self):
        self.config = get_config()
        self.image_processor = ImageProcessor()

        # 初始化屬性
        self.blocked_terms: Set[str] = set()
        self.sensitive_patterns: List[Any] = []
        self.nsfw_classifier: Optional[Any] = None
        self.text_classifier: Optional[Any] = None

        # Load safety models
        self._load_safety_models()

        # Define blocked terms and patterns
        self.blocked_terms = self._load_blocked_terms()
        self.sensitive_patterns = self._compile_patterns()

    def _detect_faces(self, image: Image.Image) -> int:
        """Basic face detection using OpenCV"""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Load face cascade - 修正 cv2.data 問題
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            if not Path(cascade_path).exists():
                # 備用方案：使用完整路徑或跳過
                logger.warning("Face cascade not found, skipping face detection")
                return 0

            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            return len(faces)

        except Exception as e:
            logger.warning(f"⚠️ Face detection failed: {e}")
            return 0

    def _load_safety_models(self):
        """Load NSFW detection and safety models"""
        try:
            # NSFW image classifier
            self.nsfw_classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=0 if torch.cuda.is_available() else -1,
            )

            # Text safety classifier (optional)
            if self.config.safety.enable_text_filter:
                self.text_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=0 if torch.cuda.is_available() else -1,
                )
            else:
                self.text_classifier = None

            logger.info("✅ Safety models loaded successfully")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load safety models: {e}")
            self.nsfw_classifier = None
            self.text_classifier = None

    def _load_blocked_terms(self) -> Set[str]:
        """Load blocked terms from config or file"""
        blocked_terms = set()

        # Default blocked terms (basic)
        default_blocked = {
            "nude",
            "naked",
            "nsfw",
            "sexual",
            "porn",
            "xxx",
            "child",
            "minor",
            "underage",
            "kid",
            "baby",
            "violence",
            "weapon",
            "blood",
            "gore",
            "death",
        }
        blocked_terms.update(default_blocked)

        # Load from config file if exists
        blocked_file = Path(self.config.safety.blocked_terms_file)  # type: ignore
        if blocked_file.exists():
            try:
                with open(blocked_file, "r", encoding="utf-8") as f:
                    for line in f:
                        term = line.strip().lower()
                        if term and not term.startswith("#"):
                            blocked_terms.add(term)
                logger.info(f"📝 Loaded {len(blocked_terms)} blocked terms")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load blocked terms file: {e}")

        return blocked_terms

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for sensitive content detection"""
        patterns = [
            # Email-like patterns
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            # Phone number patterns
            re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            # URL patterns
            re.compile(r"https?://[^\s]+"),
            # Credit card patterns (basic)
            re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        ]
        return patterns

    def check_text_safety(self, text: str) -> Dict[str, Any]:
        """Check text content for safety violations"""
        try:
            result = {
                "is_safe": True,
                "violations": [],
                "confidence": 1.0,
                "filtered_text": text,
            }

            # Check blocked terms
            text_lower = text.lower()
            found_terms = []
            for term in self.blocked_terms:
                if term in text_lower:
                    found_terms.append(term)

            if found_terms:
                result["is_safe"] = False
                result["violations"].append(
                    {"type": "blocked_terms", "terms": found_terms}
                )

            # Check sensitive patterns
            sensitive_matches = []
            for pattern in self.sensitive_patterns:
                matches = pattern.findall(text)
                if matches:
                    sensitive_matches.extend(matches)

            if sensitive_matches:
                result["violations"].append(
                    {"type": "sensitive_data", "matches": sensitive_matches}
                )
                # Optionally mask sensitive data
                filtered_text = text
                for pattern in self.sensitive_patterns:
                    filtered_text = pattern.sub("***", filtered_text)
                result["filtered_text"] = filtered_text

            # Use ML classifier if available
            if self.text_classifier and len(text.strip()) > 10:
                try:
                    ml_result = self.text_classifier(text[:512])  # Limit length
                    toxicity_score = max(
                        [
                            r["score"]
                            for r in ml_result
                            if r["label"] in ["TOXIC", "SEVERE_TOXIC"]
                        ],
                        default=0.0,
                    )

                    if toxicity_score > self.config.safety.toxicity_threshold:
                        result["is_safe"] = False
                        result["violations"].append(
                            {"type": "toxicity", "score": toxicity_score}
                        )
                        result["confidence"] = min(
                            result["confidence"], 1 - toxicity_score
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Text classification failed: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Text safety check failed: {e}")
            raise SafetyError(f"Text safety check failed: {e}")

    def check_image_safety(self, image: Image.Image) -> Dict[str, Any]:
        """Check image content for NSFW and safety violations"""
        try:
            result = {
                "is_safe": True,
                "violations": [],
                "confidence": 1.0,
                "nsfw_score": 0.0,
            }

            if not self.nsfw_classifier:
                logger.warning(
                    "⚠️ NSFW classifier not available, skipping image safety check"
                )
                return result

            # Resize image for faster processing
            image_resized = self.image_processor.resize_image(image, max_size=512)

            # NSFW detection
            nsfw_results = self.nsfw_classifier(image_resized)

            # Process results
            nsfw_score = 0.0
            for pred in nsfw_results:
                if pred["label"] in ["nsfw", "NSFW", "porn", "explicit"]:
                    nsfw_score = max(nsfw_score, pred["score"])

            result["nsfw_score"] = nsfw_score

            if nsfw_score > self.config.safety.nsfw_threshold:
                result["is_safe"] = False
                result["violations"].append(
                    {"type": "nsfw_content", "score": nsfw_score}
                )
                result["confidence"] = 1 - nsfw_score

            # Additional checks: face detection for privacy
            if self.config.safety.enable_face_detection:
                face_count = self._detect_faces(image_resized)
                if face_count > 0:
                    result["violations"].append(
                        {"type": "face_detected", "count": face_count}
                    )

            return result

        except Exception as e:
            logger.error(f"❌ Image safety check failed: {e}")
            raise SafetyError(f"Image safety check failed: {e}")

    def apply_safety_filter(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply comprehensive safety filtering to content - 修正缺失方法"""
        try:
            filtered_content = content.copy()
            safety_report = {
                "overall_safe": True,
                "checks_performed": [],
                "violations": [],
            }

            # Check text prompts - 安全檢查 None 值
            if "prompt" in content and content["prompt"]:
                try:
                    text_check = self.check_text_safety(content["prompt"])
                    safety_report["checks_performed"].append("text_prompt")

                    if not text_check["is_safe"]:
                        safety_report["overall_safe"] = False
                        safety_report["violations"].extend(text_check["violations"])

                        # Apply filtering
                        auto_filter = getattr(
                            self.config.safety, "auto_filter_text", False
                        )
                        if auto_filter:
                            filtered_content["prompt"] = text_check["filtered_text"]
                            filtered_content["_original_prompt"] = content["prompt"]
                except Exception as e:
                    logger.warning(f"Text safety check failed: {e}")

            # Check negative prompts - 安全檢查 None 值
            if "negative_prompt" in content and content["negative_prompt"]:
                try:
                    neg_check = self.check_text_safety(content["negative_prompt"])
                    safety_report["checks_performed"].append("negative_prompt")

                    allow_negative = getattr(
                        self.config.safety, "allow_negative_filtering", True
                    )
                    if not neg_check["is_safe"] and not allow_negative:
                        safety_report["overall_safe"] = False
                        safety_report["violations"].extend(neg_check["violations"])
                except Exception as e:
                    logger.warning(f"Negative prompt safety check failed: {e}")

            # Check input images - 修正 None 檢查
            if "image" in content and content["image"] is not None:
                try:
                    # 確保圖片路徑有效
                    image_input = content["image"]
                    if (
                        isinstance(image_input, (str, Path))
                        and str(image_input).strip()
                    ):
                        image = self.image_processor.load_image(image_input)
                        image_check = self.check_image_safety(image)
                        safety_report["checks_performed"].append("input_image")

                        if not image_check["is_safe"]:
                            safety_report["overall_safe"] = False
                            safety_report["violations"].extend(
                                image_check["violations"]
                            )
                    elif hasattr(image_input, "size"):  # PIL Image 物件
                        image_check = self.check_image_safety(image_input)  # type: ignore
                        safety_report["checks_performed"].append("input_image")

                        if not image_check["is_safe"]:
                            safety_report["overall_safe"] = False
                            safety_report["violations"].extend(
                                image_check["violations"]
                            )

                except Exception as e:
                    logger.warning(f"Image safety check failed: {e}")

            filtered_content["_safety_report"] = safety_report
            return filtered_content

        except Exception as e:
            logger.error(f"❌ Safety filtering failed: {e}")
            # 返回安全的預設值而不是拋出異常
            return {
                **content,
                "_safety_report": {
                    "overall_safe": False,
                    "checks_performed": [],
                    "violations": [{"type": "system_error", "message": str(e)}],
                },
            }

    def validate_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generation parameters for safety"""
        try:
            validated_params = params.copy()

            # Validate image dimensions
            max_resolution = self.config.safety.max_resolution
            if "width" in params and params["width"] > max_resolution:
                validated_params["width"] = max_resolution
                logger.warning(f"⚠️ Width clamped to {max_resolution}")

            if "height" in params and params["height"] > max_resolution:
                validated_params["height"] = max_resolution
                logger.warning(f"⚠️ Height clamped to {max_resolution}")

            # Validate steps
            max_steps = self.config.safety.max_inference_steps
            if (
                "num_inference_steps" in params
                and params["num_inference_steps"] > max_steps
            ):
                validated_params["num_inference_steps"] = max_steps
                logger.warning(f"⚠️ Steps clamped to {max_steps}")

            # Validate guidance scale
            max_guidance = self.config.safety.max_guidance_scale
            if "guidance_scale" in params and params["guidance_scale"] > max_guidance:
                validated_params["guidance_scale"] = max_guidance
                logger.warning(f"⚠️ Guidance scale clamped to {max_guidance}")

            return validated_params

        except Exception as e:
            logger.error(f"❌ Parameter validation failed: {e}")
            raise ValidationError(
                "parameter_error", f"Parameter validation failed: {e}"
            )


# Global instance
_content_filter = None


def get_content_filter() -> ContentFilter:
    """Get global content filter instance"""
    global _content_filter
    if _content_filter is None:
        _content_filter = ContentFilter()
    return _content_filter
