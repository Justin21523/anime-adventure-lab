# core/safety/detector.py
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NSFWDetector:
    """NSFW content detection using CLIP-based classification"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # NSFW detection prompts (multilingual)
        self.nsfw_prompts = [
            "explicit sexual content",
            "nudity and sexual activity",
            "adult content not suitable for work",
            "露骨的性內容",
            "裸體或性行為",
            "成人內容",
        ]

        self.safe_prompts = [
            "safe for work content",
            "family friendly image",
            "appropriate workplace content",
            "安全工作內容",
            "家庭友善圖像",
            "適宜工作場所內容",
        ]

    def detect_nsfw(
        self, image: Image.Image, threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Detect NSFW content in image
        Returns: {is_nsfw: bool, confidence: float, details: dict}
        """
        try:
            # Prepare inputs
            all_prompts = self.nsfw_prompts + self.safe_prompts
            inputs = self.processor(
                text=all_prompts, images=image, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0)

            # Calculate NSFW probability (first half of prompts)
            nsfw_prob = probs[: len(self.nsfw_prompts)].max().item()
            safe_prob = probs[len(self.nsfw_prompts) :].max().item()

            is_nsfw = nsfw_prob > threshold and nsfw_prob > safe_prob

            return {
                "is_nsfw": is_nsfw,
                "confidence": float(nsfw_prob),
                "safe_confidence": float(safe_prob),
                "threshold": threshold,
                "details": {
                    "nsfw_scores": probs[: len(self.nsfw_prompts)].cpu().tolist(),
                    "safe_scores": probs[len(self.nsfw_prompts) :].cpu().tolist(),
                },
            }

        except Exception as e:
            logger.error(f"NSFW detection failed: {e}")
            return {"is_nsfw": False, "confidence": 0.0, "error": str(e)}  # Fail safe


class FaceBlurrer:
    """Face detection and blurring for privacy protection"""

    def __init__(self):
        # Use OpenCV's pre-trained face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def blur_faces(
        self, image: Image.Image, blur_strength: int = 15
    ) -> Tuple[Image.Image, int]:
        """
        Blur detected faces in image
        Returns: (blurred_image, num_faces_detected)
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Blur each detected face
            for x, y, w, h in faces:
                # Extract face region
                face_region = cv_image[y : y + h, x : x + w]
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(
                    face_region, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0
                )
                # Replace face region with blurred version
                cv_image[y : y + h, x : x + w] = blurred_face

            # Convert back to PIL
            blurred_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            return blurred_image, len(faces)

        except Exception as e:
            logger.error(f"Face blurring failed: {e}")
            return image, 0


class PromptCleaner:
    """Clean and validate user prompts for safety"""

    def __init__(self):
        # Harmful patterns (English + Chinese)
        self.harmful_patterns = [
            # Direct injection attempts
            r"ignore\s+previous\s+instructions",
            r"forget\s+your\s+role",
            r"you\s+are\s+now",
            r"act\s+as\s+if",
            # Chinese equivalents
            r"忽略之前的指令",
            r"忘記你的角色",
            r"你現在是",
            r"假裝你是",
            # Explicit content keywords
            r"\b(nsfw|nude|sex|porn|explicit)\b",
            r"\b(色情|裸體|性愛|成人)\b",
            # Harmful content
            r"\b(violence|kill|murder|harm)\b",
            r"\b(暴力|殺死|謀殺|傷害)\b",
        ]

    def clean_prompt(self, prompt: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Clean and validate user prompt
        Returns: {clean_prompt: str, is_safe: bool, warnings: list}
        """
        import re

        original_prompt = prompt
        warnings = []

        # Length check
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
            warnings.append(f"Prompt truncated to {max_length} characters")

        # Remove excessive whitespace
        prompt = re.sub(r"\s+", " ", prompt.strip())

        # Check for harmful patterns
        is_safe = True
        for pattern in self.harmful_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                is_safe = False
                warnings.append(f"Potentially harmful content detected")
                break

        # Basic sanitization (remove special chars that could break parsing)
        prompt = re.sub(r"[<>{}[\]\\]", "", prompt)

        return {
            "clean_prompt": prompt,
            "is_safe": is_safe,
            "warnings": warnings,
            "original_length": len(original_prompt),
            "cleaned_length": len(prompt),
        }


class SafetyEngine:
    """Main safety engine coordinating all safety checks"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.nsfw_detector = NSFWDetector()
        self.face_blurrer = FaceBlurrer()
        self.prompt_cleaner = PromptCleaner()

        # Safety thresholds
        self.nsfw_threshold = self.config.get("nsfw_threshold", 0.85)
        self.auto_blur_faces = self.config.get("auto_blur_faces", False)
        self.strict_prompt_filter = self.config.get("strict_prompt_filter", True)

    def check_image_safety(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive image safety check"""
        result = {
            "is_safe": True,
            "nsfw_check": {},
            "face_check": {},
            "actions_taken": [],
        }

        # NSFW detection
        nsfw_result = self.nsfw_detector.detect_nsfw(image, self.nsfw_threshold)
        result["nsfw_check"] = nsfw_result

        if nsfw_result.get("is_nsfw", False):
            result["is_safe"] = False
            result["actions_taken"].append("flagged_as_nsfw")

        # Face detection and optional blurring
        if self.auto_blur_faces:
            blurred_image, face_count = self.face_blurrer.blur_faces(image)
            result["face_check"] = {
                "faces_detected": face_count,
                "blurred": face_count > 0,
            }
            if face_count > 0:
                result["processed_image"] = blurred_image
                result["actions_taken"].append("faces_blurred")

        return result

    def check_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        """Check prompt safety and clean if needed"""
        return self.prompt_cleaner.clean_prompt(prompt)

    def should_block_content(self, safety_result: Dict[str, Any]) -> bool:
        """Determine if content should be blocked based on safety checks"""
        if not safety_result.get("is_safe", True):
            return True

        nsfw_check = safety_result.get("nsfw_check", {})
        if nsfw_check.get("is_nsfw", False):
            return True

        return False
