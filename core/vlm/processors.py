# core/vlm/processors.py
"""
VLM Input/Output Processors and Utilities
"""

import torch
import logging
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageStat
from typing import Union, Dict, Any, List, Optional, Tuple
import base64
import io
import re
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class VLMImageProcessor:
    """Advanced image processor for VLM models with quality assessment and optimization"""

    def __init__(self, config=None):
        self.config = config or {}
        self.max_size = self.config.get("max_image_size", 1024)
        self.min_size = self.config.get("min_image_size", 224)
        self.quality_threshold = self.config.get("quality_threshold", 0.7)

        # Initialize face detection cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            )
        except Exception as e:
            logger.warning(f"Failed to load face cascade: {e}")
            self.face_cascade = None

    def preprocess_for_caption(self, image: Image.Image) -> Image.Image:
        """Preprocess image specifically for captioning models"""
        # Resize to optimal size for captioning (smaller for efficiency)
        image = self._smart_resize(image, target_size=384)

        # Enhance contrast and sharpness for better feature extraction
        image = self._enhance_image(image)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def preprocess_for_vqa(self, image: Image.Image) -> Image.Image:
        """Preprocess image specifically for VQA models"""
        # VQA models often need larger images for detail recognition
        image = self._smart_resize(image, target_size=576)

        # Apply subtle sharpening to enhance text/detail readability
        image = image.filter(
            ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)
        )

        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _smart_resize(self, image: Image.Image, target_size: int) -> Image.Image:
        """Intelligently resize image maintaining aspect ratio"""
        width, height = image.size

        # Don't upscale small images too much
        if max(width, height) < self.min_size:
            scale_factor = self.min_size / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        # Downscale large images
        elif max(width, height) > self.max_size:
            scale_factor = self.max_size / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        # Resize to target if within reasonable bounds
        else:
            if width > height:
                new_width = min(target_size, self.max_size)
                new_height = int(height * (new_width / width))
            else:
                new_height = min(target_size, self.max_size)
                new_width = int(width * (new_height / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _enhance_image(
        self, image: Image.Image, enhance_factor: float = 1.1
    ) -> Image.Image:
        """Apply subtle enhancements to improve model performance"""
        try:
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhance_factor)

            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)

            return image
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def detect_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Detect various image quality metrics using advanced analysis"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)

            # Basic quality metrics
            brightness = np.mean(img_array)
            contrast = np.std(img_array)

            # Blur detection using Laplacian variance
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Color distribution analysis
            color_channels = [img_array[:, :, i] for i in range(3)]
            color_balance = [np.std(channel) for channel in color_channels]

            # Edge density (detail richness)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Noise estimation using HF component analysis
            noise_level = self._estimate_noise_level(gray)

            # Dynamic range
            dynamic_range = np.max(img_array) - np.min(img_array)

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                brightness,  # type: ignore
                contrast,  # type: ignore
                blur_score,
                color_balance,  # type: ignore
                edge_density,  # type: ignore
                noise_level,
                dynamic_range,
            )

            return {
                "quality_score": quality_score,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "blur_score": float(blur_score),
                "edge_density": float(edge_density),
                "noise_level": float(noise_level),
                "dynamic_range": float(dynamic_range),
                "color_balance": [float(cb) for cb in color_balance],
                "is_high_quality": quality_score > self.quality_threshold,
                "recommendations": self._get_quality_recommendations(
                    brightness, contrast, blur_score, noise_level  # type: ignore
                ),
            }

        except Exception as e:
            logger.error(f"Quality detection failed: {e}")
            return {
                "quality_score": 0.5,
                "error": str(e),
                "is_high_quality": True,  # Assume OK if analysis fails
            }

    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level using high-frequency analysis"""
        try:
            # Use Sobel operator to get high frequency components
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Noise is estimated from the standard deviation of high-frequency components
            noise = np.std(sobel_magnitude)

            # Normalize to 0-1 range (empirical scaling)
            return min(noise / 100.0, 1.0)  # type: ignore

        except Exception:
            return 0.5

    def _calculate_quality_score(
        self,
        brightness: float,
        contrast: float,
        blur_score: float,
        color_balance: List[float],
        edge_density: float,
        noise_level: float,
        dynamic_range: float,
    ) -> float:
        """Calculate comprehensive quality score"""
        score = 0.0

        # Brightness score (prefer 50-200 range for RGB)
        if 50 <= brightness <= 200:
            score += 0.15
        elif brightness < 30 or brightness > 230:
            score -= 0.15
        elif 30 <= brightness <= 50 or 200 <= brightness <= 230:
            score += 0.05

        # Contrast score (higher is generally better)
        if contrast > 30:
            score += 0.15
        elif contrast < 15:
            score -= 0.15
        elif 15 <= contrast <= 30:
            score += 0.05

        # Blur score (higher means less blurry)
        if blur_score > 100:
            score += 0.2
        elif blur_score < 50:
            score -= 0.2
        elif 50 <= blur_score <= 100:
            score += 0.1

        # Color balance (prefer balanced channels)
        if len(color_balance) == 3:
            balance_diff = max(color_balance) - min(color_balance)
            if balance_diff < 20:
                score += 0.1
            elif balance_diff > 50:
                score -= 0.05

        # Edge density (detail richness)
        if 0.05 <= edge_density <= 0.3:  # Good amount of details
            score += 0.15
        elif edge_density < 0.02:  # Too smooth/blurry
            score -= 0.1
        elif edge_density > 0.5:  # Too noisy
            score -= 0.05

        # Noise level (lower is better)
        if noise_level < 0.3:
            score += 0.1
        elif noise_level > 0.7:
            score -= 0.15

        # Dynamic range (higher is better)
        if dynamic_range > 200:
            score += 0.1
        elif dynamic_range < 100:
            score -= 0.1

        # Base score adjustment
        score += 0.5

        return max(0.0, min(1.0, score))

    def _get_quality_recommendations(
        self, brightness: float, contrast: float, blur_score: float, noise_level: float
    ) -> List[str]:
        """Get recommendations for image quality improvement"""
        recommendations = []

        if brightness < 50:
            recommendations.append("圖片過暗，建議增加亮度")
        elif brightness > 200:
            recommendations.append("圖片過亮，建議降低亮度")

        if contrast < 20:
            recommendations.append("對比度不足，建議增加對比度")

        if blur_score < 50:
            recommendations.append("圖片模糊，建議使用更清晰的圖片或進行銳化")

        if noise_level > 0.6:
            recommendations.append("圖片雜訊過多，建議進行降噪處理")

        if not recommendations:
            recommendations.append("圖片品質良好")

        return recommendations

    def auto_correct_image(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """Automatically apply corrections based on quality analysis"""
        quality_info = self.detect_image_quality(image)
        corrections_applied = []

        try:
            brightness = quality_info.get("brightness", 128)
            contrast = quality_info.get("contrast", 30)
            blur_score = quality_info.get("blur_score", 100)
            noise_level = quality_info.get("noise_level", 0.5)

            # Auto brightness correction
            if brightness < 80:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.2)
                corrections_applied.append("增加亮度")
            elif brightness > 180:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.9)
                corrections_applied.append("降低亮度")

            # Auto contrast correction
            if contrast < 25:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.3)
                corrections_applied.append("增加對比度")

            # Auto sharpening for blurry images
            if blur_score < 60:
                image = image.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                )
                corrections_applied.append("銳化處理")

            # Noise reduction for noisy images
            if noise_level > 0.6:
                # Convert to opencv for bilateral filtering
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
                image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                corrections_applied.append("降噪處理")

        except Exception as e:
            logger.error(f"Auto correction failed: {e}")
            corrections_applied.append(f"自動校正失敗: {str(e)}")

        return image, corrections_applied

    def detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect faces in image with confidence scores"""
        if self.face_cascade is None:
            return []

        try:
            # Convert to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Detect faces with different parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            face_info = []
            for x, y, w, h in faces:
                # Calculate confidence based on size and position
                face_size = w * h
                image_size = image.width * image.height
                size_ratio = face_size / image_size

                # Estimate confidence (simple heuristic)
                confidence = min(0.9, 0.5 + size_ratio * 2)

                face_info.append(
                    {
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "confidence": float(confidence),
                        "size_ratio": float(size_ratio),
                        "center": [int(x + w / 2), int(y + h / 2)],
                    }
                )

            return face_info

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def apply_face_blur(
        self,
        image: Image.Image,
        blur_radius: int = 15,
        faces: List[Dict[str, Any]] = None,  # type: ignore
    ) -> Tuple[Image.Image, int]:
        """Apply blur to detected faces for privacy protection"""
        if faces is None:
            faces = self.detect_faces(image)

        if not faces:
            return image, 0

        try:
            result_image = image.copy()

            for face in faces:
                bbox = face["bbox"]
                x1, y1, x2, y2 = bbox

                # Extract face region
                face_region = result_image.crop((x1, y1, x2, y2))

                # Apply Gaussian blur
                blurred_face = face_region.filter(
                    ImageFilter.GaussianBlur(radius=blur_radius)
                )

                # Paste back
                result_image.paste(blurred_face, (x1, y1))

            return result_image, len(faces)

        except Exception as e:
            logger.error(f"Face blurring failed: {e}")
            return image, 0

    def analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition using rule of thirds and other metrics"""
        try:
            width, height = image.size
            img_array = np.array(image.convert("L"))  # Convert to grayscale

            # Rule of thirds analysis
            thirds_x = [width // 3, 2 * width // 3]
            thirds_y = [height // 3, 2 * height // 3]

            # Calculate interest points near rule of thirds intersections
            interest_score = 0
            for x in thirds_x:
                for y in thirds_y:
                    # Sample area around intersection
                    x1, y1 = max(0, x - 20), max(0, y - 20)
                    x2, y2 = min(width, x + 20), min(height, y + 20)

                    if x2 > x1 and y2 > y1:
                        region = img_array[y1:y2, x1:x2]
                        variance = np.var(region)
                        interest_score += variance

            # Normalize interest score
            interest_score = min(1.0, interest_score / 10000.0)

            # Center bias (check if main subject is centered)
            center_region = img_array[
                height // 4 : 3 * height // 4, width // 4 : 3 * width // 4
            ]
            center_variance = np.var(center_region)
            total_variance = np.var(img_array)
            center_bias = center_variance / (total_variance + 1e-6)

            # Edge distribution
            edges = cv2.Canny(img_array, 50, 150)
            edge_distribution = self._analyze_edge_distribution(edges, width, height)

            return {
                "rule_of_thirds_score": float(interest_score),
                "center_bias": float(center_bias),
                "edge_distribution": edge_distribution,
                "composition_score": float(
                    (
                        interest_score * 0.4
                        + (1 - center_bias) * 0.3
                        + edge_distribution["balance"] * 0.3
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Composition analysis failed: {e}")
            return {"composition_score": 0.5, "error": str(e)}

    def _analyze_edge_distribution(
        self, edges: np.ndarray, width: int, height: int
    ) -> Dict[str, float]:
        """Analyze distribution of edges across image regions"""
        try:
            # Divide image into 9 regions (3x3 grid)
            regions = []
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * height // 3, (i + 1) * height // 3
                    x1, x2 = j * width // 3, (j + 1) * width // 3
                    region_edges = edges[y1:y2, x1:x2]
                    edge_density = np.sum(region_edges > 0) / region_edges.size
                    regions.append(edge_density)

            # Calculate balance (lower standard deviation = more balanced)
            balance = 1.0 - min(1.0, np.std(regions) / 0.1)

            return {
                "regions": regions,  # type: ignore
                "balance": float(balance),
                "max_density": float(max(regions)),
                "min_density": float(min(regions)),
            }

        except Exception:
            return {"balance": 0.5, "regions": [0.1] * 9}  # type: ignore


class VLMTextProcessor:
    """Advanced text processor for VLM queries and responses with multilingual support"""

    def __init__(self, config=None):
        self.config = config or {}
        self.max_question_length = self.config.get("max_question_length", 200)
        self.enable_chinese_support = self.config.get("enable_chinese_support", True)

        # Language patterns
        self.chinese_patterns = {
            "traditional": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),
            "questions": [
                "什麼",
                "誰",
                "哪裡",
                "何時",
                "為什麼",
                "如何",
                "怎麼",
                "多少",
            ],
            "particles": ["嗎", "呢", "吧", "啊", "的", "了", "過"],
        }

        self.english_patterns = {
            "questions": [
                "what",
                "who",
                "where",
                "when",
                "why",
                "how",
                "which",
                "whose",
            ],
            "auxiliary": [
                "do",
                "does",
                "did",
                "can",
                "could",
                "will",
                "would",
                "should",
            ],
        }

    def preprocess_question(self, question: str) -> Dict[str, Any]:
        """Advanced question preprocessing with language detection and optimization"""
        original_question = question

        # Basic cleaning
        question = self._clean_text(question)

        # Length check
        if len(question) > self.max_question_length:
            question = self._smart_truncate(question, self.max_question_length)

        # Language detection and analysis
        language_info = self._analyze_language(question)

        # Question type detection
        question_type = self._detect_question_type(question, language_info)

        # Question enhancement based on language and type
        enhanced_question = self._enhance_question(
            question, language_info, question_type
        )

        # Quality assessment
        quality_score = self._assess_question_quality(
            enhanced_question, language_info, question_type
        )

        return {
            "processed_question": enhanced_question,
            "original_question": original_question,
            "language_info": language_info,
            "question_type": question_type,
            "quality_score": quality_score,
            "is_valid": len(enhanced_question.strip()) > 0 and quality_score > 0.3,
            "estimated_complexity": self._estimate_question_complexity(
                enhanced_question, question_type
            ),
            "suggested_improvements": self._suggest_improvements(
                question, quality_score
            ),
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf.,!?;:()"-]', "", text)

        # Normalize quotes
        text = (
            text.replace('"', '"').replace('"', '"').replace(""", "'").replace(""", "'")
        )

        return text.strip()

    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Smart truncation that preserves sentence structure"""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundaries
        sentences = re.split(r"[.!?。！？]", text)
        truncated = ""

        for sentence in sentences:
            if len(truncated + sentence) <= max_length - 3:
                truncated += sentence + "."
            else:
                break

        if not truncated:
            # Fallback to word boundary truncation
            words = text.split()
            truncated = ""
            for word in words:
                if len(truncated + word) <= max_length - 3:
                    truncated += word + " "
                else:
                    break

        return (truncated.strip() + "...").strip()

    def _analyze_language(self, text: str) -> Dict[str, Any]:
        """Comprehensive language analysis"""
        text_lower = text.lower()

        # Character type analysis
        chinese_chars = len(self.chinese_patterns["traditional"].findall(text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(re.sub(r"\s", "", text))

        if total_chars == 0:
            return {
                "primary_language": "unknown",
                "is_chinese": False,
                "is_english": False,
                "is_mixed": False,
                "chinese_ratio": 0.0,
                "english_ratio": 0.0,
                "confidence": 0.0,
            }

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        # Language detection logic
        if chinese_ratio > 0.5:
            primary_language = "chinese"
            confidence = chinese_ratio
        elif english_ratio > 0.5:
            primary_language = "english"
            confidence = english_ratio
        elif chinese_ratio > 0.2 and english_ratio > 0.2:
            primary_language = "mixed"
            confidence = min(chinese_ratio, english_ratio) * 2
        else:
            primary_language = "other"
            confidence = 0.3

        # Detect Chinese question patterns
        chinese_question_markers = sum(
            1 for marker in self.chinese_patterns["questions"] if marker in text
        )
        english_question_markers = sum(
            1 for marker in self.english_patterns["questions"] if marker in text_lower
        )

        return {
            "primary_language": primary_language,
            "is_chinese": chinese_ratio > 0.3,
            "is_english": english_ratio > 0.3,
            "is_mixed": chinese_ratio > 0.1 and english_ratio > 0.1,
            "chinese_ratio": chinese_ratio,
            "english_ratio": english_ratio,
            "confidence": confidence,
            "chinese_question_markers": chinese_question_markers,
            "english_question_markers": english_question_markers,
            "detected_script": "traditional_chinese" if chinese_chars > 0 else "latin",
        }

    def _detect_question_type(
        self, question: str, language_info: Dict[str, Any]
    ) -> str:
        """Advanced question type detection with language awareness"""
        question_lower = question.lower()

        # Define multilingual patterns
        patterns = {
            "counting": {
                "chinese": ["多少", "幾個", "幾", "數量", "個數"],
                "english": ["how many", "count", "number of", "quantity"],
            },
            "identification": {
                "chinese": ["是什麼", "什麼是", "這是", "那是", "什麼東西"],
                "english": ["what is", "what are", "identify", "what does", "what do"],
            },
            "location": {
                "chinese": ["哪裡", "在哪", "何處", "位置", "地點"],
                "english": ["where", "location", "place", "position"],
            },
            "description": {
                "chinese": ["描述", "說明", "解釋", "介紹", "講述"],
                "english": ["describe", "tell me about", "explain", "detail"],
            },
            "color": {
                "chinese": ["顏色", "什麼色", "色彩", "什麼顏色"],
                "english": ["color", "colour", "what color", "hue"],
            },
            "action": {
                "chinese": ["做什麼", "在做", "動作", "行為", "活動"],
                "english": ["doing", "action", "activity", "what doing", "performing"],
            },
            "yes_no": {
                "chinese": ["是否", "有沒有", "是不是", "對不對", "嗎"],
                "english": ["is there", "do you", "can you see", "yes or no"],
            },
            "comparison": {
                "chinese": ["比較", "相同", "不同", "差別", "區別"],
                "english": ["compare", "difference", "similar", "same", "different"],
            },
            "temporal": {
                "chinese": ["什麼時候", "何時", "時間", "日期"],
                "english": ["when", "what time", "date", "temporal"],
            },
            "reason": {
                "chinese": ["為什麼", "原因", "因為", "怎麼會"],
                "english": ["why", "reason", "because", "cause"],
            },
        }

        # Determine language context
        lang_context = "chinese" if language_info["is_chinese"] else "english"

        # Score each question type
        type_scores = {}
        for q_type, lang_patterns in patterns.items():
            score = 0
            keywords = lang_patterns.get(lang_context, []) + lang_patterns.get(
                "english", []
            )

            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
                    # Bonus for exact matches
                    if keyword == question_lower.strip():
                        score += 2

            type_scores[q_type] = score

        # Return the highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]

        return "general"

    def _enhance_question(
        self, question: str, language_info: Dict[str, Any], question_type: str
    ) -> str:
        """Enhance question for better VLM understanding"""
        enhanced = question

        # Add proper punctuation
        if not enhanced.endswith(("?", "？", ".", "。")):
            if question_type in [
                "counting",
                "identification",
                "location",
                "yes_no",
                "reason",
            ]:
                enhanced += "？" if language_info["is_chinese"] else "?"
            else:
                enhanced += "。" if language_info["is_chinese"] else "."

        # Add context hints for ambiguous questions
        if len(enhanced) < 10:  # Very short questions
            if language_info["is_chinese"]:
                if question_type == "description":
                    enhanced = f"請詳細{enhanced}"
                elif question_type == "identification":
                    enhanced = f"圖片中{enhanced}"
            else:
                if question_type == "description":
                    enhanced = f"Please describe {enhanced}"
                elif question_type == "identification":
                    enhanced = f"What is {enhanced}"

        return enhanced

    def _assess_question_quality(
        self, question: str, language_info: Dict[str, Any], question_type: str
    ) -> float:
        """Assess the quality of the processed question"""
        quality = 0.5  # Base quality

        # Length assessment
        question_len = len(question.strip())
        if 10 <= question_len <= 100:
            quality += 0.2
        elif question_len < 5:
            quality -= 0.3
        elif question_len > 150:
            quality -= 0.1

        # Language clarity
        if language_info["confidence"] > 0.7:
            quality += 0.1
        elif language_info["confidence"] < 0.3:
            quality -= 0.2

        # Question type specificity
        if question_type != "general":
            quality += 0.15

        # Punctuation and grammar
        if question.endswith(("?", "？", ".", "。")):
            quality += 0.05

        # Avoid too generic questions
        generic_patterns = ["這是什麼", "what is this", "describe", "描述"]
        if any(pattern in question.lower() for pattern in generic_patterns):
            if question_len < 15:
                quality -= 0.1

        return max(0.0, min(1.0, quality))

    def _estimate_question_complexity(self, question: str, question_type: str) -> str:
        """Estimate question complexity for processing optimization"""
        word_count = len(question.split())

        # Complex question indicators
        complex_indicators = [
            "比較",
            "分析",
            "解釋",
            "為什麼",
            "如何",
            "詳細",
            "複雜",
            "compare",
            "analyze",
            "explain",
            "why",
            "how",
            "detailed",
            "complex",
            "difference",
            "relationship",
            "interaction",
        ]

        has_complex_words = any(
            indicator in question.lower() for indicator in complex_indicators
        )

        # Question type complexity
        complex_types = ["comparison", "reason", "temporal", "description"]
        is_complex_type = question_type in complex_types

        if word_count > 15 or has_complex_words or is_complex_type:
            return "high"
        elif word_count > 8 or question_type in ["identification", "action"]:
            return "medium"
        else:
            return "low"

    def _suggest_improvements(self, question: str, quality_score: float) -> List[str]:
        """Suggest improvements for low-quality questions"""
        suggestions = []

        if quality_score < 0.4:
            if len(question) < 5:
                suggestions.append("問題太短，請提供更多詳細信息")
            if not question.strip().endswith(("?", "？", ".", "。")):
                suggestions.append("建議添加適當的標點符號")
            if "這" in question or "that" in question.lower():
                suggestions.append("可以更具體地描述你想了解的內容")

        if not suggestions:
            suggestions.append("問題品質良好")

        return suggestions

    def postprocess_response(
        self, response: str, question_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Advanced VLM response postprocessing with quality assessment"""
        original_response = response

        # Basic cleaning
        response = self._clean_text(response)

        # Language-specific postprocessing
        language_info = question_info.get("language_info", {})
        if language_info.get("is_chinese", False):
            response = self._postprocess_chinese_response(response)
        else:
            response = self._postprocess_english_response(response)

        # Response quality assessment
        quality_metrics = self._assess_response_quality(response, question_info)

        # Content analysis
        content_analysis = self._analyze_response_content(response, question_info)

        # Generate response summary if too long
        summary = self._generate_summary(response) if len(response) > 150 else None

        # Confidence estimation
        confidence = self._estimate_response_confidence(
            response, question_info, quality_metrics
        )

        return {
            "processed_response": response,
            "original_response": original_response,
            "quality_metrics": quality_metrics,
            "content_analysis": content_analysis,
            "confidence": confidence,
            "summary": summary,
            "word_count": len(response.split()),
            "estimated_reading_time": max(1, len(response.split()) // 200),  # minutes
            "language_detected": self._detect_response_language(response),
        }

    def _postprocess_chinese_response(self, response: str) -> str:
        """Postprocess Chinese responses for better readability"""
        # Add proper punctuation
        if response and not response.endswith(("。", "！", "？", ".", "!", "?")):
            response += "。"

        # Fix common spacing issues with Chinese punctuation
        response = re.sub(r"\s*([，。！？；：])\s*", r"\1", response)
        response = re.sub(r"([，。！？；：])\s*([a-zA-Z])", r"\1 \2", response)

        # Normalize whitespace
        response = re.sub(r"\s+", " ", response).strip()

        return response

    def _postprocess_english_response(self, response: str) -> str:
        """Postprocess English responses for better readability"""
        # Capitalize first letter
        if response and response[0].islower():
            response = response[0].upper() + response[1:]

        # Add proper punctuation
        if response and not response.endswith((".", "!", "?")):
            response += "."

        # Fix spacing around punctuation
        response = re.sub(r"\s+([,.!?;:])", r"\1", response)
        response = re.sub(r"([,.!?;:])\s*([a-zA-Z])", r"\1 \2", response)

        return response

    def _assess_response_quality(
        self, response: str, question_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Comprehensive response quality assessment"""
        metrics = {
            "length_score": 0.0,
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "specificity_score": 0.0,
        }

        try:
            response_length = len(response.split())
            question_complexity = question_info.get("estimated_complexity", "medium")
            question_type = question_info.get("question_type", "general")

            # Length score
            if question_complexity == "high" and response_length > 15:
                metrics["length_score"] = 0.9
            elif question_complexity == "medium" and 8 <= response_length <= 25:
                metrics["length_score"] = 0.8
            elif question_complexity == "low" and 3 <= response_length <= 15:
                metrics["length_score"] = 0.8
            elif response_length < 2:
                metrics["length_score"] = 0.2
            else:
                metrics["length_score"] = 0.6

            # Relevance score (keyword matching)
            question_text = question_info.get("processed_question", "").lower()
            response_lower = response.lower()

            # Extract key terms from question
            question_words = set(re.findall(r"\b\w+\b", question_text))
            response_words = set(re.findall(r"\b\w+\b", response_lower))

            if question_words:
                overlap = len(question_words.intersection(response_words))
                metrics["relevance_score"] = min(1.0, overlap / len(question_words) * 2)

            # Completeness score (based on question type)
            if question_type == "counting" and re.search(r"\d+", response):
                metrics["completeness_score"] += 0.3
            if question_type == "yes_no" and any(
                word in response_lower
                for word in ["yes", "no", "是", "不是", "有", "沒有"]
            ):
                metrics["completeness_score"] += 0.3
            if question_type == "color" and any(
                color in response_lower
                for color in [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "white",
                    "紅",
                    "藍",
                    "綠",
                    "黃",
                    "黑",
                    "白",
                ]
            ):
                metrics["completeness_score"] += 0.3

            # Base completeness
            metrics["completeness_score"] += 0.5
            metrics["completeness_score"] = min(1.0, metrics["completeness_score"])

            # Clarity score (sentence structure)
            sentences = re.split(r"[.!?。！？]", response)
            complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            if complete_sentences:
                avg_sentence_length = sum(
                    len(s.split()) for s in complete_sentences
                ) / len(complete_sentences)
                if 5 <= avg_sentence_length <= 20:
                    metrics["clarity_score"] = 0.8
                else:
                    metrics["clarity_score"] = 0.6
            else:
                metrics["clarity_score"] = 0.4

            # Specificity score (avoid generic responses)
            generic_patterns = [
                "i cannot",
                "unable to",
                "不能",
                "無法",
                "不知道",
                "不清楚",
                "cannot tell",
                "unclear",
                "difficult to say",
            ]

            if any(pattern in response_lower for pattern in generic_patterns):
                metrics["specificity_score"] = 0.2
            elif (
                len(set(response_lower.split())) > len(response_lower.split()) * 0.7
            ):  # Vocabulary diversity
                metrics["specificity_score"] = 0.8
            else:
                metrics["specificity_score"] = 0.6

        except Exception as e:
            logger.error(f"Response quality assessment failed: {e}")

        return metrics

    def _analyze_response_content(
        self, response: str, question_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze response content for insights"""
        try:
            analysis = {
                "entities_mentioned": [],
                "sentiment": "neutral",
                "confidence_indicators": [],
                "uncertainty_indicators": [],
                "factual_claims": 0,
            }

            response_lower = response.lower()

            # Simple entity extraction (numbers, colors, common objects)
            numbers = re.findall(r"\d+", response)
            if numbers:
                analysis["entities_mentioned"].extend([f"number: {n}" for n in numbers])

            colors = re.findall(
                r"\b(?:red|blue|green|yellow|black|white|紅|藍|綠|黃|黑|白)\b",
                response_lower,
            )
            if colors:
                analysis["entities_mentioned"].extend([f"color: {c}" for c in colors])

            # Confidence indicators
            confidence_words = [
                "clearly",
                "definitely",
                "certainly",
                "確實",
                "明確",
                "肯定",
            ]
            uncertainty_words = ["maybe", "possibly", "might", "可能", "也許", "大概"]

            analysis["confidence_indicators"] = [
                word for word in confidence_words if word in response_lower
            ]
            analysis["uncertainty_indicators"] = [
                word for word in uncertainty_words if word in response_lower
            ]

            # Simple sentiment analysis
            positive_words = ["good", "beautiful", "nice", "great", "好", "美", "棒"]
            negative_words = ["bad", "ugly", "poor", "terrible", "壞", "醜", "糟"]

            pos_count = sum(1 for word in positive_words if word in response_lower)
            neg_count = sum(1 for word in negative_words if word in response_lower)

            if pos_count > neg_count:
                analysis["sentiment"] = "positive"
            elif neg_count > pos_count:
                analysis["sentiment"] = "negative"

            # Count factual claims (sentences with specific information)
            sentences = re.split(r"[.!?。！？]", response)
            factual_patterns = [
                r"\d+",
                r"\b(?:is|are|has|have|shows|contains|包含|顯示|有)\b",
            ]

            for sentence in sentences:
                if any(
                    re.search(pattern, sentence.lower()) for pattern in factual_patterns
                ):
                    analysis["factual_claims"] += 1

            return analysis

        except Exception as e:
            logger.error(f"Response content analysis failed: {e}")
            return {"error": str(e)}

    def _detect_response_language(self, response: str) -> str:
        """Detect the primary language of the response"""
        chinese_chars = len(self.chinese_patterns["traditional"].findall(response))
        english_chars = len(re.findall(r"[a-zA-Z]", response))
        total_chars = len(re.sub(r"\s", "", response))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.5:
            return "chinese"
        elif english_ratio > 0.5:
            return "english"
        elif chinese_ratio > 0.2 and english_ratio > 0.2:
            return "mixed"
        else:
            return "other"

    def _estimate_response_confidence(
        self,
        response: str,
        question_info: Dict[str, Any],
        quality_metrics: Dict[str, float],
    ) -> float:
        """Estimate overall confidence in the response"""
        try:
            # Base confidence from quality metrics
            base_confidence = sum(quality_metrics.values()) / len(quality_metrics)

            # Adjust based on response characteristics
            response_lower = response.lower()

            # Penalty for uncertainty expressions
            uncertainty_penalty = 0.0
            uncertainty_words = [
                "maybe",
                "possibly",
                "might",
                "unclear",
                "可能",
                "也許",
                "不確定",
            ]
            for word in uncertainty_words:
                if word in response_lower:
                    uncertainty_penalty += 0.1

            # Bonus for specific information
            specificity_bonus = 0.0
            if re.search(r"\d+", response):  # Contains numbers
                specificity_bonus += 0.1
            if len(response.split()) > 10:  # Detailed response
                specificity_bonus += 0.05

            # Question type adjustment
            question_type = question_info.get("question_type", "general")
            if question_type in ["yes_no", "counting"] and len(response.split()) < 5:
                # Short answers appropriate for simple questions
                specificity_bonus += 0.1

            final_confidence = base_confidence + specificity_bonus - uncertainty_penalty
            return max(0.0, min(1.0, final_confidence))

        except Exception:
            return 0.5

    def _generate_summary(self, response: str, max_length: int = 50) -> str:
        """Generate a summary of long responses"""
        if len(response) <= max_length:
            return response

        try:
            # Simple extractive summary - take first sentence and key points
            sentences = re.split(r"[.!?。！？]", response)

            if not sentences:
                return response[:max_length] + "..."

            # Start with first sentence
            summary = sentences[0].strip()

            if len(summary) > max_length:
                # Truncate first sentence
                words = summary.split()
                truncated = ""
                for word in words:
                    if len(truncated + word) <= max_length - 3:
                        truncated += word + " "
                    else:
                        break
                return truncated.strip() + "..."

            # Try to add more sentences if there's room
            for sentence in sentences[1:]:
                sentence = sentence.strip()
                if sentence and len(summary + " " + sentence) <= max_length:
                    summary += " " + sentence
                else:
                    break

            return summary + ("..." if len(response) > len(summary) + 10 else "")

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return response[:max_length] + "..."

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
            # Simple keyword extraction based on frequency and importance
            words = re.findall(r"\b\w+\b", text.lower())

            # Filter out common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
            }

            # Count word frequencies
            word_freq = {}
            for word in words:
                if len(word) > 2 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Sort by frequency and return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:max_keywords]]

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []


class TextProcessor:
    """Text processing utilities for VLM"""

    @staticmethod
    def preprocess_question(question: str, target_language: str = "auto") -> str:
        """Preprocess question for VLM input"""
        # Clean whitespace
        question = " ".join(question.split()).strip()

        # Add language hints if needed
        if target_language == "zh" or TextProcessor.detect_chinese(question):
            if not question.startswith(("請", "用中文", "用繁體中文")):
                question = f"請用中文回答：{question}"

        return question

    @staticmethod
    def postprocess_answer(answer: str, question: str = "") -> str:
        """Postprocess VLM answer"""
        # Remove repeated question
        if question and question.lower() in answer.lower():
            answer = re.sub(
                re.escape(question), "", answer, flags=re.IGNORECASE
            ).strip()

        # Remove common prefixes
        prefixes = [
            r"^(the\s+answer\s+is\s*:?\s*)",
            r"^(answer\s*:?\s*)",
            r"^(based\s+on\s+the\s+image\s*,?\s*)",
            r"^(looking\s+at\s+the\s+image\s*,?\s*)",
            r"^(in\s+the\s+image\s*,?\s*)",
            r"^(根據圖片\s*,?\s*)",
            r"^(從圖片中可以看到\s*,?\s*)",
        ]

        for prefix_pattern in prefixes:
            answer = re.sub(prefix_pattern, "", answer, flags=re.IGNORECASE).strip()

        # Clean up whitespace
        answer = " ".join(answer.split())

        # Ensure proper sentence ending
        if answer and not answer.endswith((".", "!", "?", "。", "！", "？")):
            if TextProcessor.detect_chinese(answer):
                answer += "。"
            else:
                answer += "."

        return answer

    @staticmethod
    def detect_chinese(text: str) -> bool:
        """Detect if text contains Chinese characters"""
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def detect_traditional_chinese(text: str) -> bool:
        """Detect Traditional Chinese using common character patterns"""
        traditional_chars = [
            "繁體",
            "臺灣",
            "進行",
            "開始",
            "關於",
            "問題",
            "說明",
            "資訊",
            "環境",
            "檔案",
            "網路",
            "電腦",
            "軟體",
            "應用",
            "設定",
            "選擇",
            "確認",
            "執行",
            "處理",
            "結果",
            "內容",
            "畫面",
            "視窗",
            "檔案",
        ]
        return any(char in text for char in traditional_chars)

    @staticmethod
    def get_language_code(text: str) -> str:
        """Get language code for text"""
        if TextProcessor.detect_chinese(text):
            if TextProcessor.detect_traditional_chinese(text):
                return "zh-TW"
            else:
                return "zh-CN"
        return "en"


class PromptTemplate:
    """VLM prompt templates for different tasks"""

    CAPTION_TEMPLATES = {
        "en": "Describe this image in detail:",
        "zh": "請詳細描述這張圖片：",
        "zh-TW": "請詳細描述這張圖片：",
    }

    VQA_TEMPLATES = {
        "en": "Question: {question}\nAnswer:",
        "zh": "問題：{question}\n答案：",
        "zh-TW": "問題：{question}\n答案：",
    }

    SYSTEM_PROMPTS = {
        "caption": {
            "en": "You are an AI assistant that describes images accurately and concisely.",
            "zh": "你是一個能夠準確簡潔地描述圖片的AI助手。",
            "zh-TW": "你是一個能夠準確簡潔地描述圖片的AI助手。",
        },
        "vqa": {
            "en": "You are an AI assistant that answers questions about images accurately.",
            "zh": "你是一個能夠準確回答圖片相關問題的AI助手。",
            "zh-TW": "你是一個能夠準確回答圖片相關問題的AI助手。",
        },
    }

    @classmethod
    def get_caption_prompt(cls, language: str = "en") -> str:
        """Get caption generation prompt"""
        return cls.CAPTION_TEMPLATES.get(language, cls.CAPTION_TEMPLATES["en"])

    @classmethod
    def get_vqa_prompt(cls, question: str, language: str = "auto") -> str:
        """Get VQA prompt with question"""
        if language == "auto":
            language = TextProcessor.get_language_code(question)

        template = cls.VQA_TEMPLATES.get(language, cls.VQA_TEMPLATES["en"])
        return template.format(question=question)

    @classmethod
    def get_system_prompt(cls, task: str, language: str = "en") -> str:
        """Get system prompt for task"""
        task_prompts = cls.SYSTEM_PROMPTS.get(task, {})
        return task_prompts.get(language, task_prompts.get("en", ""))


class ModelCompatibility:
    """Handle compatibility across different VLM models"""

    # Model-specific configurations
    MODEL_CONFIGS = {
        "blip": {
            "supports_chat_template": False,
            "input_format": "text_image",
            "max_length": 512,
            "special_tokens": {"pad": "[PAD]", "eos": "[SEP]"},
        },
        "blip2": {
            "supports_chat_template": False,
            "input_format": "text_image",
            "max_length": 512,
            "special_tokens": {"pad": "[PAD]", "eos": "</s>"},
        },
        "llava": {
            "supports_chat_template": True,
            "input_format": "conversation",
            "max_length": 2048,
            "special_tokens": {"pad": "<pad>", "eos": "</s>"},
        },
        "qwen": {
            "supports_chat_template": True,
            "input_format": "conversation",
            "max_length": 2048,
            "special_tokens": {"pad": "<|endoftext|>", "eos": "<|endoftext|>"},
        },
    }

    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """Detect model type from model name"""
        model_name_lower = model_name.lower()

        if "llava" in model_name_lower:
            return "llava"
        elif "qwen" in model_name_lower:
            return "qwen"
        elif "blip2" in model_name_lower:
            return "blip2"
        elif "blip" in model_name_lower:
            return "blip"
        else:
            return "unknown"

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for model"""
        model_type = cls.get_model_type(model_name)
        return cls.MODEL_CONFIGS.get(model_type, cls.MODEL_CONFIGS["blip2"])

    @classmethod
    def prepare_inputs(
        cls, processor, image: Image.Image, text: str, model_name: str
    ) -> Dict[str, Any]:
        """Prepare inputs based on model type"""
        config = cls.get_model_config(model_name)

        if config["supports_chat_template"] and hasattr(
            processor, "apply_chat_template"
        ):
            # Modern chat-based models (LLaVA, Qwen-VL)
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": text}],
                }
            ]

            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = processor(prompt, images=image, return_tensors="pt")
        else:
            # Legacy models (BLIP, BLIP-2)
            inputs = processor(images=image, text=text, return_tensors="pt")

        return inputs

    @classmethod
    def get_generation_config(cls, model_name: str, **kwargs) -> Dict[str, Any]:
        """Get generation configuration for model"""
        config = cls.get_model_config(model_name)

        generation_config = {
            "max_length": kwargs.get("max_length", config["max_length"]),
            "temperature": kwargs.get("temperature", 0.7),
            "do_sample": kwargs.get("do_sample", False),
            "num_beams": kwargs.get("num_beams", 3),
        }

        # Add model-specific parameters
        if "pad_token_id" not in kwargs:
            generation_config["pad_token_id"] = None  # Will be set by processor

        return generation_config


class OutputFormatter:
    """Format VLM outputs for different use cases"""

    @staticmethod
    def format_caption_response(
        caption: str,
        confidence: float,
        model_name: str,
        parameters: Dict[str, Any],
        image_info: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, Any]:
        """Format caption generation response"""
        return {
            "caption": caption,
            "confidence": round(confidence, 3),
            "model_used": model_name,
            "language": language,
            "parameters": parameters,
            "image_info": image_info,
            "timestamp": (
                torch.cuda.Event().query() if torch.cuda.is_available() else None
            ),
        }

    @staticmethod
    def format_vqa_response(
        question: str,
        answer: str,
        confidence: float,
        model_name: str,
        parameters: Dict[str, Any],
        image_info: Dict[str, Any],
        language_detected: str = "en",
    ) -> Dict[str, Any]:
        """Format VQA response"""
        return {
            "question": question,
            "answer": answer,
            "confidence": round(confidence, 3),
            "model_used": model_name,
            "language_detected": language_detected,
            "parameters": parameters,
            "image_info": image_info,
            "timestamp": (
                torch.cuda.Event().query() if torch.cuda.is_available() else None
            ),
        }

    @staticmethod
    def format_batch_response(
        results: List[Dict[str, Any]],
        total_time: float,
        successful_count: int,
        failed_count: int,
    ) -> Dict[str, Any]:
        """Format batch processing response"""
        return {
            "results": results,
            "summary": {
                "total_items": len(results),
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": successful_count / len(results) if results else 0,
                "total_time_seconds": round(total_time, 2),
                "average_time_per_item": (
                    round(total_time / len(results), 2) if results else 0
                ),
            },
        }
