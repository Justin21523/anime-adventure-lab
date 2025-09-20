# core/train/evaluators.py
import torch
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import clip
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import os

from core.config import get_config
from core.shared_cache import get_shared_cache
from ..utils.image import get_image_processor
from ..exceptions import EvaluationError


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    clip_score: float = 0.0
    aesthetic_score: float = 0.0
    consistency_score: float = 0.0
    face_similarity: float = 0.0
    prompt_adherence: float = 0.0
    generation_time: float = 0.0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "clip_score": self.clip_score,
            "aesthetic_score": self.aesthetic_score,
            "consistency_score": self.consistency_score,
            "face_similarity": self.face_similarity,
            "prompt_adherence": self.prompt_adherence,
            "generation_time": self.generation_time,
            "memory_usage_mb": self.memory_usage_mb,
        }


class ModelEvaluator:
    """Comprehensive model evaluation system"""

    def __init__(self):
        self.config = get_config()
        self.image_processor = get_image_processor()

        # Load evaluation models
        self._load_evaluation_models()

        # Evaluation prompts
        self.test_prompts = [
            "a photo of a person",
            "anime character portrait",
            "a beautiful landscape",
            "abstract art painting",
            "a cute cat",
            "architectural building",
            "fantasy creature",
            "still life painting",
        ]

    def _load_evaluation_models(self):
        """Load models for evaluation"""
        try:
            # CLIP for text-image similarity
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            logger.info("✅ Evaluation models loaded")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load evaluation models: {e}")
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_processor = None

    def evaluate_image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Evaluate CLIP similarity between image and text"""
        if not self.clip_model:
            return 0.0

        try:
            # Preprocess image and text
            image_input = self.clip_preprocess(image).unsqueeze(0)
            text_input = clip.tokenize([text])

            # Move to device
            device = next(self.clip_model.parameters()).device
            image_input = image_input.to(device)
            text_input = text_input.to(device)

            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    image_features, text_features, dim=-1
                )

            return float(similarity.cpu().item())

        except Exception as e:
            logger.error(f"❌ CLIP evaluation failed: {e}")
            return 0.0

    def evaluate_aesthetic_quality(self, image: Image.Image) -> float:
        """Evaluate aesthetic quality using heuristics"""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Color distribution
            color_variance = np.var(img_array, axis=(0, 1)).mean()
            color_score = min(color_variance / 1000.0, 1.0)  # Normalize

            # Edge detection for composition
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            composition_score = min(edge_density * 10, 1.0)

            # Brightness and contrast
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0

            # Avoid over/under exposure
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            contrast_score = min(contrast * 4, 1.0)

            # Combined aesthetic score
            aesthetic_score = (
                color_score * 0.3
                + composition_score * 0.3
                + brightness_score * 0.2
                + contrast_score * 0.2
            )

            return float(aesthetic_score)

        except Exception as e:
            logger.error(f"❌ Aesthetic evaluation failed: {e}")
            return 0.0

    def evaluate_consistency(self, images: List[Image.Image]) -> float:
        """Evaluate consistency across multiple generated images"""
        if len(images) < 2:
            return 1.0

        try:
            # Extract CLIP features for all images
            features = []
            for image in images:
                if self.clip_model:
                    image_input = self.clip_preprocess(image).unsqueeze(0)
                    device = next(self.clip_model.parameters()).device
                    image_input = image_input.to(device)

                    with torch.no_grad():
                        feature = self.clip_model.encode_image(image_input)
                        feature = feature / feature.norm(dim=-1, keepdim=True)
                        features.append(feature.cpu().numpy())

            if not features:
                return 0.0

            # Calculate pairwise similarities
            features_array = np.vstack(features)
            similarities = cosine_similarity(features_array)

            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
            avg_similarity = similarities[mask].mean()

            return float(avg_similarity)

        except Exception as e:
            logger.error(f"❌ Consistency evaluation failed: {e}")
            return 0.0

    def evaluate_face_similarity(
        self, reference_image: Image.Image, generated_images: List[Image.Image]
    ) -> float:
        """Evaluate face similarity for character consistency"""
        try:
            # Simple face detection using OpenCV
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            # Extract face from reference
            ref_array = np.array(reference_image)
            ref_gray = cv2.cvtColor(ref_array, cv2.COLOR_RGB2GRAY)
            ref_faces = face_cascade.detectMultiScale(ref_gray, 1.1, 4)

            if len(ref_faces) == 0:
                return 0.0  # No face in reference

            # Get largest face
            ref_face = max(ref_faces, key=lambda x: x[2] * x[3])
            x, y, w, h = ref_face
            ref_face_img = ref_array[y : y + h, x : x + w]

            similarities = []
            for gen_image in generated_images:
                gen_array = np.array(gen_image)
                gen_gray = cv2.cvtColor(gen_array, cv2.COLOR_RGB2GRAY)
                gen_faces = face_cascade.detectMultiScale(gen_gray, 1.1, 4)

                if len(gen_faces) > 0:
                    # Get largest face
                    gen_face = max(gen_faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = gen_face
                    gen_face_img = gen_array[y : y + h, x : x + w]

                    # Simple correlation-based similarity
                    try:
                        # Resize to same size
                        ref_resized = cv2.resize(ref_face_img, (64, 64))
                        gen_resized = cv2.resize(gen_face_img, (64, 64))

                        # Calculate normalized cross-correlation
                        correlation = cv2.matchTemplate(
                            ref_resized, gen_resized, cv2.TM_CCOEFF_NORMED
                        )[0, 0]
                        similarities.append(max(0, correlation))

                    except Exception:
                        pass

            return float(np.mean(similarities)) if similarities else 0.0

        except Exception as e:
            logger.error(f"❌ Face similarity evaluation failed: {e}")
            return 0.0

    def evaluate_model(
        self,
        model_pipeline,
        test_prompts: Optional[List[str]] = None,
        num_images_per_prompt: int = 4,
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        test_prompts = test_prompts or self.test_prompts

        results = {
            "overall_metrics": {},
            "per_prompt_metrics": [],
            "generation_stats": {},
        }

        all_clip_scores = []
        all_aesthetic_scores = []
        all_consistency_scores = []
        total_generation_time = 0
        peak_memory_usage = 0

        logger.info(f"🔍 Starting evaluation with {len(test_prompts)} prompts")

        for i, prompt in enumerate(test_prompts):
            logger.info(f"Evaluating prompt {i+1}/{len(test_prompts)}: {prompt}")

            # Generate images
            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                # Generate multiple images for consistency evaluation
                generated_images = []
                for _ in range(num_images_per_prompt):
                    with torch.no_grad():
                        result = model_pipeline(
                            prompt,
                            num_inference_steps=25,
                            guidance_scale=7.5,
                            width=512,
                            height=512,
                        )
                        generated_images.append(result.images[0])

                generation_time = time.time() - start_time
                total_generation_time += generation_time

                # Memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1024**2
                    peak_memory_usage = max(peak_memory_usage, memory_used)
                else:
                    memory_used = 0

                # Evaluate metrics
                clip_scores = [
                    self.evaluate_image_text_similarity(img, prompt)
                    for img in generated_images
                ]
                aesthetic_scores = [
                    self.evaluate_aesthetic_quality(img) for img in generated_images
                ]
                consistency_score = self.evaluate_consistency(generated_images)

                prompt_metrics = {
                    "prompt": prompt,
                    "clip_score": np.mean(clip_scores),
                    "clip_score_std": np.std(clip_scores),
                    "aesthetic_score": np.mean(aesthetic_scores),
                    "aesthetic_score_std": np.std(aesthetic_scores),
                    "consistency_score": consistency_score,
                    "generation_time": generation_time,
                    "memory_usage_mb": memory_used,
                    "num_images": len(generated_images),
                }

                results["per_prompt_metrics"].append(prompt_metrics)

                # Accumulate for overall metrics
                all_clip_scores.extend(clip_scores)
                all_aesthetic_scores.extend(aesthetic_scores)
                all_consistency_scores.append(consistency_score)

            except Exception as e:
                logger.error(f"❌ Failed to evaluate prompt '{prompt}': {e}")
                # Add failed prompt metrics
                results["per_prompt_metrics"].append(
                    {
                        "prompt": prompt,
                        "error": str(e),
                        "clip_score": 0.0,
                        "aesthetic_score": 0.0,
                        "consistency_score": 0.0,
                        "generation_time": 0.0,
                        "memory_usage_mb": 0.0,
                        "num_images": 0,
                    }
                )

        # Calculate overall metrics
        results["overall_metrics"] = {
            "clip_score": np.mean(all_clip_scores) if all_clip_scores else 0.0,
            "clip_score_std": np.std(all_clip_scores) if all_clip_scores else 0.0,
            "aesthetic_score": (
                np.mean(all_aesthetic_scores) if all_aesthetic_scores else 0.0
            ),
            "aesthetic_score_std": (
                np.std(all_aesthetic_scores) if all_aesthetic_scores else 0.0
            ),
            "consistency_score": (
                np.mean(all_consistency_scores) if all_consistency_scores else 0.0
            ),
            "consistency_score_std": (
                np.std(all_consistency_scores) if all_consistency_scores else 0.0
            ),
        }

        results["generation_stats"] = {
            "total_generation_time": total_generation_time,
            "avg_generation_time": total,
        }


class TrainingEvaluator:
    """Evaluate training progress and model quality"""

    def __init__(self):
        self._clip_model = None
        self._face_model = None

    def evaluate_batch(
        self,
        generated_images: List[Image.Image],
        target_prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, float]:
        """Evaluate a batch of generated images"""
        try:
            batch_size = len(generated_images)

            # Mock evaluation metrics
            clip_scores = np.random.random(batch_size) * 0.3 + 0.7  # 0.7-1.0
            aesthetic_scores = np.random.random(batch_size) * 0.4 + 0.6  # 0.6-1.0

            # Face similarity (if reference images provided)
            face_similarity = 0.0
            if reference_images and len(reference_images) == batch_size:
                face_similarities = np.random.random(batch_size) * 0.3 + 0.7
                face_similarity = np.mean(face_similarities)

            return {
                "clip_score_mean": float(np.mean(clip_scores)),
                "clip_score_std": float(np.std(clip_scores)),
                "aesthetic_score_mean": float(np.mean(aesthetic_scores)),
                "aesthetic_score_std": float(np.std(aesthetic_scores)),
                "face_similarity": face_similarity,  # type: ignore
                "batch_size": batch_size,
            }

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def compute_fid(
        self, generated_images: List[Image.Image], real_images: List[Image.Image]
    ) -> float:
        """Compute Fréchet Inception Distance"""
        # Mock FID computation
        return np.random.random() * 20 + 10  # 10-30 range

    def tag_consistency_score(
        self, generated_images: List[Image.Image], expected_tags: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate tag consistency in generated images"""
        # Mock tag consistency evaluation
        consistency_scores = np.random.random(len(generated_images)) * 0.4 + 0.6

        return {
            "consistency_mean": float(np.mean(consistency_scores)),
            "consistency_std": float(np.std(consistency_scores)),
            "tag_accuracy": float(np.random.random() * 0.3 + 0.7),
        }


class CLIPSimilarityEvaluator:
    """Evaluate image-text similarity using CLIP"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.cache = get_shared_cache()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load CLIP model"""
        try:
            cache_dir = self.cache.get_path("MODELS_CLIP")
            self.model = CLIPModel.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name, cache_dir=cache_dir
            )

            if torch.cuda.is_available():
                self.model = self.model.cuda()  # type: ignore

        except Exception as e:
            print(f"Failed to load CLIP model: {e}")

    def evaluate_similarity(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[float]:
        """Compute CLIP similarity scores"""
        if not self.model or not self.processor:
            return [0.5] * len(images)  # Return dummy scores

        try:
            # Process inputs
            inputs = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Compute similarities
            similarities = torch.cosine_similarity(
                outputs.image_embeds, outputs.text_embeds, dim=1
            )

            return similarities.cpu().numpy().tolist()

        except Exception as e:
            print(f"CLIP evaluation failed: {e}")
            return [0.5] * len(images)


class TagConsistencyEvaluator:
    """Evaluate tag consistency in generated images"""

    def __init__(self):
        self.expected_tags = set()

    def set_expected_tags(self, tags: List[str]):
        """Set expected tags for evaluation"""
        self.expected_tags = set(tag.lower().strip() for tag in tags)

    def evaluate_consistency(self, generated_tags: List[List[str]]) -> Dict[str, float]:
        """Evaluate tag consistency"""
        if not self.expected_tags:
            return {"consistency": 1.0}

        total_precision = 0.0
        total_recall = 0.0

        for tags in generated_tags:
            tags_set = set(tag.lower().strip() for tag in tags)

            # Precision: how many predicted tags are expected
            precision = (
                len(tags_set.intersection(self.expected_tags)) / len(tags_set)
                if tags_set
                else 0.0
            )

            # Recall: how many expected tags are predicted
            recall = len(tags_set.intersection(self.expected_tags)) / len(
                self.expected_tags
            )

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / len(generated_tags)
        avg_recall = total_recall / len(generated_tags)

        # F1 score
        f1 = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1,
            "consistency": f1,  # Use F1 as overall consistency score
        }
