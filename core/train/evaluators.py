# core/train/evaluators.py
import torch
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict
import clip
import cv2
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import os

import clip
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

from core.config import get_config
from core.shared_cache import get_shared_cache
from ..utils.image import get_image_processor
from ..exceptions import EvaluationError, ModelLoadError

logger = logging.getLogger(__name__)


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

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.clip_model = None
        self.clip_processor = None
        self.clip_preprocess = None  # Fix: 添加缺失的屬性
        self._models_loaded = False

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

        self.config = get_config()
        self.image_processor = get_image_processor()

        # Load evaluation models
        self._load_evaluation_models()

    def _load_evaluation_models(self):
        """Load models for evaluation"""
        if self._models_loaded:
            return

        try:
            # CLIP for text-image similarity
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            logger.info("✅ Evaluation models loaded")

            self._models_loaded = True

        except Exception as e:
            logger.warning(f"⚠️ Failed to load evaluation models: {e}")
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_processor = None
            raise EvaluationError(
                f"Failed to load evaluation models: {e}", "model_loading"
            )

    def evaluate_model(
        self,
        model_pipeline,
        test_prompts: Optional[List[str]] = None,
        num_images_per_prompt: int = 4,
    ) -> Dict[str, Any]:  # type: ignore
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
        }

    def _evaluate_prompt(
        self, model_path: str, prompt: str, num_samples: int, **generation_kwargs
    ) -> EvaluationMetrics:
        """Evaluate model on a single prompt"""

        metrics = EvaluationMetrics()

        try:
            # Simulate image generation (replace with actual generation)
            start_time = time.time()

            # This would be replaced with actual model inference
            generated_images = self._simulate_generation(prompt, num_samples)

            generation_time = time.time() - start_time
            metrics.generation_time = generation_time

            # Calculate metrics
            if generated_images:
                metrics.clip_score = self._calculate_clip_score(
                    prompt, generated_images
                )
                metrics.aesthetic_score = self._calculate_aesthetic_score(
                    generated_images
                )
                metrics.consistency_score = self._calculate_consistency_score(
                    generated_images
                )

            # Memory usage (approximate)
            if torch.cuda.is_available():
                metrics.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024

            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate prompt '{prompt}': {e}")
            return metrics

    def _simulate_generation(self, prompt: str, num_samples: int) -> List[Image.Image]:
        """Simulate image generation (replace with actual generation)"""
        # This is a placeholder - replace with actual model generation
        images = []
        for _ in range(num_samples):
            # Create a dummy image
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))
            images.append(img)
        return images

    def _calculate_clip_score(self, prompt: str, images: List[Image.Image]) -> float:
        """Calculate CLIP score between prompt and images"""
        if not self.clip_model:
            return 0.0

        try:
            # Preprocess text
            text = clip.tokenize([prompt]).to(self.device)

            scores = []
            for image in images:
                # Preprocess image
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)  # type: ignore

                with torch.no_grad():
                    # Calculate features
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text)

                    # Calculate similarity
                    similarity = torch.cosine_similarity(image_features, text_features)
                    scores.append(float(similarity))

            return float(np.mean(scores)) if scores else 0.0

        except Exception as e:
            logger.error(f"CLIP score calculation failed: {e}")
            return 0.0

    def _calculate_aesthetic_score(self, images: List[Image.Image]) -> float:
        """Calculate aesthetic score (placeholder implementation)"""
        return float(np.random.uniform(5.0, 8.0))

    def _calculate_consistency_score(self, images: List[Image.Image]) -> float:
        """Calculate consistency score between generated images"""
        if len(images) < 2:
            return 0.0

        try:
            # Extract CLIP features for all images
            features = []
            for image in images:
                if self.clip_model:
                    image_input = (
                        self.clip_preprocess(image).unsqueeze(0).to(self.device)  # type: ignore
                    )
                    with torch.no_grad():
                        feature = self.clip_model.encode_image(image_input)
                        features.append(feature.cpu().numpy())

            if len(features) < 2:
                return 0.0

            # Calculate pairwise similarities
            features_array = np.vstack(features)
            similarity_matrix = cosine_similarity(features_array)

            # Get upper triangular part (excluding diagonal)
            upper_triangular = similarity_matrix[
                np.triu_indices_from(similarity_matrix, k=1)
            ]

            return (
                float(np.mean(upper_triangular)) if len(upper_triangular) > 0 else 0.0
            )

        except Exception as e:
            logger.error(f"Consistency score calculation failed: {e}")
            return 0.0

    def _calculate_overall_metrics(
        self, all_metrics: List[EvaluationMetrics]
    ) -> EvaluationMetrics:
        """Calculate overall metrics from per-prompt metrics"""
        if not all_metrics:
            return EvaluationMetrics()

        # Calculate averages
        overall = EvaluationMetrics()

        # Calculate means for each metric
        overall.clip_score = np.mean([m.clip_score for m in all_metrics])  # type: ignore
        overall.aesthetic_score = np.mean([m.aesthetic_score for m in all_metrics])  # type: ignore
        overall.consistency_score = np.mean([m.consistency_score for m in all_metrics])  # type: ignore
        overall.face_similarity = np.mean([m.face_similarity for m in all_metrics])  # type: ignore
        overall.prompt_adherence = np.mean([m.prompt_adherence for m in all_metrics])  # type: ignore
        overall.generation_time = np.mean([m.generation_time for m in all_metrics])  # type: ignore
        overall.memory_usage_mb = np.max([m.memory_usage_mb for m in all_metrics])  # type: ignore

        return overall

    def evaluate_consistency(self, images: List[Image.Image]) -> float:
        """Public method to evaluate consistency between images"""
        return self._calculate_consistency_score(images)

    def compare_models(
        self, model_paths: List[str], test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models"""

        results = {
            "models": model_paths,
            "comparison": {},
            "rankings": {},
        }

        model_results = {}

        # Evaluate each model
        for model_path in model_paths:
            try:
                model_metrics = self.evaluate_model(model_path, test_prompts)
                model_results[model_path] = model_metrics["metrics"]
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_path}: {e}")
                model_results[model_path] = {}

        results["comparison"] = model_results

        # Create rankings
        if model_results:
            metrics_to_rank = ["clip_score", "aesthetic_score", "consistency_score"]

            for metric in metrics_to_rank:
                # Sort models by this metric (descending)
                sorted_models = sorted(
                    model_paths,
                    key=lambda x: model_results.get(x, {}).get(metric, 0),
                    reverse=True,
                )
                results["rankings"][metric] = sorted_models

        return results

    def evaluate_image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Evaluate CLIP similarity between image and text"""
        if not self.clip_model:
            return 0.0

        try:
            # Preprocess image and text
            image_input = self.clip_preprocess(image).unsqueeze(0)  # type: ignore
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

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""

        report_lines = [
            "# Model Evaluation Report",
            f"**Model Path:** {evaluation_results.get('model_path', 'Unknown')}",
            "",
            "## Overall Metrics",
        ]

        metrics = evaluation_results.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(
                    f"- **{metric_name.replace('_', ' ').title()}:** {value:.3f}"
                )

        report_lines.extend(
            [
                "",
                "## Per-Prompt Results",
            ]
        )

        per_prompt = evaluation_results.get("per_prompt_metrics", {})
        for prompt, prompt_metrics in per_prompt.items():
            report_lines.extend(
                [
                    f'### "{prompt}"',
                    f"- CLIP Score: {prompt_metrics.clip_score:.3f}",
                    f"- Aesthetic Score: {prompt_metrics.aesthetic_score:.3f}",
                    f"- Generation Time: {prompt_metrics.generation_time:.2f}s",
                    "",
                ]
            )

        return "\n".join(report_lines)

    def evaluate_face_similarity(
        self, reference_image: Image.Image, generated_images: List[Image.Image]
    ) -> float:
        """Evaluate face similarity for character consistency"""
        try:
            # Simple face detection using OpenCV
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
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


class TrainingEvaluator(ModelEvaluator):
    """Training-specific evaluator (backward compatibility)"""

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    def evaluate_training_batch(
        self,
        generated_images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Evaluate a training batch"""

        if len(generated_images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")

        try:
            self._load_evaluation_models()

            clip_scores = []
            aesthetic_scores = []
            face_similarities = []

            for image, prompt in zip(generated_images, prompts):
                # CLIP score
                if self.clip_model:
                    clip_score = self._calculate_clip_score(prompt, [image])
                    clip_scores.append(clip_score)

                # Aesthetic score (mock)
                aesthetic_score = self._calculate_aesthetic_score([image])
                aesthetic_scores.append(aesthetic_score)

                # Face similarity (mock)
                face_similarity = np.random.uniform(0.6, 0.9)
                face_similarities.append(face_similarity)

            return {
                "clip_score_mean": float(np.mean(clip_scores)) if clip_scores else 0.0,
                "clip_score_std": float(np.std(clip_scores)) if clip_scores else 0.0,
                "aesthetic_score_mean": float(np.mean(aesthetic_scores)),
                "aesthetic_score_std": float(np.std(aesthetic_scores)),
                "face_similarity": float(np.mean(face_similarities)),
                "batch_size": batch_size,
            }

        except Exception as e:
            raise EvaluationError(
                f"Training batch evaluation failed: {e}", "batch_evaluation"
            )


class CLIPSimilarityEvaluator:
    """Evaluate image-text similarity using CLIP"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load CLIP model"""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self.model = self.model.cuda()  # type: ignore

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")

    def evaluate_similarity(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[float]:
        """Compute CLIP similarity scores"""
        if not self.model or not self.processor:
            return [0.0] * len(images)

        try:
            similarities = []

            for image, text in zip(images, texts):
                inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt", padding=True
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    similarity = torch.cosine_similarity(
                        outputs.image_embeds, outputs.text_embeds
                    )
                    similarities.append(float(similarity))

            return similarities

        except Exception as e:
            logger.error(f"CLIP similarity evaluation failed: {e}")
            return [0.0] * len(images)

    def compute_fid(
        self, generated_images: List[Image.Image], real_images: List[Image.Image]
    ) -> float:
        """Compute Fréchet Inception Distance (mock implementation)"""
        return float(np.random.uniform(10, 30))

    def tag_consistency_score(
        self, generated_images: List[Image.Image], expected_tags: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate tag consistency in generated images (mock implementation)"""
        consistency_scores = np.random.uniform(0.6, 0.9, len(generated_images))

        return {
            "consistency_mean": float(np.mean(consistency_scores)),
            "consistency_std": float(np.std(consistency_scores)),
            "tag_accuracy": float(np.random.uniform(0.7, 0.95)),
        }


class TagConsistencyEvaluator:
    """Evaluate tag consistency in generated images"""

    def __init__(self):
        self.clip_evaluator = CLIPSimilarityEvaluator()

    def evaluate_tag_consistency(
        self, images: List[Image.Image], tags: List[str], expected_presence: List[bool]
    ) -> Dict[str, float]:
        """Evaluate how well generated images match expected tags"""

        if len(images) != len(expected_presence):
            raise ValueError("Number of images must match expected_presence list")

        tag_scores = []

        for image, tag, expected in zip(images, tags, expected_presence):
            # Use CLIP to evaluate if tag is present in image
            similarity_scores = self.clip_evaluator.evaluate_similarity([image], [tag])
            tag_present = similarity_scores[0] > 0.5  # threshold

            # Score based on whether prediction matches expectation
            if tag_present == expected:
                tag_scores.append(1.0)
            else:
                tag_scores.append(0.0)

        return {
            "tag_accuracy": float(np.mean(tag_scores)),
            "tag_precision": float(np.mean(tag_scores)),  # simplified
            "tag_recall": float(np.mean(tag_scores)),  # simplified
            "num_evaluated": len(tag_scores),
        }


# Factory functions
def get_model_evaluator(device: Optional[str] = None) -> ModelEvaluator:
    """Get model evaluator instance"""
    return ModelEvaluator(device=device)


def get_training_evaluator(device: Optional[str] = None) -> TrainingEvaluator:
    """Get training evaluator instance"""
    return TrainingEvaluator(device=device)


def get_clip_evaluator(
    model_name: str = "openai/clip-vit-base-patch32",
) -> CLIPSimilarityEvaluator:
    """Get CLIP similarity evaluator instance"""
    return CLIPSimilarityEvaluator(model_name=model_name)


def get_tag_evaluator() -> TagConsistencyEvaluator:
    """Get tag consistency evaluator instance"""
    return TagConsistencyEvaluator()
