# core/train/evaluators.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

from core.config import get_config
from core.shared_cache import get_shared_cache


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
