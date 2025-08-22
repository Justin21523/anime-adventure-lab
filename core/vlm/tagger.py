# core/vlm/tagger.py
"""
Image tagging using WD14 (Waifu Diffusion 1.4) tagger
Generates anime-style tags for character features, clothing, poses, etc.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
import logging

logger = logging.getLogger(__name__)


class WD14Tagger:
    """WD14 (Waifu Diffusion 1.4) image tagger for anime-style images"""

    def __init__(
        self,
        model_name: str = "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        device: str = "auto",
        threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.tags = None
        self._load_model()

    def _load_model(self):
        """Load WD14 model and tag labels"""
        try:
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map=self.device if self.device != "auto" else "auto",
                low_cpu_mem_usage=True,
            )

            # Load tag labels
            self.tags = self.model.config.id2label

            logger.info(f"WD14 tagger loaded: {self.model_name}")
            logger.info(f"Total tags available: {len(self.tags)}")

        except Exception as e:
            logger.error(f"Failed to load WD14 tagger: {e}")
            raise

    def tag_image(self, image: Image.Image, return_scores: bool = False) -> List[str]:
        """Generate tags for image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess image
            inputs = self.processor(image, return_tensors="pt")

            # Move to device if needed
            if self.device != "auto" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            # Filter tags by threshold
            tag_results = []
            for i, score in enumerate(predictions):
                if score >= self.threshold:
                    tag_name = self.tags[i]
                    if return_scores:
                        tag_results.append((tag_name, float(score)))
                    else:
                        tag_results.append(tag_name)

            # Sort by score if returning scores
            if return_scores:
                tag_results.sort(key=lambda x: x[1], reverse=True)
            else:
                # Sort by score for consistent ordering
                scored_tags = [
                    (self.tags[i], predictions[i])
                    for i in range(len(predictions))
                    if predictions[i] >= self.threshold
                ]
                scored_tags.sort(key=lambda x: x[1], reverse=True)
                tag_results = [tag for tag, _ in scored_tags]

            return tag_results

        except Exception as e:
            logger.error(f"Tagging failed: {e}")
            raise

    def get_character_tags(self, image: Image.Image) -> Dict[str, List[str]]:
        """Get character-specific tags organized by category"""
        all_tags = self.tag_image(image, return_scores=True)

        # Categorize tags
        categories = {
            "hair": [],
            "eyes": [],
            "clothing": [],
            "accessories": [],
            "pose": [],
            "expression": [],
            "body": [],
            "style": [],
        }

        # Define tag patterns for each category
        patterns = {
            "hair": [
                "hair",
                "ahoge",
                "bangs",
                "braid",
                "ponytail",
                "twintails",
                "hair_ornament",
            ],
            "eyes": ["eyes", "eye_color", "heterochromia", "closed_eyes", "wink"],
            "clothing": [
                "shirt",
                "dress",
                "skirt",
                "pants",
                "jacket",
                "coat",
                "uniform",
                "kimono",
                "swimsuit",
                "bikini",
                "school_uniform",
                "sailor_collar",
            ],
            "accessories": [
                "hat",
                "cap",
                "glasses",
                "earrings",
                "necklace",
                "bow",
                "ribbon",
                "gloves",
                "stockings",
                "thighhighs",
                "socks",
            ],
            "pose": [
                "sitting",
                "standing",
                "lying",
                "arms_up",
                "peace_sign",
                "pointing",
                "hands_on_hips",
                "crossed_arms",
            ],
            "expression": [
                "smile",
                "blush",
                "angry",
                "sad",
                "surprised",
                "embarrassed",
                "serious",
                "crying",
                "laughing",
            ],
            "body": [
                "large_breasts",
                "small_breasts",
                "flat_chest",
                "muscular",
                "slim",
            ],
            "style": [
                "anime",
                "realistic",
                "chibi",
                "sketch",
                "monochrome",
                "watercolor",
            ],
        }

        # Categorize tags
        for tag, score in all_tags:
            tag_lower = tag.lower().replace(" ", "_")
            categorized = False

            for category, keywords in patterns.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categories[category].append((tag, score))
                    categorized = True
                    break

            # If not categorized, add to style as general tag
            if not categorized and score > 0.5:
                categories["style"].append((tag, score))

        # Convert to simple lists and limit per category
        result = {}
        for category, tag_list in categories.items():
            if tag_list:
                # Sort by score and take top 5 per category
                tag_list.sort(key=lambda x: x[1], reverse=True)
                result[category] = [tag for tag, _ in tag_list[:5]]

        return result

    def compare_consistency(
        self, image: Image.Image, expected_tags: List[str]
    ) -> Dict[str, float | int | list[str]]:
        """Compare generated tags with expected tags for consistency"""
        generated_tags = self.tag_image(image)

        # Calculate consistency metrics
        generated_set = set(tag.lower().replace(" ", "_") for tag in generated_tags)
        expected_set = set(tag.lower().replace(" ", "_") for tag in expected_tags)

        # Intersection over union
        intersection = len(generated_set & expected_set)
        union = len(generated_set | expected_set)
        iou = intersection / union if union > 0 else 0

        # Precision and recall
        precision = intersection / len(generated_set) if generated_set else 0
        recall = intersection / len(expected_set) if expected_set else 0

        # F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "matched_tags": list(generated_set & expected_set),
            "missing_tags": list(expected_set - generated_set),
            "extra_tags": list(generated_set - expected_set),
        }

    def filter_nsfw_tags(self, tags: List[str]) -> Tuple[List[str], List[str]]:
        """Filter NSFW tags from tag list"""
        nsfw_keywords = [
            "nude",
            "naked",
            "nipples",
            "pussy",
            "penis",
            "sex",
            "nsfw",
            "explicit",
            "uncensored",
            "breasts_out",
            "topless",
            "bottomless",
            "cum",
            "orgasm",
            "masturbation",
            "pussy_juice",
        ]

        safe_tags = []
        nsfw_tags = []

        for tag in tags:
            tag_lower = tag.lower().replace(" ", "_")
            if any(keyword in tag_lower for keyword in nsfw_keywords):
                nsfw_tags.append(tag)
            else:
                safe_tags.append(tag)

        return safe_tags, nsfw_tags

    def generate_prompt_tags(self, image: Image.Image, style: str = "anime") -> str:
        """Generate comma-separated tags suitable for AI art prompts"""
        character_tags = self.get_character_tags(image)

        # Build prompt from most important categories
        prompt_parts = []

        # Character appearance (high priority)
        for category in ["hair", "eyes", "clothing", "accessories"]:
            if category in character_tags:
                prompt_parts.extend(character_tags[category][:3])  # Top 3 per category

        # Pose and expression (medium priority)
        for category in ["pose", "expression"]:
            if category in character_tags:
                prompt_parts.extend(character_tags[category][:2])  # Top 2 per category

        # Style tags
        if style:
            prompt_parts.append(style)

        # Join with commas and clean up
        prompt = ", ".join(prompt_parts)
        prompt = prompt.replace("_", " ")  # Convert underscores to spaces

        return prompt

    def unload(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        logger.info("WD14 tagger unloaded")


# Example usage and testing
if __name__ == "__main__":
    # Initialize tagger
    tagger = WD14Tagger(threshold=0.3)

    # Load test image
    test_image = Image.open("test_anime_character.jpg")

    # Basic tagging
    tags = tagger.tag_image(test_image)
    print(f"Generated tags: {tags}")

    # Character-specific tags
    char_tags = tagger.get_character_tags(test_image)
    print("\nCharacter tags by category:")
    for category, tag_list in char_tags.items():
        if tag_list:
            print(f"  {category}: {tag_list}")

    # Generate prompt
    prompt = tagger.generate_prompt_tags(test_image)
    print(f"\nGenerated prompt: {prompt}")

    # Test consistency
    expected = ["blue_hair", "school_uniform", "smile", "anime"]
    consistency = tagger.compare_consistency(test_image, expected)
    print(f"\nConsistency metrics: {consistency}")

    # Filter NSFW
    all_tags_with_scores = tagger.tag_image(test_image, return_scores=True)
    all_tags = [tag for tag, _ in all_tags_with_scores]
    safe_tags, nsfw_tags = tagger.filter_nsfw_tags(all_tags)
    print(f"\nSafe tags: {len(safe_tags)}, NSFW tags: {len(nsfw_tags)}")
