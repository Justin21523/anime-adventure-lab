# core/vlm/caption_pipeline.py
"""
Image Caption Generation Pipeline
"""

import torch
import logging
import re
from PIL import Image
from typing import Union, Dict, Any, List

from ..exceptions import VLMError, ImageProcessingError

logger = logging.getLogger(__name__)


class CaptionPipeline:
    """Image caption generation pipeline"""

    def __init__(self, model_manager, image_processor):
        self.model_manager = model_manager
        self.image_processor = image_processor

        # Caption cleaning patterns
        self.cleanup_patterns = [
            (r"^(a|an|the)\s+", ""),  # Remove leading articles
            (r"\s+", " "),  # Multiple spaces to single
            (r"[.]{2,}", "."),  # Multiple dots to single
            (r"^[^a-zA-Z]*", ""),  # Remove leading non-letters
        ]

    def generate_caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 0.7,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        length_penalty: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate caption for image"""

        # Ensure caption model is loaded
        if not self.model_manager._caption_loaded:
            self.model_manager.load_caption_model()

        try:
            # Process image using the existing ImageProcessor
            pil_image = self.image_processor.load_image(image)

            # Get image info
            img_info = self._get_basic_image_info(pil_image)

            # Get models
            caption_model, caption_processor, _, _ = self.model_manager.get_models()

            if caption_model is None or caption_processor is None:
                raise VLMError("Caption model not loaded")

            # Prepare inputs
            inputs = caption_processor(images=pil_image, return_tensors="pt")

            # Move to appropriate device
            device = next(caption_model.parameters()).device
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            # Generation parameters
            generation_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "temperature": temperature,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "pad_token_id": caption_processor.tokenizer.eos_token_id,
                "early_stopping": True,
                **kwargs,
            }

            # Generate caption
            with torch.no_grad():
                outputs = caption_model.generate(**inputs, **generation_kwargs)

            # Decode caption
            if hasattr(caption_processor, "decode"):
                # For BLIP-2 and newer models
                caption = caption_processor.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()
            else:
                # For BLIP-1
                caption = caption_processor.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()

            # Clean up caption
            caption = self._clean_caption(caption)

            # Estimate confidence (simple heuristic)
            confidence = self._estimate_confidence(caption, outputs[0])

            # Generate alternative captions for diversity
            alternatives = []
            if num_beams > 1:
                alternatives = self._generate_alternatives(
                    caption_model, caption_processor, inputs, generation_kwargs
                )

            return {
                "caption": caption,
                "confidence": confidence,
                "alternatives": alternatives,
                "image_info": img_info,
                "generation_params": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "model_used": getattr(caption_model, "name_or_path", "unknown"),
                },
            }

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise VLMError(f"Failed to generate caption: {str(e)}")

    def _get_basic_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get basic image information"""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": getattr(image, "format", "Unknown"),
            "aspect_ratio": round(image.width / image.height, 2),
        }

    def _clean_caption(self, caption: str) -> str:
        """Clean and format caption"""
        if not caption:
            return "Unable to generate caption"

        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            caption = re.sub(pattern, replacement, caption)

        # Capitalize first letter
        caption = caption.strip()
        if caption:
            caption = caption[0].upper() + caption[1:]

        # Ensure ends with period
        if caption and not caption.endswith((".", "!", "?")):
            caption += "."

        return caption

    def _estimate_confidence(self, caption: str, output_tokens: torch.Tensor) -> float:
        """Estimate caption confidence using simple heuristics"""
        try:
            # Basic heuristics
            confidence = 0.5

            # Length-based confidence
            if 10 <= len(caption) <= 100:
                confidence += 0.2
            elif len(caption) < 5:
                confidence -= 0.3

            # Check for repetition
            words = caption.lower().split()
            if len(words) != len(set(words)):
                confidence -= 0.2

            # Check for completeness (ends properly)
            if caption.endswith((".", "!", "?")):
                confidence += 0.1

            # Token length penalty for very short/long sequences
            token_len = len(output_tokens)
            if 5 <= token_len <= 30:
                confidence += 0.1

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    def _generate_alternatives(
        self, model, processor, inputs, base_kwargs, num_alternatives: int = 2
    ) -> List[str]:
        """Generate alternative captions with different parameters"""
        alternatives = []

        try:
            # Try different sampling strategies
            alt_configs = [
                {"temperature": 0.9, "do_sample": True, "top_p": 0.9},
                {"num_beams": 5, "early_stopping": True},
            ]

            for config in alt_configs[:num_alternatives]:
                alt_kwargs = {**base_kwargs, **config}

                with torch.no_grad():
                    outputs = model.generate(**inputs, **alt_kwargs)

                if hasattr(processor, "decode"):
                    alt_caption = processor.decode(
                        outputs[0], skip_special_tokens=True
                    ).strip()
                else:
                    alt_caption = processor.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    ).strip()

                alt_caption = self._clean_caption(alt_caption)
                if alt_caption and alt_caption not in alternatives:
                    alternatives.append(alt_caption)

        except Exception as e:
            logger.warning(f"Failed to generate alternatives: {e}")

        return alternatives

    def generate_batch_captions(
        self, images: List[Union[str, bytes, Image.Image]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate captions for multiple images"""
        results = []

        for i, image in enumerate(images):
            try:
                result = self.generate_caption(image, **kwargs)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "caption": "Error processing image",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )

        return results

    def analyze_image_content(
        self, image: Union[str, bytes, Image.Image], detailed: bool = False
    ) -> Dict[str, Any]:
        """Analyze image content beyond basic captioning"""
        try:
            # Generate basic caption
            caption_result = self.generate_caption(image, max_length=100)

            analysis = {
                "basic_caption": caption_result["caption"],
                "confidence": caption_result["confidence"],
                "image_info": caption_result["image_info"],
            }

            if detailed:
                # Generate different types of descriptions
                descriptive_result = self.generate_caption(
                    image, max_length=150, temperature=0.3, num_beams=5  # More focused
                )

                creative_result = self.generate_caption(
                    image,
                    max_length=80,
                    temperature=0.8,  # More creative
                    do_sample=True,
                    top_p=0.9,
                )

                analysis.update(
                    {
                        "detailed_description": descriptive_result["caption"],
                        "creative_description": creative_result["caption"],
                        "alternatives": caption_result.get("alternatives", []),
                    }
                )

            return analysis

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "basic_caption": "Analysis failed",
                "confidence": 0.0,
                "error": str(e),
            }

    def analyze_style_and_content(
        self, image: Union[str, bytes, Image.Image]
    ) -> Dict[str, Any]:
        """Analyze image style and content with multiple approaches"""
        try:
            # Load image
            pil_image = self.image_processor.load_image(image)

            # Generate different style descriptions
            style_focused = self.generate_caption(
                pil_image, max_length=80, temperature=0.6, num_beams=4
            )

            # Generate content-focused description
            content_focused = self.generate_caption(
                pil_image, max_length=120, temperature=0.4, num_beams=6
            )

            # Generate artistic description
            artistic_focused = self.generate_caption(
                pil_image, max_length=100, temperature=0.9, do_sample=True, top_p=0.8
            )

            return {
                "style_description": style_focused["caption"],
                "content_description": content_focused["caption"],
                "artistic_description": artistic_focused["caption"],
                "overall_confidence": (
                    style_focused["confidence"]
                    + content_focused["confidence"]
                    + artistic_focused["confidence"]
                )
                / 3,
                "image_info": style_focused["image_info"],
            }

        except Exception as e:
            logger.error(f"Style and content analysis failed: {e}")
            return {
                "error": str(e),
                "style_description": "Analysis failed",
                "content_description": "Analysis failed",
                "artistic_description": "Analysis failed",
            }

    def compare_captions(
        self, images: List[Union[str, bytes, Image.Image]], **kwargs
    ) -> Dict[str, Any]:
        """Compare captions across multiple images to find similarities and differences"""
        try:
            if len(images) < 2:
                raise ValueError("Need at least 2 images for comparison")

            # Generate captions for all images
            captions_data = []
            for i, image in enumerate(images):
                result = self.generate_caption(image, **kwargs)
                captions_data.append(
                    {
                        "index": i,
                        "caption": result["caption"],
                        "confidence": result["confidence"],
                        "image_info": result["image_info"],
                    }
                )

            # Analyze similarities and differences
            all_captions = [data["caption"] for data in captions_data]

            # Extract common words
            all_words = []
            for caption in all_captions:
                words = re.findall(r"\b\w+\b", caption.lower())
                all_words.extend(words)

            # Find common themes
            from collections import Counter

            word_counts = Counter(all_words)
            common_words = [
                word for word, count in word_counts.most_common(10) if count > 1
            ]

            # Calculate caption similarities (simple word overlap)
            similarities = []
            for i, caption1 in enumerate(all_captions):
                for j, caption2 in enumerate(all_captions[i + 1 :], i + 1):
                    words1 = set(re.findall(r"\b\w+\b", caption1.lower()))
                    words2 = set(re.findall(r"\b\w+\b", caption2.lower()))

                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(
                            words1.union(words2)
                        )
                        similarities.append(
                            {
                                "image1_index": i,
                                "image2_index": j,
                                "similarity_score": similarity,
                                "common_words": list(words1.intersection(words2)),
                            }
                        )

            return {
                "individual_captions": captions_data,
                "common_themes": common_words,
                "caption_similarities": similarities,
                "summary": {
                    "total_images": len(images),
                    "avg_confidence": sum(data["confidence"] for data in captions_data)
                    / len(captions_data),
                    "most_similar_pair": (
                        max(similarities, key=lambda x: x["similarity_score"])
                        if similarities
                        else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Caption comparison failed: {e}")
            return {
                "error": str(e),
                "individual_captions": [],
                "common_themes": [],
                "caption_similarities": [],
            }

    def get_caption_variations(
        self,
        image: Union[str, bytes, Image.Image],
        num_variations: int = 5,
        max_length: int = 50,
    ) -> Dict[str, Any]:
        """Generate multiple caption variations with different parameters"""
        try:
            variations = []

            # Define different generation strategies
            strategies = [
                {"temperature": 0.3, "num_beams": 5, "name": "focused"},
                {"temperature": 0.7, "num_beams": 3, "name": "balanced"},
                {
                    "temperature": 0.9,
                    "do_sample": True,
                    "top_p": 0.8,
                    "name": "creative",
                },
                {
                    "temperature": 0.5,
                    "num_beams": 4,
                    "repetition_penalty": 1.2,
                    "name": "detailed",
                },
                {"temperature": 0.8, "do_sample": True, "top_k": 50, "name": "diverse"},
            ]

            # Generate variations
            for i, strategy in enumerate(strategies[:num_variations]):
                try:
                    strategy_params = {k: v for k, v in strategy.items() if k != "name"}
                    result = self.generate_caption(
                        image, max_length=max_length, **strategy_params
                    )

                    variations.append(
                        {
                            "variation_id": i + 1,
                            "strategy": strategy["name"],
                            "caption": result["caption"],
                            "confidence": result["confidence"],
                            "parameters": strategy_params,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to generate variation {i+1}: {e}")
                    variations.append(
                        {
                            "variation_id": i + 1,
                            "strategy": strategy["name"],
                            "caption": "Generation failed",
                            "confidence": 0.0,
                            "error": str(e),
                        }
                    )

            # Find best variation
            best_variation = max(variations, key=lambda x: x.get("confidence", 0))

            return {
                "variations": variations,
                "best_variation": best_variation,
                "total_variations": len(variations),
                "avg_confidence": sum(v.get("confidence", 0) for v in variations)
                / len(variations),
            }

        except Exception as e:
            logger.error(f"Caption variation generation failed: {e}")
            return {"error": str(e), "variations": [], "best_variation": None}
