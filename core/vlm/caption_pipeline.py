# core/vlm/caption_pipeline.py
"""
Image Caption Generation Pipeline
"""

import torch
import logging
from PIL import Image
from typing import Union, Dict, Any

from ..exceptions import VLMError, ImageProcessingError

logger = logging.getLogger(__name__)


class CaptionPipeline:
    """Image caption generation pipeline"""

    def __init__(self, model_manager, image_processor):
        self.model_manager = model_manager
        self.image_processor = image_processor

    def generate_caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 0.7,
        do_sample: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate caption for image"""

        # Ensure caption model is loaded
        if not self.model_manager._caption_loaded:
            self.model_manager.load_caption_model()

        try:
            # Process image
            pil_image = self.image_processor.process_image(image)

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

            # Generate caption
            with torch.no_grad():
                outputs = caption_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=caption_processor.tokenizer.eos_token_id,
                    **kwargs,
                )

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

            return {
                "caption": caption,
                "confidence": self._estimate_confidence(outputs, inputs),
                "model_used": caption_model.config.name_or_path,
                "parameters": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "do_sample": do_sample,
                },
                "image_info": {
                    "size": pil_image.size,
                    "mode": pil_image.mode,
                },
            }

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise VLMError(f"Caption generation failed: {str(e)}")

    def _clean_caption(self, caption: str) -> str:
        """Clean and format caption text"""
        # Remove common prefixes
        prefixes_to_remove = [
            "a picture of",
            "an image of",
            "this is",
            "there is",
            "the image shows",
        ]

        caption_lower = caption.lower()
        for prefix in prefixes_to_remove:
            if caption_lower.startswith(prefix):
                caption = caption[len(prefix) :].strip()
                break

        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]

        # Ensure proper ending
        if caption and not caption.endswith((".", "!", "?")):
            caption += "."

        return caption

    def _estimate_confidence(self, outputs, inputs) -> float:
        """Estimate confidence score (placeholder implementation)"""
        # This is a simplified confidence estimation
        # In practice, you might use model attention weights or probability scores
        try:
            # Get sequence length as a proxy for confidence
            seq_length = outputs[0].shape[0]
            base_confidence = min(0.95, 0.5 + (seq_length / 100))
            return round(base_confidence, 2)
        except:
            return 0.85  # Default confidence

    def generate_batch_captions(
        self, images: list, max_length: int = 50, num_beams: int = 3, **kwargs
    ) -> list[Dict[str, Any]]:
        """Generate captions for multiple images"""
        results = []

        for i, image in enumerate(images):
            try:
                result = self.generate_caption(
                    image=image, max_length=max_length, num_beams=num_beams, **kwargs
                )
                result["batch_index"] = i
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to caption image {i}: {e}")
                results.append(
                    {
                        "caption": f"Error processing image {i+1}",
                        "confidence": 0.0,
                        "error": str(e),
                        "batch_index": i,
                    }
                )

        return results
