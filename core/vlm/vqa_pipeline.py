# core/vlm/vqa_pipeline.py
"""
Visual Question Answering Pipeline
"""

import torch
import logging
from PIL import Image
from typing import Union, Dict, Any

from ..exceptions import VLMError, ImageProcessingError

logger = logging.getLogger(__name__)


class VQAPipeline:
    """Visual Question Answering pipeline"""

    def __init__(self, model_manager, image_processor):
        self.model_manager = model_manager
        self.image_processor = image_processor

    def answer_question(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Answer question about image"""

        # Ensure VQA model is loaded
        if not self.model_manager._vqa_loaded:
            self.model_manager.load_vqa_model()

        try:
            # Process image
            pil_image = self.image_processor.process_image(image)

            # Process question for Traditional Chinese support
            processed_question = self._process_question(question)

            # Get models
            _, _, vqa_model, vqa_processor = self.model_manager.get_models()

            if vqa_model is None or vqa_processor is None:
                raise VLMError("VQA model not loaded")

            # Prepare inputs based on model type
            if hasattr(vqa_processor, "apply_chat_template"):
                # For newer LLaVA models
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": processed_question},
                        ],
                    }
                ]
                prompt = vqa_processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
                inputs = vqa_processor(prompt, images=pil_image, return_tensors="pt")
            else:
                # For BLIP-2 style models
                inputs = vqa_processor(
                    images=pil_image, text=processed_question, return_tensors="pt"
                )

            # Move to appropriate device
            device = next(vqa_model.parameters()).device
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            # Generate answer
            with torch.no_grad():
                outputs = vqa_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=(
                        vqa_processor.tokenizer.eos_token_id
                        if hasattr(vqa_processor, "tokenizer")
                        else None
                    ),
                    **kwargs,
                )

            # Decode answer
            if hasattr(vqa_processor, "decode"):
                if hasattr(inputs, "input_ids") and "input_ids" in inputs:
                    # Skip the input tokens for generation-only output
                    answer = vqa_processor.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).strip()
                else:
                    answer = vqa_processor.decode(
                        outputs[0], skip_special_tokens=True
                    ).strip()
            else:
                # Fallback
                answer = vqa_processor.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()

            # Clean answer
            answer = self._clean_answer(answer, question)

            return {
                "question": question,
                "answer": answer,
                "confidence": self._estimate_confidence(outputs, inputs),
                "model_used": vqa_model.config.name_or_path,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "do_sample": do_sample,
                },
                "image_info": {
                    "size": pil_image.size,
                    "mode": pil_image.mode,
                },
                "language_detected": self._detect_language(question),
            }

        except Exception as e:
            logger.error(f"VQA failed: {e}")
            raise VLMError(f"VQA failed: {str(e)}")

    def _process_question(self, question: str) -> str:
        """Process question for better Traditional Chinese support"""
        # Add language context for Traditional Chinese questions
        if self._is_traditional_chinese(question):
            # Add instruction for Traditional Chinese response
            return f"請用繁體中文回答：{question}"
        elif self._is_chinese(question):
            # General Chinese question
            return f"請用中文回答：{question}"

        return question

    def _clean_answer(self, answer: str, original_question: str) -> str:
        """Clean and format answer text"""
        # Remove repeated question text
        if original_question.lower() in answer.lower():
            answer = answer.replace(original_question, "").strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is",
            "answer:",
            "response:",
            "based on the image",
            "looking at the image",
        ]

        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix) :].strip()
                break

        # Clean up whitespace and formatting
        answer = " ".join(answer.split())

        return answer

    def _estimate_confidence(self, outputs, inputs) -> float:
        """Estimate confidence score"""
        try:
            # Simple heuristic based on answer length and coherence
            answer_length = outputs[0].shape[0]
            if answer_length < 5:
                return 0.6  # Very short answers might be less reliable
            elif answer_length > 50:
                return 0.8  # Longer, more detailed answers
            else:
                return 0.75  # Medium confidence for normal length
        except:
            return 0.75  # Default confidence

    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple character-based detection
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        total_chars = len([c for c in text if c.isalpha() or "\u4e00" <= c <= "\u9fff"])

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        else:
            return "en"

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _is_traditional_chinese(self, text: str) -> bool:
        """Simple heuristic to detect Traditional Chinese"""
        # Check for some common Traditional Chinese characters
        traditional_indicators = [
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
        ]

        return any(indicator in text for indicator in traditional_indicators)

    def answer_batch_questions(
        self, image_question_pairs: list, max_length: int = 100, **kwargs
    ) -> list[Dict[str, Any]]:
        """Answer multiple questions about images"""
        results = []

        for i, (image, question) in enumerate(image_question_pairs):
            try:
                result = self.answer_question(
                    image=image, question=question, max_length=max_length, **kwargs
                )
                result["batch_index"] = i
                results.append(result)

            except Exception as e:
                logger.error(f"Failed VQA for pair {i}: {e}")
                results.append(
                    {
                        "question": question,
                        "answer": f"Error processing question {i+1}",
                        "confidence": 0.0,
                        "error": str(e),
                        "batch_index": i,
                    }
                )

        return results
