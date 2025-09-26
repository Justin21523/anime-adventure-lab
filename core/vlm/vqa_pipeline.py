# core/vlm/vqa_pipeline.py
"""
Visual Question Answering Pipeline
"""

import torch
import logging
import re
from PIL import Image
from typing import Union, Dict, Any, List

from ..exceptions import VLMError, ImageProcessingError

logger = logging.getLogger(__name__)


class VQAPipeline:
    """Visual Question Answering pipeline"""

    def __init__(self, model_manager, image_processor):
        self.model_manager = model_manager
        self.image_processor = image_processor

        # Predefined question templates for common queries
        self.question_templates = {
            "describe": "請詳細描述這張圖片的內容。",
            "objects": "這張圖片中有哪些物體？",
            "people": "圖片中有多少人？他們在做什麼？",
            "colors": "圖片中主要的顏色是什麼？",
            "location": "這張圖片是在什麼地方拍攝的？",
            "emotion": "圖片中的人物表情如何？",
            "action": "圖片中正在發生什麼動作？",
        }

    def answer_question(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = False,
        num_beams: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Answer question about image"""

        # Ensure VQA model is loaded
        if not self.model_manager._vqa_loaded:
            self.model_manager.load_vqa_model()

        try:
            # Process image
            pil_image = self.image_processor.process_image(image)

            # Get image info
            img_info = self.image_processor.get_image_info(pil_image)

            # Process question for Traditional Chinese support
            processed_question = self._process_question(question)

            # Get models
            _, _, vqa_model, vqa_processor = self.model_manager.get_models()

            if vqa_model is None or vqa_processor is None:
                raise VLMError("VQA model not loaded")

            # Prepare inputs based on model type
            device = next(vqa_model.parameters()).device

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
                # For BLIP-2 style models or other processors
                inputs = vqa_processor(
                    images=pil_image, text=processed_question, return_tensors="pt"
                )

            inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            # Generation parameters
            generation_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "pad_token_id": vqa_processor.tokenizer.eos_token_id,
                "early_stopping": True,
                **kwargs,
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

            # Estimate confidence
            confidence = self._estimate_confidence(answer, question, outputs[0])

            # Check for safety concerns
            safety_check = self._safety_check(question, answer)

            return {
                "question": processed_question,
                "answer": answer,
                "confidence": confidence,
                "model_used": vqa_model.config.name_or_path,
                "image_info": img_info,
                "safety_check": safety_check,
                "generation_params": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "num_beams": num_beams,
                    "model_used": getattr(vqa_model, "name_or_path", "unknown"),
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

    def _get_basic_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get basic image information"""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": getattr(image, "format", "Unknown"),
            "aspect_ratio": round(image.width / image.height, 2),
        }

    def _process_question(self, question: str) -> str:
        """Process and normalize question"""
        question = question.strip()

        # Handle empty or very short questions
        if len(question) < 2:
            return "請描述這張圖片。"

        # Add question mark if missing
        if not question.endswith(("?", "？", ".", "。")):
            if any(
                word in question.lower()
                for word in [
                    "what",
                    "who",
                    "where",
                    "when",
                    "why",
                    "how",
                    "什麼",
                    "誰",
                    "哪裡",
                    "何時",
                    "為什麼",
                    "如何",
                ]
            ):
                question += "?"
            else:
                question += "。"

        return question

    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean and format VQA answer"""
        if not answer:
            return "無法回答此問題。"

        # Remove the question from the answer if it's repeated
        question_start = answer.lower().find(question.lower())
        if question_start >= 0:
            answer = answer[question_start + len(question) :].strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "answer:",
            "答案：",
            "回答：",
            "the answer is",
            "this is",
            "it is",
            "這是",
            "答案是",
        ]

        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix) :].strip()
                break

        # Clean up formatting
        answer = re.sub(r"\s+", " ", answer)
        answer = answer.strip()

        if answer:
            # Capitalize first letter for English
            if answer[0].isalpha() and answer[0].islower():
                answer = answer[0].upper() + answer[1:]

        return answer if answer else "無法提供明確答案。"

    def _estimate_confidence(
        self, answer: str, question: str, output_tokens: torch.Tensor
    ) -> float:
        """Estimate answer confidence"""
        try:
            confidence = 0.5

            # Length-based confidence
            if 5 <= len(answer) <= 200:
                confidence += 0.2
            elif len(answer) < 3:
                confidence -= 0.3

            # Check for uncertainty markers
            uncertainty_markers = [
                "不確定",
                "可能",
                "也許",
                "似乎",
                "大概",
                "probably",
                "maybe",
                "uncertain",
                "unsure",
                "might be",
            ]
            if any(marker in answer.lower() for marker in uncertainty_markers):
                confidence -= 0.2

            # Check for definitive answers
            definitive_markers = [
                "是",
                "有",
                "沒有",
                "確實",
                "明確",
                "clearly",
                "definitely",
                "yes",
                "no",
                "absolutely",
            ]
            if any(marker in answer.lower() for marker in definitive_markers):
                confidence += 0.1

            # Question type based confidence
            question_lower = question.lower()
            if any(word in question_lower for word in ["多少", "how many", "count"]):
                # Counting questions - check if answer contains numbers
                if re.search(r"\d+", answer):
                    confidence += 0.2

            # Check for repetition or generic responses
            generic_responses = [
                "i don't know",
                "我不知道",
                "無法確定",
                "不清楚",
                "cannot tell",
                "unable to determine",
            ]
            if any(generic in answer.lower() for generic in generic_responses):
                confidence -= 0.3

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    def _safety_check(self, question: str, answer: str) -> Dict[str, Any]:
        """Basic safety check for question and answer"""
        safety_result = {"is_safe": True, "concerns": [], "filtered": False}

        try:
            # Check for inappropriate content
            inappropriate_keywords = [
                "暴力",
                "血腥",
                "裸體",
                "色情",
                "violence",
                "blood",
                "nude",
                "nsfw",
                "explicit",
            ]

            combined_text = f"{question} {answer}".lower()

            for keyword in inappropriate_keywords:
                if keyword in combined_text:
                    safety_result["concerns"].append(
                        f"Potential inappropriate content: {keyword}"
                    )
                    safety_result["is_safe"] = False

            # Check for personal information requests
            personal_info_patterns = [
                r"身分證",
                r"電話",
                r"地址",
                r"email",
                r"phone",
                r"address",
                r"social security",
                r"credit card",
            ]

            for pattern in personal_info_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    safety_result["concerns"].append("Personal information detected")
                    safety_result["is_safe"] = False

        except Exception as e:
            logger.warning(f"Safety check failed: {e}")

        return safety_result

    def batch_vqa(
        self,
        images: List[Union[str, bytes, Image.Image]],
        questions: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Process multiple image-question pairs"""
        if len(images) != len(questions):
            raise ValueError("Number of images must match number of questions")

        results = []

        for i, (image, question) in enumerate(zip(images, questions)):
            try:
                result = self.answer_question(image, question, **kwargs)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "question": question,
                        "answer": "處理失敗",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )

        return results

    def ask_predefined_question(
        self, image: Union[str, bytes, Image.Image], question_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Ask a predefined question type"""
        if question_type not in self.question_templates:
            available_types = list(self.question_templates.keys())
            raise ValueError(f"Unknown question type. Available: {available_types}")

        question = self.question_templates[question_type]
        return self.answer_question(image, question, **kwargs)

    def multi_aspect_analysis(
        self, image: Union[str, bytes, Image.Image], aspects: List[str] = None, **kwargs  # type: ignore
    ) -> Dict[str, Any]:
        """Analyze image from multiple aspects"""
        if aspects is None:
            aspects = ["describe", "objects", "colors", "action"]

        results = {}

        for aspect in aspects:
            if aspect in self.question_templates:
                try:
                    result = self.ask_predefined_question(image, aspect, **kwargs)
                    results[aspect] = {
                        "answer": result["answer"],
                        "confidence": result["confidence"],
                    }
                except Exception as e:
                    logger.error(f"Failed to analyze aspect '{aspect}': {e}")
                    results[aspect] = {
                        "answer": "分析失敗",
                        "confidence": 0.0,
                        "error": str(e),
                    }
            else:
                results[aspect] = {
                    "answer": f"未知的分析類型: {aspect}",
                    "confidence": 0.0,
                }

        return {
            "multi_aspect_results": results,
            "overall_analysis": self._combine_aspects(results),
        }

    def _combine_aspects(self, aspect_results: Dict[str, Dict]) -> str:
        """Combine multiple aspect analyses into a comprehensive description"""
        try:
            valid_results = {
                k: v
                for k, v in aspect_results.items()
                if v.get("confidence", 0) > 0.3 and "error" not in v
            }

            if not valid_results:
                return "無法提供綜合分析。"

            combined = []

            # Prioritize description first
            if "describe" in valid_results:
                combined.append(valid_results["describe"]["answer"])

            # Add specific details
            for aspect in ["objects", "colors", "action", "people"]:
                if aspect in valid_results and aspect != "describe":
                    answer = valid_results[aspect]["answer"]
                    if answer and len(answer) > 5:
                        combined.append(answer)

            return " ".join(combined) if combined else "綜合分析不可用。"

        except Exception as e:
            logger.error(f"Failed to combine aspects: {e}")
            return "綜合分析處理失敗。"

    def interactive_qa(
        self,
        image: Union[str, bytes, Image.Image],
        conversation_history: List[Dict[str, str]] = None,  # type: ignore
        current_question: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Interactive Q&A with conversation context"""
        if conversation_history is None:
            conversation_history = []

        # Build context from conversation history
        context_questions = []
        for turn in conversation_history[-3:]:  # Keep last 3 turns for context
            if "question" in turn:
                context_questions.append(turn["question"])

        # Enhance current question with context if needed
        enhanced_question = self._enhance_question_with_context(
            current_question, context_questions
        )

        # Get answer
        result = self.answer_question(image, enhanced_question, **kwargs)

        # Update conversation history
        new_turn = {
            "question": current_question,
            "enhanced_question": enhanced_question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "timestamp": (
                torch.cuda.current_device() if torch.cuda.is_available() else 0
            ),
        }

        conversation_history.append(new_turn)

        result["conversation_history"] = conversation_history
        result["context_used"] = len(context_questions) > 0

        return result

    def _enhance_question_with_context(
        self, question: str, context_questions: List[str]
    ) -> str:
        """Enhance question with conversational context"""
        if not context_questions:
            return question

        # Simple context enhancement - could be more sophisticated
        context_indicators = [
            "它",
            "他們",
            "這個",
            "那個",
            "this",
            "that",
            "it",
            "they",
        ]

        if any(indicator in question for indicator in context_indicators):
            # Add context hint
            context_summary = f"(參考之前的問題: {'; '.join(context_questions[-2:])})"
            return f"{question} {context_summary}"

        return question

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
