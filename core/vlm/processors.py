# core/vlm/processors.py
"""
VLM Input/Output Processors and Utilities
"""

import re
import torch
import logging
from PIL import Image
from typing import Union, Dict, Any, List, Optional

logger = logging.getLogger(__name__)


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
