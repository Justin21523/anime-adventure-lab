# core/vlm/engine.py
"""
Vision-Language Model Engine
Handles caption generation and visual question answering
"""

import base64
import io
import torch
import logging
from PIL import Image
from typing import Union, Optional, Dict, Any, List

from ..exceptions import (
    VLMError,
    ImageProcessingError,
    ModelLoadError,
    handle_cuda_oom,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache
from ..utils.image import ImageProcessor
from .model_manager import VLMModelManager
from .caption_pipeline import CaptionPipeline
from .vqa_pipeline import VQAPipeline
from .processors import VLMImageProcessor, VLMTextProcessor

logger = logging.getLogger(__name__)


class VLMEngine:
    """Vision-Language Model engine for caption and VQA"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.image_processor = ImageProcessor()

        # Core processors
        self.image_processor = ImageProcessor()
        self.vlm_image_processor = VLMImageProcessor(self.config)
        self.text_processor = VLMTextProcessor(self.config)

        # Initialize managers and pipelines
        self.model_manager = VLMModelManager(self.config, self.cache)
        self.caption_pipeline = CaptionPipeline(
            self.model_manager, self.image_processor
        )
        self.vqa_pipeline = VQAPipeline(self.model_manager, self.image_processor)

    def load_caption_model(self, model_name: Optional[str] = None) -> None:
        """Load image captioning model"""
        self.model_manager.load_caption_model(model_name)

    def load_vqa_model(self, model_name: Optional[str] = None) -> None:
        """Load VQA model"""
        self.model_manager.load_vqa_model(model_name)

    @handle_cuda_oom
    @handle_model_error
    def caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
        enhance_image: bool = True,
        analyze_quality: bool = False,
        auto_correct: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image caption with advanced preprocessing options"""
        try:
            # Load base image
            pil_image = self.image_processor.load_image(image)

            # Advanced preprocessing if requested
            if enhance_image:
                pil_image = self.vlm_image_processor.preprocess_for_caption(pil_image)

            # Quality analysis and auto-correction
            quality_info = None
            corrections_applied = []

            if analyze_quality or auto_correct:
                quality_info = self.vlm_image_processor.detect_image_quality(pil_image)

                if auto_correct and not quality_info["is_high_quality"]:
                    pil_image, corrections_applied = (
                        self.vlm_image_processor.auto_correct_image(pil_image)
                    )
                    quality_info["auto_corrections"] = corrections_applied

            # Generate caption using pipeline
            result = self.caption_pipeline.generate_caption(
                image=pil_image, max_length=max_length, num_beams=num_beams, **kwargs
            )

            # Add advanced analysis results
            if quality_info:
                result["image_quality"] = quality_info
            if corrections_applied:
                result["corrections_applied"] = corrections_applied

            return result

        except Exception as e:
            logger.error(f"Enhanced caption generation failed: {e}")
            raise VLMError(f"Failed to generate caption: {str(e)}")

    @handle_cuda_oom
    @handle_model_error
    def vqa(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
        enhance_image: bool = True,
        process_question: bool = True,
        analyze_quality: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Answer question about image with enhanced processing"""
        try:
            # Load base image
            pil_image = self.image_processor.load_image(image)

            # Advanced image preprocessing
            if enhance_image:
                pil_image = self.vlm_image_processor.preprocess_for_vqa(pil_image)

            # Quality analysis if requested
            quality_info = None
            if analyze_quality:
                quality_info = self.vlm_image_processor.detect_image_quality(pil_image)

            # Advanced question processing
            question_info = None
            processed_question = question
            if process_question:
                question_info = self.text_processor.preprocess_question(question)
                processed_question = question_info["processed_question"]

                # Validate question quality
                if not question_info["is_valid"]:
                    return {
                        "answer": "問題品質不佳，請重新提問",
                        "question": processed_question,
                        "confidence": 0.0,
                        "question_analysis": question_info,
                        "error": "Invalid question quality",
                    }

            # Generate answer using pipeline
            result = self.vqa_pipeline.answer_question(
                image=pil_image,
                question=processed_question,
                max_length=max_length,
                **kwargs,
            )

            # Advanced response postprocessing
            if process_question and question_info:
                response_info = self.text_processor.postprocess_response(
                    result["answer"], question_info
                )
                result.update(
                    {
                        "question_analysis": question_info,
                        "response_analysis": response_info,
                        "processed_answer": response_info["processed_response"],
                        "response_confidence": response_info["confidence"],
                    }
                )

            # Add quality analysis if available
            if quality_info:
                result["image_quality"] = quality_info

            return result

        except Exception as e:
            logger.error(f"Enhanced VQA failed: {e}")
            raise VLMError(f"Failed to answer question: {str(e)}")

    def batch_caption(
        self, images: List[Union[str, bytes, Image.Image]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate captions for multiple images with advanced processing"""
        results = []

        for i, image in enumerate(images):
            try:
                result = self.caption(image, **kwargs)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i} in batch: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "caption": "批次處理失敗",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )

        return results

    def batch_vqa(
        self,
        images: List[Union[str, bytes, Image.Image]],
        questions: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Process multiple image-question pairs with advanced processing"""
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")

        results = []

        for i, (image, question) in enumerate(zip(images, questions)):
            try:
                result = self.vqa(image, question, **kwargs)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                results.append(
                    {
                        "batch_index": i,
                        "question": question,
                        "answer": "批次處理失敗",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )

        return results

    def comprehensive_image_analysis(
        self,
        image: Union[str, bytes, Image.Image],
        include_caption: bool = True,
        include_vqa: bool = True,
        include_quality: bool = True,
        include_composition: bool = True,
        include_faces: bool = False,
        vqa_aspects: List[str] = None,  # type: ignore
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform comprehensive image analysis using all available tools"""
        try:
            # Load and preprocess image
            pil_image = self.image_processor.load_image(image)

            analysis_result = {
                "analysis_timestamp": (
                    torch.cuda.current_device() if torch.cuda.is_available() else 0
                ),
                "image_info": {
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "mode": pil_image.mode,
                    "aspect_ratio": round(pil_image.width / pil_image.height, 2),
                },
            }

            # Quality analysis
            if include_quality:
                try:
                    quality_info = self.vlm_image_processor.detect_image_quality(
                        pil_image
                    )
                    analysis_result["quality_analysis"] = quality_info
                except Exception as e:
                    logger.warning(f"Quality analysis failed: {e}")
                    analysis_result["quality_analysis"] = {"error": str(e)}

            # Composition analysis
            if include_composition:
                try:
                    composition_info = self.vlm_image_processor.analyze_composition(
                        pil_image
                    )
                    analysis_result["composition_analysis"] = composition_info
                except Exception as e:
                    logger.warning(f"Composition analysis failed: {e}")
                    analysis_result["composition_analysis"] = {"error": str(e)}

            # Face detection
            if include_faces:
                try:
                    faces_info = self.vlm_image_processor.detect_faces(pil_image)
                    analysis_result["face_analysis"] = {
                        "faces_detected": len(faces_info),
                        "face_details": faces_info,
                    }
                except Exception as e:
                    logger.warning(f"Face analysis failed: {e}")
                    analysis_result["face_analysis"] = {"error": str(e)}

            # Caption analysis
            if include_caption:
                try:
                    caption_result = self.caption_pipeline.analyze_image_content(
                        pil_image, detailed=True
                    )
                    analysis_result["caption_analysis"] = caption_result
                except Exception as e:
                    logger.warning(f"Caption analysis failed: {e}")
                    analysis_result["caption_analysis"] = {"error": str(e)}

            # VQA multi-aspect analysis
            if include_vqa:
                try:
                    if vqa_aspects is None:
                        vqa_aspects = [
                            "describe",
                            "objects",
                            "colors",
                            "action",
                            "people",
                        ]

                    vqa_result = self.vqa_pipeline.multi_aspect_analysis(
                        pil_image, vqa_aspects, **kwargs
                    )
                    analysis_result["vqa_analysis"] = vqa_result
                except Exception as e:
                    logger.warning(f"VQA analysis failed: {e}")
                    analysis_result["vqa_analysis"] = {"error": str(e)}

            # Generate comprehensive summary
            analysis_result["comprehensive_summary"] = (
                self._generate_comprehensive_summary(analysis_result)
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Comprehensive image analysis failed: {e}")
            raise VLMError(f"Failed to analyze image comprehensively: {str(e)}")

    def _generate_comprehensive_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive summary from all analysis results"""
        try:
            summary_parts = []

            # Image basic info
            img_info = analysis_result.get("image_info", {})
            if img_info:
                summary_parts.append(
                    f"圖片尺寸: {img_info.get('width', 0)}x{img_info.get('height', 0)}, "
                    f"縱橫比: {img_info.get('aspect_ratio', 'N/A')}"
                )

            # Quality summary
            quality = analysis_result.get("quality_analysis", {})
            if quality and "error" not in quality:
                quality_score = quality.get("quality_score", 0)
                quality_status = (
                    "高品質" if quality.get("is_high_quality", False) else "中等品質"
                )
                summary_parts.append(
                    f"圖片品質: {quality_status} (分數: {quality_score:.2f})"
                )

            # Face detection summary
            face_analysis = analysis_result.get("face_analysis", {})
            if face_analysis and "error" not in face_analysis:
                face_count = face_analysis.get("faces_detected", 0)
                if face_count > 0:
                    summary_parts.append(f"檢測到 {face_count} 張人臉")

            # Caption summary
            caption_analysis = analysis_result.get("caption_analysis", {})
            if caption_analysis and "error" not in caption_analysis:
                basic_caption = caption_analysis.get("basic_caption", "")
                if basic_caption:
                    summary_parts.append(f"基本描述: {basic_caption}")

            # VQA summary
            vqa_analysis = analysis_result.get("vqa_analysis", {})
            if vqa_analysis and "error" not in vqa_analysis:
                overall_analysis = vqa_analysis.get("overall_analysis", "")
                if overall_analysis:
                    summary_parts.append(f"詳細分析: {overall_analysis}")

            # Composition summary
            composition = analysis_result.get("composition_analysis", {})
            if composition and "error" not in composition:
                comp_score = composition.get("composition_score", 0)
                if comp_score > 0.7:
                    summary_parts.append("構圖良好")
                elif comp_score < 0.4:
                    summary_parts.append("構圖需要改善")

            return " | ".join(summary_parts) if summary_parts else "綜合分析完成"

        except Exception as e:
            logger.error(f"Failed to generate comprehensive summary: {e}")
            return "綜合分析總結生成失敗"

    def _extract_session_topics(
        self, conversation_history: List[Dict[str, str]]
    ) -> List[str]:
        """Extract main topics from conversation history"""
        try:
            all_questions = " ".join(
                [turn.get("question", "") for turn in conversation_history]
            )
            keywords = self.text_processor.extract_keywords(
                all_questions, max_keywords=5
            )
            return keywords
        except Exception:
            return []

    def _generate_session_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate a summary of the interactive session"""
        try:
            history = session_data.get("conversation_history", [])
            topics = self._extract_session_topics(history)

            summary = f"會話包含 {len(history)} 個問答"
            if topics:
                summary += f"，主要討論了: {', '.join(topics)}"

            return summary
        except Exception:
            return "會話總結生成失敗"

    def apply_privacy_protection(
        self,
        image: Union[str, bytes, Image.Image],
        blur_faces: bool = True,
        blur_radius: int = 15,
        detect_text: bool = False,
        blur_text: bool = False,
    ) -> Dict[str, Any]:
        """Apply privacy protection measures to image"""
        try:
            pil_image = self.image_processor.load_image(image)
            protected_image = pil_image.copy()
            protection_applied = []

            # Face blurring
            if blur_faces:
                protected_image, face_count = self.vlm_image_processor.apply_face_blur(
                    protected_image, blur_radius=blur_radius
                )
                if face_count > 0:
                    protection_applied.append(f"已模糊 {face_count} 張人臉")

            # Could add text detection and blurring here if needed
            if detect_text:
                protection_applied.append("文字檢測功能尚未實現")

            return {
                "protected_image": protected_image,
                "original_image": pil_image,
                "protection_applied": protection_applied,
                "privacy_score": len(protection_applied) / 2.0,  # Simple metric
            }

        except Exception as e:
            logger.error(f"Privacy protection failed: {e}")
            raise VLMError(f"Failed to apply privacy protection: {str(e)}")

    def analyze_image(
        self,
        image: Union[str, bytes, Image.Image],
        analysis_type: str = "comprehensive",
        **kwargs,
    ) -> Dict[str, Any]:
        """Comprehensive image analysis"""
        try:
            # Load base image
            pil_image = self.image_processor.load_image(image)

            result = {
                "analysis_type": analysis_type,
                "timestamp": (
                    torch.cuda.current_device() if torch.cuda.is_available() else 0
                ),
            }

            # Quality analysis
            quality_info = self.vlm_image_processor.detect_image_quality(pil_image)
            result["quality_analysis"] = quality_info

            if analysis_type in ["comprehensive", "caption"]:
                # Basic captioning
                caption_result = self.caption_pipeline.analyze_image_content(
                    pil_image, detailed=(analysis_type == "comprehensive")
                )
                result["caption_analysis"] = caption_result

            if analysis_type in ["comprehensive", "vqa"]:
                # Multi-aspect VQA analysis
                aspects = ["describe", "objects", "colors", "action"]
                vqa_result = self.vqa_pipeline.multi_aspect_analysis(
                    pil_image, aspects, **kwargs
                )
                result["vqa_analysis"] = vqa_result

            return result

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise VLMError(f"Failed to analyze image: {str(e)}")

    def interactive_session(
        self,
        image: Union[str, bytes, Image.Image],
        conversation_history: List[Dict[str, str]] = None,  # type: ignore
        session_config: Dict[str, Any] = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Start or continue an interactive VQA session with advanced features"""
        if conversation_history is None:
            conversation_history = []

        if session_config is None:
            session_config = {
                "enable_quality_analysis": True,
                "enable_composition_analysis": False,
                "enable_face_detection": False,
                "auto_enhance": True,
            }

        try:
            # Process image once for the session
            pil_image = self.image_processor.load_image(image)

            # Perform initial analysis based on session config
            initial_analysis = {}

            if session_config.get("enable_quality_analysis", False):
                quality_info = self.vlm_image_processor.detect_image_quality(pil_image)
                initial_analysis["image_quality"] = quality_info

            if session_config.get("enable_composition_analysis", False):
                composition_info = self.vlm_image_processor.analyze_composition(
                    pil_image
                )
                initial_analysis["composition"] = composition_info

            if session_config.get("enable_face_detection", False):
                faces_info = self.vlm_image_processor.detect_faces(pil_image)
                initial_analysis["faces"] = faces_info

            # Auto-enhance image if requested
            enhanced_image = pil_image
            if session_config.get("auto_enhance", False):
                enhanced_image = self.vlm_image_processor.preprocess_for_vqa(pil_image)

            session_id = f"vlm_session_{hash(str(pil_image.tobytes()[:100]))}"

            return {
                "session_id": session_id,
                "image_processed": True,
                "conversation_history": conversation_history,
                "session_config": session_config,
                "initial_analysis": initial_analysis,
                "available_commands": [
                    "ask <question>",
                    "describe",
                    "analyze",
                    "quality_check",
                    "faces",
                    "composition",
                    "enhance",
                    "summary",
                    "end",
                ],
                "session_ready": True,
                "image": enhanced_image,  # Store processed image
                "original_image": pil_image,  # Keep original for comparison
            }

        except Exception as e:
            logger.error(f"Failed to start interactive session: {e}")
            return {"session_ready": False, "error": str(e)}

    def process_interactive_command(
        self, session_data: Dict[str, Any], command: str, **kwargs
    ) -> Dict[str, Any]:
        """Process interactive session command with advanced capabilities"""
        try:
            if not session_data.get("session_ready"):
                raise VLMError("Session not ready")

            command = command.strip().lower()
            image = session_data["image"]
            original_image = session_data.get("original_image", image)

            if command.startswith("ask "):
                question = command[4:].strip()
                result = self.vqa_pipeline.interactive_qa(
                    image,
                    session_data.get("conversation_history", []),
                    question,
                    **kwargs,
                )
                return result

            elif command == "describe":
                return self.caption_pipeline.analyze_image_content(image, detailed=True)

            elif command == "analyze":
                return self.comprehensive_image_analysis(image)

            elif command == "quality_check":
                return self.vlm_image_processor.detect_image_quality(original_image)

            elif command == "faces":
                faces = self.vlm_image_processor.detect_faces(original_image)
                return {
                    "faces_detected": len(faces),
                    "face_details": faces,
                    "message": (
                        f"檢測到 {len(faces)} 張人臉" if faces else "未檢測到人臉"
                    ),
                }

            elif command == "composition":
                return self.vlm_image_processor.analyze_composition(original_image)

            elif command == "enhance":
                enhanced = self.vlm_image_processor.preprocess_for_vqa(original_image)
                corrected, corrections = self.vlm_image_processor.auto_correct_image(
                    original_image
                )
                return {
                    "message": "圖片已增強處理",
                    "corrections_applied": corrections,
                    "enhanced_image_ready": True,
                }

            elif command == "summary":
                # Generate session summary
                history = session_data.get("conversation_history", [])
                return {
                    "session_summary": {
                        "total_questions": len(history),
                        "session_duration": "N/A",  # Could track time if needed
                        "topics_discussed": self._extract_session_topics(history),
                        "image_info": session_data.get("initial_analysis", {}),
                    }
                }

            elif command == "end":
                return {
                    "session_ended": True,
                    "message": "互動會話已結束",
                    "final_summary": self._generate_session_summary(session_data),
                }

            else:
                return {
                    "error": f"未知指令: {command}",
                    "available_commands": [
                        "ask <question>",
                        "describe",
                        "analyze",
                        "quality_check",
                        "faces",
                        "composition",
                        "enhance",
                        "summary",
                        "end",
                    ],
                    "help": "使用 'ask <問題>' 來提問，或使用其他指令進行分析",
                }

        except Exception as e:
            logger.error(f"Interactive command failed: {e}")
            return {"error": str(e)}

    def get_supported_features(self) -> Dict[str, Any]:
        """Get comprehensive list of supported features and capabilities"""
        return {
            "captioning": {
                "available": True,
                "models": ["blip2", "blip", "git"],
                "languages": ["en", "zh-TW", "zh-CN"],
                "features": [
                    "batch_processing",
                    "quality_analysis",
                    "auto_enhancement",
                    "alternative_generation",
                    "detailed_analysis",
                ],
            },
            "vqa": {
                "available": True,
                "models": ["llava", "qwen-vl", "blip2"],
                "languages": ["en", "zh-TW", "zh-CN", "ja", "ko"],
                "features": [
                    "interactive_qa",
                    "multi_aspect_analysis",
                    "conversation_context",
                    "safety_filtering",
                    "question_preprocessing",
                    "response_analysis",
                ],
            },
            "image_processing": {
                "formats": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"],
                "max_size": getattr(self.config, "max_image_size", 1024),
                "features": [
                    "auto_resize",
                    "quality_detection",
                    "auto_correction",
                    "blur_detection",
                    "brightness_adjustment",
                    "composition_analysis",
                    "face_detection",
                    "privacy_protection",
                ],
            },
            "text_processing": {
                "languages": ["chinese", "english", "mixed"],
                "features": [
                    "language_detection",
                    "question_enhancement",
                    "response_postprocessing",
                    "quality_assessment",
                    "keyword_extraction",
                    "sentiment_analysis",
                ],
            },
            "advanced_features": {
                "comprehensive_analysis": True,
                "interactive_sessions": True,
                "privacy_protection": True,
                "batch_processing": True,
                "quality_assessment": True,
                "auto_enhancement": True,
            },
            "performance": {
                "gpu_acceleration": torch.cuda.is_available(),
                "quantization": True,
                "memory_optimization": True,
                "caching": True,
            },
        }

    def unload_models(self) -> None:
        """Unload all VLM models to free memory"""
        self.model_manager.unload_all_models()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive VLM engine status"""
        base_status = self.model_manager.get_status()

        base_status.update(
            {
                "engine_ready": True,
                "supported_features": self.get_supported_features(),
                "processor_status": {
                    "basic_image_processor": "ready",
                    "vlm_image_processor": "ready",
                    "vlm_text_processor": "ready",
                },
                "advanced_capabilities": {
                    "quality_analysis": True,
                    "composition_analysis": True,
                    "face_detection": self.vlm_image_processor.face_cascade is not None,
                    "privacy_protection": True,
                    "interactive_sessions": True,
                },
            }
        )

        return base_status


# Global VLM engine instance
_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine() -> VLMEngine:
    """Get global VLM engine instance"""
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = VLMEngine()
    return _vlm_engine


def reset_vlm_engine() -> None:
    """Reset global VLM engine instance"""
    global _vlm_engine
    if _vlm_engine is not None:
        _vlm_engine.unload_models()
        _vlm_engine = None
