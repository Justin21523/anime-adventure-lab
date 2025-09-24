# core/safety/watermark.py
import os
import json
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from core.config import get_config

logger = logging.getLogger(__name__)


class WatermarkGenerator:
    """Generate watermarks for images with attribution info"""

    def __init__(self):
        self.config = get_config()
        self.default_font_size = 12
        self.default_font = self._load_font()
        self.watermark_opacity = 128  # 50% transparency

        # Try to load a font
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.ImageFont:
        """Load watermark font"""
        try:
            # 可能的字體路徑
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        return ImageFont.truetype(font_path, self.default_font_size)  # type: ignore
                    except Exception:
                        continue

            # 備用方案：使用預設字體
            return ImageFont.load_default()  # type: ignore

        except Exception:
            # 最終備用方案
            return ImageFont.load_default()  # type: ignore

    def add_visible_watermark(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom_right",
        opacity: float = 0.5,
    ) -> Image.Image:
        """Add visible watermark to image"""
        try:
            # 建立副本避免修改原圖
            watermarked = image.copy()
            draw = ImageDraw.Draw(watermarked)

            # 計算文字位置
            text_bbox = draw.textbbox((0, 0), text, font=self.default_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            margin = 10
            if position == "bottom_right":
                x = watermarked.width - text_width - margin
                y = watermarked.height - text_height - margin
            elif position == "bottom_left":
                x = margin
                y = watermarked.height - text_height - margin
            else:  # top_right
                x = watermarked.width - text_width - margin
                y = margin

            # 繪製半透明背景
            overlay = Image.new("RGBA", watermarked.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [x - 5, y - 2, x + text_width + 5, y + text_height + 2],
                fill=(0, 0, 0, 128),
            )

            # 合併背景
            watermarked = Image.alpha_composite(
                watermarked.convert("RGBA"), overlay
            ).convert("RGB")

            # 繪製文字
            draw = ImageDraw.Draw(watermarked)
            draw.text((x, y), text, font=self.default_font, fill=(255, 255, 255))

            return watermarked

        except Exception as e:
            logger.warning(f"⚠️ Failed to add watermark: {e}")
            return image

    def add_invisible_watermark(
        self, image: Image.Image, metadata: Dict[str, Any]
    ) -> Image.Image:
        """Add invisible metadata watermark to image"""
        try:
            # Convert to RGB if necessary (PNG metadata preservation)
            if image.mode == "RGBA":
                # Create a white background and paste the image
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[-1] if len(image.split()) == 4 else None
                )
                image = background

            # 正確的方式：使用 PngInfo 處理 metadata
            if isinstance(metadata, dict):
                # 建立 PNG metadata
                png_info = PngInfo()

                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        png_info.add_text(f"SagaForge_{key}", str(value))
                    else:
                        png_info.add_text(f"SagaForge_{key}", str(value))

                # 將 metadata 附加到圖片
                # 注意：這裡不是直接修改 image.text，而是在保存時使用 pnginfo 參數
                # 我們將 metadata 存儲在圖片的 info 字典中
                if not hasattr(image, "info"):
                    image.info = {}

                # 更新圖片的 info 字典（這是正確的方式）
                image.info.update(
                    {f"SagaForge_{k}": str(v) for k, v in metadata.items()}
                )

            return image

        except Exception as e:
            logger.warning(f"⚠️ Failed to add invisible watermark: {e}")
            return image


class AttributionManager:
    """Manage attribution and licensing information for generated content"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.watermark_gen = WatermarkGenerator()

    def create_attribution_metadata(
        self,
        generation_params: Dict[str, Any],
        source_files: list = None,  # type: ignore
        model_info: Dict[str, Any] = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Create comprehensive attribution metadata for generated content"""

        from datetime import datetime, timezone

        metadata = {
            "generated_by": "SagaForge T2I System",
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "generation_params": generation_params,
            "attribution": {"sources": [], "models": [], "licenses": []},
        }

        # Add source file attributions
        if source_files:
            for source_file in source_files:
                if isinstance(source_file, dict):
                    metadata["attribution"]["sources"].append(source_file)
                else:
                    metadata["attribution"]["sources"].append(
                        {"file_id": str(source_file)}
                    )

        # Add model information
        if model_info:
            metadata["attribution"]["models"] = [model_info]

        return metadata

    def generate_attribution_text(self, metadata: Dict[str, Any]) -> str:
        """Generate human-readable attribution text"""

        parts = ["Generated by SagaForge"]

        # Add model info
        models = metadata.get("attribution", {}).get("models", [])
        if models:
            model_names = [m.get("name", "Unknown") for m in models]
            parts.append(f"Models: {', '.join(model_names)}")

        # Add source info
        sources = metadata.get("attribution", {}).get("sources", [])
        if sources:
            parts.append(f"Sources: {len(sources)} file(s)")

        # Add timestamp
        timestamp = metadata.get("generation_timestamp", "")
        if timestamp:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                parts.append(dt.strftime("%Y-%m-%d"))
            except:
                pass

        return " | ".join(parts)

    def process_generated_image(
        self,
        image: Image.Image,
        generation_params: Dict[str, Any],
        source_files: list = None,  # type: ignore
        model_info: Dict[str, Any] = None,  # type: ignore
        add_visible_watermark: bool = True,
        watermark_position: str = "bottom_right",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Process generated image with attribution and watermarks"""

        # Create attribution metadata
        metadata = self.create_attribution_metadata(
            generation_params, source_files, model_info
        )

        # Add invisible metadata watermark
        processed_image = self.watermark_gen.add_invisible_watermark(image, metadata)

        # Add visible watermark if requested
        if add_visible_watermark:
            attribution_text = self.generate_attribution_text(metadata)
            processed_image = self.watermark_gen.add_visible_watermark(
                processed_image,
                attribution_text,
                position=watermark_position,
                opacity=0.6,
            )

        return processed_image, metadata

    def save_with_attribution(
        self,
        image: Image.Image,
        output_path: Optional[Path],  # str
        metadata: Dict[str, Any],
        format: str = "PNG",
    ) -> str:
        """Save image with proper attribution metadata"""

        output_path = Path(output_path)  # type: ignore

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image with metadata
        if format.upper() == "PNG" and hasattr(image, "text"):
            # PNG with metadata
            from PIL.PngImagePlugin import PngInfo

            png_info = PngInfo()

            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    png_info.add_text(f"SagaForge_{key}", str(value))
                else:
                    png_info.add_text(f"SagaForge_{key}", str(value))

            image.save(str(output_path), format="PNG", pnginfo=png_info)
        else:
            # Fallback to regular save
            image.save(str(output_path), format=format)

        # Save metadata as sidecar JSON
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved image with attribution: {output_path}")
        return str(output_path)


class ComplianceLogger:
    """Log compliance-related activities for audit purposes"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.audit_log_path = self.cache_root / "logs" / "compliance_audit.jsonl"

        # Ensure log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_upload(
        self, file_id: str, metadata: Dict[str, Any], safety_result: Dict[str, Any]
    ):
        """Log file upload with safety and license checks"""

        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "file_upload",
            "file_id": file_id,
            "license_type": metadata.get("license_info", {}).get("license_type"),
            "safety_check": {
                "is_safe": safety_result.get("is_safe", False),
                "nsfw_detected": safety_result.get("nsfw_check", {}).get(
                    "is_nsfw", False
                ),
                "actions_taken": safety_result.get("actions_taken", []),
            },
            "uploader_id": metadata.get("uploader_id", "unknown"),
        }

        self._write_log_entry(log_entry)

    def log_generation(
        self,
        output_path: str,
        generation_params: Dict[str, Any],
        safety_result: Dict[str, Any],
    ):
        """Log content generation with safety checks"""

        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "content_generation",
            "output_path": str(output_path),
            "prompt": generation_params.get("prompt", ""),
            "model_used": generation_params.get("model_id", ""),
            "safety_check": safety_result,
            "has_watermark": generation_params.get("add_watermark", True),
        }

        self._write_log_entry(log_entry)

    def log_safety_violation(
        self, violation_type: str, content_info: Dict[str, Any], action_taken: str
    ):
        """Log safety policy violations"""

        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "safety_violation",
            "violation_type": violation_type,
            "content_info": content_info,
            "action_taken": action_taken,
            "severity": "high" if "nsfw" in violation_type.lower() else "medium",
        }

        self._write_log_entry(log_entry)

    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of compliance activities for the last N days"""

        from datetime import datetime, timezone, timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        summary = {
            "period_days": days,
            "total_events": 0,
            "uploads": 0,
            "generations": 0,
            "safety_violations": 0,
            "license_types": {},
            "safety_actions": {},
        }

        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"])

                        if entry_time >= cutoff_time:
                            summary["total_events"] += 1

                            event_type = entry.get("event_type", "")
                            if event_type == "file_upload":
                                summary["uploads"] += 1
                                license_type = entry.get("license_type", "unknown")
                                summary["license_types"][license_type] = (
                                    summary["license_types"].get(license_type, 0) + 1
                                )

                            elif event_type == "content_generation":
                                summary["generations"] += 1

                            elif event_type == "safety_violation":
                                summary["safety_violations"] += 1
                                action = entry.get("action_taken", "unknown")
                                summary["safety_actions"][action] = (
                                    summary["safety_actions"].get(action, 0) + 1
                                )

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except FileNotFoundError:
            pass

        return summary

    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write log entry to audit log file"""
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
