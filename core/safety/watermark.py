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

logger = logging.getLogger(__name__)


class WatermarkGenerator:
    """Generate watermarks for images with attribution info"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.default_font_size = 12
        self.watermark_opacity = 128  # 50% transparency

        # Try to load a font
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.ImageFont:
        """Load font for watermark text"""
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/arial.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/TTF/arial.ttf",  # Arch Linux
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, self.default_font_size)
                except Exception:
                    continue

        # Fallback to default font
        return ImageFont.load_default()

    def add_visible_watermark(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom_right",
        opacity: float = 0.5,
    ) -> Image.Image:
        """Add visible watermark to image"""

        # Create a copy to avoid modifying original
        watermarked = image.copy()

        # Create watermark overlay
        overlay = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate text size and position
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = 10
        positions = {
            "bottom_right": (
                watermarked.width - text_width - margin,
                watermarked.height - text_height - margin,
            ),
            "bottom_left": (margin, watermarked.height - text_height - margin),
            "top_right": (watermarked.width - text_width - margin, margin),
            "top_left": (margin, margin),
            "center": (
                (watermarked.width - text_width) // 2,
                (watermarked.height - text_height) // 2,
            ),
        }

        text_pos = positions.get(position, positions["bottom_right"])

        # Draw text with semi-transparent background
        padding = 5
        bg_color = (0, 0, 0, int(opacity * 255 * 0.7))  # Dark background
        draw.rectangle(
            [
                text_pos[0] - padding,
                text_pos[1] - padding,
                text_pos[0] + text_width + padding,
                text_pos[1] + text_height + padding,
            ],
            fill=bg_color,
        )

        # Draw text
        text_color = (255, 255, 255, int(opacity * 255))  # White text
        draw.text(text_pos, text, font=self.font, fill=text_color)

        # Composite the overlay onto the image
        if watermarked.mode != "RGBA":
            watermarked = watermarked.convert("RGBA")

        watermarked = Image.alpha_composite(watermarked, overlay)

        return watermarked

    def add_invisible_watermark(
        self, image: Image.Image, metadata: Dict[str, Any]
    ) -> Image.Image:
        """Add invisible metadata watermark to image"""

        # Convert to RGB if necessary (PNG metadata preservation)
        if image.mode == "RGBA":
            # Create a white background and paste the image
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(
                image, mask=image.split()[-1] if len(image.split()) == 4 else None
            )
            image = background

        # For PNG files, we can add metadata directly
        png_info = PngInfo()

        # Add metadata to PNG info
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                png_info.add_text(f"SagaForge_{key}", str(value))
            else:
                png_info.add_text(
                    f"SagaForge_{key}", json.dumps(value, ensure_ascii=False)
                )

        # Store the PNG info for later saving
        if not hasattr(image, "text"):
            image.text = {}
        image.text.update(png_info.text)

        return image


class AttributionManager:
    """Manage attribution and licensing information for generated content"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.watermark_gen = WatermarkGenerator(cache_root)

    def create_attribution_metadata(
        self,
        generation_params: Dict[str, Any],
        source_files: list = None,
        model_info: Dict[str, Any] = None,
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
        source_files: list = None,
        model_info: Dict[str, Any] = None,
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
        output_path: str,
        metadata: Dict[str, Any],
        format: str = "PNG",
    ) -> str:
        """Save image with proper attribution metadata"""

        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image with metadata
        if format.upper() == "PNG" and hasattr(image, "text"):
            # PNG with metadata
            from PIL.PngImagePlugin import PngInfo

            png_info = PngInfo()

            for key, value in image.text.items():
                png_info.add_text(key, value)

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
