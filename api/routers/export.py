# api/routers/export.py
"""
Export/Import Router
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException

# Ensure cache root is writable in restricted environments
os.environ.setdefault("AI_CACHE_ROOT", "/tmp/ai_cache")

from core.export import get_model_exporter, get_format_converter
from core.export.story_exporter import StoryExporter, ExportConfig, StorySession, StoryTurn
from core.exceptions import ValidationError, ExportError
from schemas.export import (
    ExportRequest,
    ExportResponse,
    ConvertRequest,
    ConvertResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_model_exporter_instance = None
_format_converter_instance = None
_story_exporter_instance = None


def _get_model_exporter():
    global _model_exporter_instance
    if _model_exporter_instance is None:
        try:
            _model_exporter_instance = get_model_exporter()
        except PermissionError:
            os.environ["AI_CACHE_ROOT"] = "/tmp/ai_cache"
            try:
                from core.config import get_config

                cfg = get_config()
                cfg.cache.root = "/tmp/ai_cache"  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                from core.utils import cache as cache_module
                if hasattr(cache_module, "_cache_manager"):
                    cache_module._cache_manager = None  # type: ignore[attr-defined]
            except Exception:
                pass
            _model_exporter_instance = get_model_exporter()
    return _model_exporter_instance


def _get_format_converter():
    global _format_converter_instance
    if _format_converter_instance is None:
        try:
            _format_converter_instance = get_format_converter()
        except PermissionError:
            os.environ["AI_CACHE_ROOT"] = "/tmp/ai_cache"
            try:
                from core.config import get_config

                cfg = get_config()
                cfg.cache.root = "/tmp/ai_cache"  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                from core.utils import cache as cache_module
                from core.utils.cache import CacheManager

                cache_module._cache_manager = CacheManager()  # type: ignore[attr-defined]
            except Exception:
                pass
            _format_converter_instance = get_format_converter()
    return _format_converter_instance


def _get_story_exporter():
    global _story_exporter_instance
    if _story_exporter_instance is None:
        try:
            _story_exporter_instance = StoryExporter(ExportConfig())
        except PermissionError:
            os.environ["AI_CACHE_ROOT"] = "/tmp/ai_cache"
            try:
                from core.config import get_config

                cfg = get_config()
                cfg.cache.root = "/tmp/ai_cache"  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                from core.utils import cache as cache_module
                if hasattr(cache_module, "_cache_manager"):
                    cache_module._cache_manager = None  # type: ignore[attr-defined]
            except Exception:
                pass
            _story_exporter_instance = StoryExporter(ExportConfig())
    return _story_exporter_instance


def _export_root() -> Path:
    """Resolve export root with safe fallback."""
    cache_root = Path(os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache"))
    out_dir = cache_root / "outputs" / "exports"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        out_dir = Path("/tmp/ai_cache/outputs/exports")
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _file_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 4)
    except Exception:
        return 0.0


@router.post("/export", response_model=ExportResponse)
async def export_data(request: ExportRequest):
    """Export system data"""
    try:
        export_id = f"export_{request.export_type}_{uuid.uuid4().hex[:8]}"
        warnings: List[str] = []
        results: List[Dict[str, Any]] = []
        out_dir = _export_root()

        if request.export_type == "models":
            if not request.items:
                raise ValidationError("items", None, "At least one model path is required")
            model_exporter = _get_model_exporter()
            for item in request.items:
                model_path = Path(item)
                if not model_path.exists():
                    raise ValidationError("items", item, "Model path does not exist")
                output_path = out_dir / f"{model_path.stem}.{request.format}"
                try:
                    result = model_exporter.export_model(
                        model_path=str(model_path),
                        output_path=str(output_path),
                        export_format=request.format,
                        **request.options,
                    )
                    result["output_path"] = str(output_path)
                    result["file_size_mb"] = _file_size_mb(output_path)
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(500, f"Model export failed: {exc}") from exc

            file_path = results[0].get("output_path", str(out_dir))
            file_size_mb = results[0].get("file_size_mb", 0.0)

        elif request.export_type in {"documents", "configurations"}:
            if not request.items:
                raise ValidationError("items", None, "At least one file path is required")
            src = Path(request.items[0])
            if not src.exists():
                raise ValidationError("items", request.items[0], "Input path does not exist")
            target_ext = request.format
            output_path = out_dir / f"{src.stem}.{target_ext}"
            format_converter = _get_format_converter()
            try:
                conv_result = format_converter.convert(
                    input_path=str(src),
                    output_path=str(output_path),
                    source_format=src.suffix.lstrip(".") or "json",
                    target_format=request.format,
                    **request.options,
                )
                results.append(conv_result)
                file_path = str(output_path)
                file_size_mb = _file_size_mb(output_path)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"Format conversion failed: {exc}") from exc

        elif request.export_type == "sessions":
            # Basic story session export (expects JSON-like items)
            if not request.items:
                raise ValidationError("items", None, "Session data required")
            story_exporter = _get_story_exporter()
            try:
                # Assume first item is a path to session json
                session_path = Path(request.items[0])
                if not session_path.exists():
                    raise ValidationError("items", request.items[0], "Session file not found")
                import json

                with open(session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                story = StorySession(**data)
                export_path = (
                    story_exporter.export_to_html(story)
                    if request.format == "html"
                    else story_exporter.export_to_json(story)
                )
                file_path = export_path
                file_size_mb = _file_size_mb(Path(export_path))
                results.append({"output_path": export_path, "format": request.format})
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"Story export failed: {exc}") from exc

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export type: {request.export_type}")

        return ExportResponse(  # type: ignore
            export_id=export_id,
            file_path=file_path,
            file_size_mb=file_size_mb,
            items_exported=len(request.items),
            results=results or None,
            warnings=warnings or None,
        )
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("Export failed: %s", e)
        raise HTTPException(500, f"Export failed: {str(e)}") from e


@router.post("/convert", response_model=ConvertResponse)
async def convert_format(request: ConvertRequest):
    """Convert formats using the format converter utility."""
    try:
        format_converter = _get_format_converter()
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise HTTPException(404, "Input path not found")

        output_path = (
            Path(request.output_path)
            if request.output_path
            else _export_root() / f"{input_path.stem}.{request.target_format}"
        )

        result = format_converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            source_format=request.source_format,
            target_format=request.target_format,
            **request.options,
        )

        return ConvertResponse(
            input_path=str(input_path),
            output_path=str(output_path),
            source_format=request.source_format,
            target_format=request.target_format,
            details=result,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Conversion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/export/formats")
async def list_export_formats():
    """List supported model export formats and converter mappings."""
    model_exporter = _get_model_exporter()
    format_converter = _get_format_converter()
    return {
        "model_formats": list(model_exporter.supported_formats.keys()),
        "converter_mappings": list(format_converter.conversions.keys()),
    }
