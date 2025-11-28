"""Celery task exports.

This package bridges to the legacy `workers/tasks.py` module (file-based) to keep
imports like `from workers.tasks import batch_caption_task` working.
"""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import ModuleType


def _load_legacy_tasks() -> ModuleType:
    """Load the legacy tasks module by path to avoid package-name conflicts."""
    module_path = Path(__file__).resolve().parent.parent / "tasks.py"
    spec = spec_from_file_location("workers_tasks_legacy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy tasks module at {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_legacy = _load_legacy_tasks()

# Re-export required Celery tasks/helpers
batch_caption_task = _legacy.batch_caption_task
batch_vqa_task = _legacy.batch_vqa_task
batch_chat_task = _legacy.batch_chat_task
get_task_status = _legacy.get_task_status
cancel_task = _legacy.cancel_task

__all__ = [
    "batch_caption_task",
    "batch_vqa_task",
    "batch_chat_task",
    "get_task_status",
    "cancel_task",
]
