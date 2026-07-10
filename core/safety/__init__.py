"""Safety package with dependency-safe lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ContentFilter": ("core.safety.content_filter", "ContentFilter"),
    "get_content_filter": ("core.safety.content_filter", "get_content_filter"),
    "RateLimiterManager": ("core.safety.rate_limiter", "RateLimiterManager"),
    "get_rate_limiter": ("core.safety.rate_limiter", "get_rate_limiter"),
    "InputValidator": ("core.safety.validator", "InputValidator"),
    "get_input_validator": ("core.safety.validator", "get_input_validator"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
