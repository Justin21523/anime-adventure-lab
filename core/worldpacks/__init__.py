"""
WorldPacks (世界包) - Story World Studio

WorldPack 是 Story 的「世界設定 + 角色/NPC 模板 + 視覺風格(LoRA)」整包資料。
存放位置以 AI_WORLDPACKS_ROOT（core/shared_cache.py）為準，若無權限則 fallback 到 repo 內 outputs/。
"""

from .manager import WorldPackManager, get_worldpack_manager

__all__ = ["WorldPackManager", "get_worldpack_manager"]

