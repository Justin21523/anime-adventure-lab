from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.shared_cache import get_shared_cache
from schemas.world import (
    WorldCharacterTemplate,
    WorldCreateRequest,
    WorldPack,
    WorldPlayerTemplate,
    WorldSummary,
    WorldVisualStyle,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass(frozen=True)
class WorldPackStoragePaths:
    root: Path

    def file_for(self, world_id: str) -> Path:
        return self.root / f"{world_id}.json"


class WorldPackManager:
    """WorldPack 的 CRUD 與落盤管理。"""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        if root_dir is None:
            root_dir = self._resolve_root_dir()
        self.paths = WorldPackStoragePaths(root=root_dir)
        try:
            self.paths.root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to ensure worldpacks root dir: %s", exc)

        # Best-effort: ensure default exists so Story 永遠有可用世界
        try:
            self.ensure_default_worldpack()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip ensure default worldpack: %s", exc)

    def _resolve_root_dir(self) -> Path:
        """Prefer AI_WORLDPACKS_ROOT; fallback to repo-local outputs/worldpacks."""
        try:
            cache = get_shared_cache()
            path_str = cache.get_path("WORLDPACKS")
            root = Path(path_str)
            root.mkdir(parents=True, exist_ok=True)
            return root
        except Exception as exc:  # noqa: BLE001
            logger.warning("Worldpacks root fallback to outputs/: %s", exc)
            root = Path("outputs") / "worldpacks"
            root.mkdir(parents=True, exist_ok=True)
            return root

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def list_worldpacks(self) -> List[WorldSummary]:
        items: List[WorldSummary] = []

        if not self.paths.root.exists():
            return items

        for file_path in sorted(self.paths.root.glob("*.json")):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                pack = WorldPack(**data)
                items.append(
                    WorldSummary(
                        world_id=pack.world_id,
                        name=pack.name,
                        description=pack.description,
                        setting=pack.setting,
                        difficulty=pack.difficulty,
                        updated_at=pack.updated_at,
                        player_templates_count=len(pack.player_templates),
                        characters_count=len(pack.characters),
                        default_loras_count=len(pack.visual.default_loras),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip invalid worldpack file %s: %s", file_path, exc)
                continue

        return items

    def get_worldpack(self, world_id: str) -> Optional[WorldPack]:
        file_path = self.paths.file_for(world_id)
        if not file_path.exists():
            if world_id == "default":
                return self.ensure_default_worldpack()
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            pack = WorldPack(**data)
            # 若檔名與內容不一致，仍以內容為準，但 log 一下
            if pack.world_id != world_id:
                logger.warning(
                    "Worldpack id mismatch: file=%s payload=%s",
                    world_id,
                    pack.world_id,
                )
            return pack
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load worldpack %s: %s", world_id, exc)
            raise

    def create_worldpack(self, request: WorldCreateRequest) -> WorldPack:
        if self.get_worldpack(request.world_id) is not None:
            raise ValueError(f"world_id already exists: {request.world_id}")

        pack = WorldPack(
            world_id=request.world_id,
            name=request.name,
            description=request.description,
            setting=request.setting,
            difficulty=request.difficulty,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self.save_worldpack(pack)
        return pack

    def save_worldpack(self, pack: WorldPack) -> None:
        file_path = self.paths.file_for(pack.world_id)
        payload = pack.model_dump()
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update_worldpack(self, world_id: str, pack: WorldPack) -> WorldPack:
        if world_id != pack.world_id:
            raise ValueError("world_id path param must match payload.world_id")

        existing = self.get_worldpack(world_id)
        if existing is None:
            raise FileNotFoundError(f"worldpack not found: {world_id}")

        pack.updated_at = _now_iso()  # type: ignore[misc]
        if not getattr(pack, "created_at", None):
            pack.created_at = existing.created_at  # type: ignore[misc]

        self.save_worldpack(pack)
        return pack

    def delete_worldpack(self, world_id: str) -> None:
        if world_id == "default":
            raise ValueError("default worldpack cannot be deleted")

        file_path = self.paths.file_for(world_id)
        if not file_path.exists():
            raise FileNotFoundError(f"worldpack not found: {world_id}")
        file_path.unlink()

    # ---------------------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------------------
    def ensure_default_worldpack(self) -> WorldPack:
        file_path = self.paths.file_for("default")
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                return WorldPack(**data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Default worldpack corrupted, recreating: %s", exc)

        starter = WorldPack(
            world_id="default",
            name="預設世界",
            description="通用冒險世界（可在「世界工作室」修改）",
            setting="fantasy",
            difficulty="medium",
            runtime_preset_id="rtx_5080_16gb",
            visual=WorldVisualStyle(
                prompt_prefix="anime style, cinematic lighting, high detail",
                negative_prompt="lowres, blurry, bad anatomy, extra fingers",
                base_model=None,
                default_loras=[],
            ),
            player_templates=[
                WorldPlayerTemplate(
                    template_id="adventurer",
                    name="冒險者",
                    description="均衡型主角，適合探索與互動",
                    personality_traits=["勇敢", "好奇", "善良"],
                    speaking_style="直接而友善",
                    background_story="一位踏上旅途的冒險者，渴望探索未知。",
                    motivations=["探索世界", "幫助他人", "成長"],
                    persona_prompt="你是玩家主角，行動果斷但尊重他人，遇到選擇會先觀察再決定。",
                ),
            ],
            characters=[
                WorldCharacterTemplate(
                    character_id="wise_guide",
                    name="智者艾莉亞",
                    role="companion",
                    personality_traits=["智慧", "神秘", "指導性"],
                    speaking_style="充滿智慧且神秘的語調",
                    background_story="一位古老的魔法師，掌握著許多秘密知識。",
                    motivations=["指導後輩", "保護古老知識", "維持平衡"],
                    relationships={"player": "mentor"},
                    persona_prompt="你是一位智慧的嚮導，總是給予有用的建議，但保留關鍵伏筆。",
                    content_restrictions=["避免劇透核心謎底", "保持神秘感"],
                    start_in_opening=True,
                ),
                WorldCharacterTemplate(
                    character_id="merchant_bob",
                    name="商人巴布",
                    role="npc",
                    personality_traits=["精明", "友善", "商業頭腦"],
                    speaking_style="熱情且具說服力的商業語調",
                    background_story="一個經驗豐富的旅行商人，足跡遍布各地。",
                    motivations=["獲利", "建立人脈", "分享故事"],
                    relationships={"player": "business_partner"},
                    persona_prompt="你是一個友善的商人，喜歡交易與聊天，會用小故事推銷商品。",
                    content_restrictions=["不做詐騙交易", "避免過度暴力內容"],
                    start_in_opening=False,
                ),
            ],
            world_flags={
                "magic_enabled": True,
                "combat_enabled": True,
            },
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )

        self.save_worldpack(starter)
        return starter


_worldpack_manager: Optional[WorldPackManager] = None


def get_worldpack_manager() -> WorldPackManager:
    global _worldpack_manager
    if _worldpack_manager is None:
        _worldpack_manager = WorldPackManager()
    return _worldpack_manager
