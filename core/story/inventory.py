"""
Inventory System
Item management, equipment, and consumables.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ItemType(Enum):
    WEAPON = "weapon"
    ARMOR = "armor"
    ACCESSORY = "accessory"
    POTION = "potion"
    MATERIAL = "material"
    QUEST = "quest"
    FOOD = "food"
    SCROLL = "scroll"
    KEY = "key"


class EquipmentSlot(Enum):
    MAIN_HAND = "main_hand"
    OFF_HAND = "off_hand"
    HEAD = "head"
    BODY = "body"
    LEGS = "legs"
    FEET = "feet"
    NECK = "neck"
    RING = "ring"
    ACCESSORY = "accessory"


class Rarity(Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

    @property
    def color(self) -> str:
        return {
            self.COMMON: "#FFFFFF",
            self.UNCOMMON: "#1EFF00",
            self.RARE: "#0070DD",
            self.EPIC: "#A335EE",
            self.LEGENDARY: "#FF8000",
        }[self]


# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A single item definition."""
    item_id: str
    name: str
    item_type: ItemType
    rarity: Rarity = Rarity.COMMON
    description: str = ""

    # Equipment
    equipment_slot: Optional[EquipmentSlot] = None
    stats_bonus: Dict[str, int] = field(default_factory=dict)
    # e.g. {"combat.attack": 5, "combat.defense": 3}

    # Consumable
    is_consumable: bool = False
    consume_effect: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"heal_hp": 50, "remove_effect": "poison"}

    # Stackable
    stackable: bool = False
    max_stack: int = 1

    # Quest/Key items
    quest_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "item_type": self.item_type.value,
            "rarity": self.rarity.value,
            "description": self.description,
            "equipment_slot": self.equipment_slot.value if self.equipment_slot else None,
            "stats_bonus": self.stats_bonus.copy(),
            "is_consumable": self.is_consumable,
            "consume_effect": self.consume_effect.copy(),
            "stackable": self.stackable,
            "max_stack": self.max_stack,
            "quest_id": self.quest_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Item":
        return cls(
            item_id=data["item_id"],
            name=data["name"],
            item_type=ItemType(data["item_type"]),
            rarity=Rarity(data.get("rarity", "common")),
            description=data.get("description", ""),
            equipment_slot=EquipmentSlot(data["equipment_slot"]) if data.get("equipment_slot") else None,
            stats_bonus=data.get("stats_bonus", {}),
            is_consumable=data.get("is_consumable", False),
            consume_effect=data.get("consume_effect", {}),
            stackable=data.get("stackable", False),
            max_stack=data.get("max_stack", 1),
            quest_id=data.get("quest_id"),
        )


# ---------------------------------------------------------------------------
# Inventory Entry (item + quantity + equipped state)
# ---------------------------------------------------------------------------

@dataclass
class InventoryEntry:
    """An item slot in the inventory."""
    item: Item
    quantity: int = 1
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    equipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "quantity": self.quantity,
            "instance_id": self.instance_id,
            "equipped": self.equipped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InventoryEntry":
        return cls(
            item=Item.from_dict(data["item"]),
            quantity=data.get("quantity", 1),
            instance_id=data.get("instance_id", uuid.uuid4().hex[:8]),
            equipped=data.get("equipped", False),
        )


# ---------------------------------------------------------------------------
# Inventory Manager
# ---------------------------------------------------------------------------

class InventoryManager:
    """Manages a character's inventory and equipment."""

    def __init__(self, max_slots: int = 50):
        self.max_slots = max_slots
        self.entries: Dict[str, InventoryEntry] = {}

    # ------------------------------------------------------------------
    # Add / Remove
    # ------------------------------------------------------------------

    def add_item(self, item: Item, quantity: int = 1) -> Tuple[int, str]:
        """
        Add item to inventory.
        Returns (added_quantity, message).
        """
        if item.stackable:
            # Try to stack with existing
            for entry in self.entries.values():
                if entry.item.item_id == item.item_id:
                    space = entry.item.max_stack - entry.quantity
                    add = min(quantity, space)
                    if add > 0:
                        entry.quantity += add
                        quantity -= add
                    if quantity == 0:
                        return add, f"已將 {item.name} x{add} 堆疊到現有物品中"

        # Need new slot(s)
        if len(self.entries) + quantity > self.max_slots:
            available = self.max_slots - len(self.entries)
            if available <= 0:
                return 0, f"背包已滿 (最多 {self.max_slots} 格)"
            quantity = available

        added = 0
        for _ in range(quantity):
            entry = InventoryEntry(item=item)
            self.entries[entry.instance_id] = entry
            added += 1

        return added, f"獲得 {item.name} x{added}"

    def remove_item(self, item_id: str, quantity: int = 1) -> Tuple[int, str]:
        """
        Remove item by item_id (FIFO for stackable).
        Returns (removed_quantity, message).
        """
        removed = 0
        to_delete = []

        for iid, entry in list(self.entries.items()):
            if entry.item.item_id == item_id:
                take = min(quantity - removed, entry.quantity)
                entry.quantity -= take
                removed += take
                if entry.quantity <= 0:
                    to_delete.append(iid)
                if removed >= quantity:
                    break

        for iid in to_delete:
            del self.entries[iid]

        name = next((e.item.name for e in self.entries.values() if e.item.item_id == item_id), item_id)
        return removed, f"移除 {name} x{removed}"

    # ------------------------------------------------------------------
    # Equipment
    # ------------------------------------------------------------------

    def equip_item(self, item_id: str) -> Tuple[bool, str]:
        """Equip an item. Unequip existing item in same slot."""
        entry = self._find_by_item_id(item_id)
        if entry is None:
            return False, "找不到該物品"

        item = entry.item
        if item.equipment_slot is None:
            return False, f"{item.name} 無法裝備"

        # Unequip current item in slot
        old_equipped = self._find_equipped_in_slot(item.equipment_slot)
        if old_equipped:
            old_equipped.equipped = False

        entry.equipped = True
        return True, f"裝備了 [{item.name}]"

    def unequip_item(self, item_id: str) -> Tuple[bool, str]:
        """Unequip an item."""
        entry = self._find_by_item_id(item_id)
        if entry is None or not entry.equipped:
            return False, f"{item_id} 未裝備"

        entry.equipped = False
        return True, f"卸下了 [{entry.item.name}]"

    def unequip_slot(self, slot: EquipmentSlot) -> Optional[InventoryEntry]:
        """Unequip whatever is in a slot. Returns the entry."""
        entry = self._find_equipped_in_slot(slot)
        if entry:
            entry.equipped = False
        return entry

    # ------------------------------------------------------------------
    # Consumables
    # ------------------------------------------------------------------

    def use_item(self, item_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Use a consumable item.
        Returns (success, effect_result).
        """
        entry = self._find_by_item_id(item_id)
        if entry is None:
            return False, {"error": "找不到該物品"}

        item = entry.item
        if not item.is_consumable:
            return False, {"error": f"{item.name} 無法使用"}

        if entry.quantity <= 0:
            return False, {"error": "數量不足"}

        # Consume one
        entry.quantity -= 1
        if entry.quantity <= 0:
            for iid, e in list(self.entries.items()):
                if e.instance_id == entry.instance_id:
                    del self.entries[iid]
                    break

        return True, item.consume_effect.copy()

    # ------------------------------------------------------------------
    # Stats calculation
    # ------------------------------------------------------------------

    def get_equipment_bonus(self) -> Dict[str, int]:
        """Sum up stats bonuses from all equipped items."""
        bonus: Dict[str, int] = {}
        for entry in self.entries.values():
            if entry.equipped and entry.item.stats_bonus:
                for stat, value in entry.item.stats_bonus.items():
                    bonus[stat] = bonus.get(stat, 0) + value
        return bonus

    def get_equipped_by_slot(self) -> Dict[EquipmentSlot, Item]:
        """Return equipped items by slot."""
        result = {}
        for entry in self.entries.values():
            if entry.equipped and entry.item.equipment_slot:
                result[entry.item.equipment_slot] = entry.item
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_items(self, item_type: Optional[ItemType] = None) -> List[InventoryEntry]:
        """List all items, optionally filtered by type."""
        items = list(self.entries.values())
        if item_type:
            items = [e for e in items if e.item.item_type == item_type]
        return items

    def count_item(self, item_id: str) -> int:
        """Count total quantity of an item."""
        return sum(e.quantity for e in self.entries.values() if e.item.item_id == item_id)

    def has_item(self, item_id: str, quantity: int = 1) -> bool:
        """Check if player has enough of an item."""
        return self.count_item(item_id) >= quantity

    def get_slot_count(self) -> int:
        """Number of occupied inventory slots."""
        return len(self.entries)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_slots": self.max_slots,
            "entries": {iid: e.to_dict() for iid, e in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InventoryManager":
        inv = cls(max_slots=data.get("max_slots", 50))
        for iid, edata in data.get("entries", {}).items():
            entry = InventoryEntry.from_dict(edata)
            inv.entries[iid] = entry
        return inv

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_by_item_id(self, item_id: str) -> Optional[InventoryEntry]:
        for entry in self.entries.values():
            if entry.item.item_id == item_id:
                return entry
        return None

    def _find_equipped_in_slot(self, slot: EquipmentSlot) -> Optional[InventoryEntry]:
        for entry in self.entries.values():
            if entry.equipped and entry.item.equipment_slot == slot:
                return entry
        return None

    def get_display_list(self) -> List[Dict[str, Any]]:
        """Return inventory formatted for UI display."""
        result = []
        for entry in sorted(self.entries.values(), key=lambda e: (not e.equipped, e.item.name)):
            result.append({
                "instance_id": entry.instance_id,
                "item_id": entry.item.item_id,
                "name": entry.item.name,
                "type": entry.item.item_type.value,
                "rarity": entry.item.rarity.value,
                "quantity": entry.quantity,
                "equipped": entry.equipped,
                "description": entry.item.description,
                "stats_bonus": entry.item.stats_bonus if entry.item.equipment_slot else {},
            })
        return result
