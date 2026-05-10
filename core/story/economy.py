"""
Economy System
Gold, shops, purchases, sales, and quest rewards.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .inventory import Item, ItemType, Rarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShopItem:
    """An item available for purchase at a shop."""
    item_id: str
    item: Item
    buy_price: int
    sell_price: int
    stock: int = 10
    restock_per_day: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item": self.item.to_dict(),
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "stock": self.stock,
            "restock_per_day": self.restock_per_day,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShopItem":
        return cls(
            item_id=data["item_id"],
            item=Item.from_dict(data["item"]),
            buy_price=data["buy_price"],
            sell_price=data["sell_price"],
            stock=data.get("stock", 10),
            restock_per_day=data.get("restock_per_day", 0),
        )


@dataclass
class Shop:
    """A shop/location where trading happens."""
    shop_id: str
    name: str
    description: str = ""
    items: Dict[str, ShopItem] = field(default_factory=dict)
    discount: float = 0.0       # 0.0-1.0, relation discount
    surcharge: float = 0.0      # 0.0-1.0, enemy territory surcharge

    def add_item(self, shop_item: ShopItem) -> None:
        self.items[shop_item.item_id] = shop_item

    def get_item(self, item_id: str) -> Optional[ShopItem]:
        return self.items.get(item_id)

    def list_available(self) -> List[ShopItem]:
        return [si for si in self.items.values() if si.stock > 0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shop_id": self.shop_id,
            "name": self.name,
            "description": self.description,
            "items": {iid: si.to_dict() for iid, si in self.items.items()},
            "discount": self.discount,
            "surcharge": self.surcharge,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Shop":
        shop = cls(
            shop_id=data["shop_id"],
            name=data["name"],
            description=data.get("description", ""),
            discount=data.get("discount", 0.0),
            surcharge=data.get("surcharge", 0.0),
        )
        for iid, idata in data.get("items", {}).items():
            shop.items[iid] = ShopItem.from_dict(idata)
        return shop


@dataclass
class QuestReward:
    """Reward for completing a quest."""
    quest_id: str
    gold: int = 0
    xp: int = 0
    items: List[Tuple[str, int]] = field(default_factory=list)  # (item_id, quantity)
    stat_bonus: Dict[str, int] = field(default_factory=dict)
    reputation_change: Dict[str, int] = field(default_factory=dict)  # faction: delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quest_id": self.quest_id,
            "gold": self.gold,
            "xp": self.xp,
            "items": self.items,
            "stat_bonus": self.stat_bonus,
            "reputation_change": self.reputation_change,
        }


# ---------------------------------------------------------------------------
# Economy Manager
# ---------------------------------------------------------------------------

class EconomyManager:
    """Manages gold, shops, and transactions."""

    def __init__(self):
        self.shops: Dict[str, Shop] = {}
        self.quest_rewards: Dict[str, QuestReward] = {}

    # ------------------------------------------------------------------
    # Gold
    # ------------------------------------------------------------------

    def spend_gold(self, amount: int, current_gold: int) -> Tuple[int, int, str]:
        """
        Spend gold. Returns (new_gold, amount_spent, message).
        """
        if amount > current_gold:
            return current_gold, 0, f"💰 金幣不足 (需要 {amount}, 持有 {current_gold})"
        new_gold = current_gold - amount
        return new_gold, amount, f"💰 花費 {amount} 金幣，剩餘 {new_gold}"

    def earn_gold(self, amount: int, current_gold: int, reason: str = "") -> Tuple[int, str]:
        """
        Earn gold. Returns (new_gold, message).
        """
        new_gold = current_gold + amount
        msg = f"💰 獲得 {amount} 金幣"
        if reason:
            msg += f" ({reason})"
        return new_gold, msg

    # ------------------------------------------------------------------
    # Shop management
    # ------------------------------------------------------------------

    def register_shop(self, shop: Shop) -> None:
        self.shops[shop.shop_id] = shop

    def get_shop(self, shop_id: str) -> Optional[Shop]:
        return self.shops.get(shop_id)

    def list_shops(self) -> List[Shop]:
        return list(self.shops.values())

    # ------------------------------------------------------------------
    # Buy / Sell
    # ------------------------------------------------------------------

    def buy(
        self,
        shop_id: str,
        item_id: str,
        quantity: int,
        player_gold: int,
        player_inventory,  # InventoryManager instance
    ) -> Dict[str, Any]:
        """
        Buy items from a shop.
        Returns {"success": bool, "message": str, "gold_spent": int, "new_gold": int, "items_added": int}
        """
        shop = self.shops.get(shop_id)
        if shop is None:
            return {"success": False, "message": "找不到該商店"}

        shop_item = shop.get_item(item_id)
        if shop_item is None:
            return {"success": False, "message": "商店沒有這個物品"}

        if shop_item.stock < quantity:
            return {"success": False, "message": f"庫存不足 (剩餘 {shop_item.stock})"}

        # Calculate price with modifiers
        unit_price = int(shop_item.buy_price * (1 - shop.discount + shop.surcharge))
        total_price = unit_price * quantity

        new_gold, spent, msg = self.spend_gold(total_price, player_gold)
        if not (new_gold >= 0 and spent > 0):
            # Partial buy
            affordable = player_gold // unit_price if unit_price > 0 else 0
            if affordable <= 0:
                return {"success": False, "message": "金幣不足"}
            quantity = affordable
            total_price = unit_price * quantity
            new_gold, spent, msg = self.spend_gold(total_price, player_gold)

        # Deduct stock
        shop_item.stock -= quantity

        # Add to inventory
        added, add_msg = player_inventory.add_item(shop_item.item, quantity)

        return {
            "success": True,
            "message": f"購買 {shop_item.item.name} x{added}，花費 {spent} 金幣",
            "gold_spent": spent,
            "new_gold": new_gold,
            "items_added": added,
        }

    def sell(
        self,
        shop_id: str,
        item_id: str,
        quantity: int,
        player_inventory,
    ) -> Dict[str, Any]:
        """
        Sell items to a shop.
        Returns {"success": bool, "message": str, "gold_earned": int}
        """
        shop = self.shops.get(shop_id)
        if shop is None:
            return {"success": False, "message": "找不到該商店"}

        # Check player has the item
        if not player_inventory.has_item(item_id, quantity):
            available = player_inventory.count_item(item_id)
            return {"success": False, "message": f"你沒有足夠的該物品 (持有 {available})"}

        # Find best sell price across all shop items with same item_id
        shop_item = shop.get_item(item_id)
        if shop_item is None:
            # Shop doesn't buy this type — offer generic scrap value
            unit_price = 5  # Scrap price
        else:
            unit_price = int(shop_item.sell_price * (1 - shop.discount + shop.surcharge))

        gold_earned = unit_price * quantity

        # Remove from inventory
        removed, _ = player_inventory.remove_item(item_id, quantity)
        gold_earned = unit_price * removed

        return {
            "success": True,
            "message": f"出售 {removed} 個物品，獲得 {gold_earned} 金幣",
            "gold_earned": gold_earned,
            "items_sold": removed,
        }

    # ------------------------------------------------------------------
    # Quest rewards
    # ------------------------------------------------------------------

    def register_reward(self, reward: QuestReward) -> None:
        self.quest_rewards[reward.quest_id] = reward

    def collect_reward(
        self,
        quest_id: str,
        player_gold: int,
        player_stats,  # CharacterStats
        player_inventory,  # InventoryManager
    ) -> Dict[str, Any]:
        """
        Collect quest completion rewards.
        Returns {"success": bool, "gold": int, "xp": int, "items": list, "messages": list}
        """
        reward = self.quest_rewards.get(quest_id)
        if reward is None:
            return {"success": False, "messages": ["找不到該任務的獎勵"]}

        messages = []
        new_gold = player_gold

        # Gold
        if reward.gold > 0:
            new_gold, msg = self.earn_gold(reward.gold, new_gold, f"任務: {quest_id}")
            messages.append(msg)

        # XP
        xp_gained = 0
        if reward.xp > 0 and hasattr(player_stats, "gain_xp"):
            remaining, leveled = player_stats.gain_xp(reward.xp)
            xp_gained = reward.xp
            messages.append(f"✨ 獲得 {reward.xp} 經驗值")
            if leveled:
                messages.append(f"🎉 等級提升至 Lv.{player_stats.level}！")

        # Items
        items_added = []
        for item_id, qty in reward.items:
            # Create a basic item if we don't have a registry
            item = Item(
                item_id=item_id,
                name=item_id,
                item_type=ItemType.MATERIAL,
            )
            added, _ = player_inventory.add_item(item, qty)
            items_added.append({"item_id": item_id, "added": added})
            messages.append(f"📦 獲得 {item_id} x{added}")

        # Stat bonus
        stat_changes = []
        for stat_path, delta in reward.stat_bonus.items():
            if hasattr(player_stats, "apply_bonus"):
                player_stats.apply_bonus(stat_path, delta)
                stat_changes.append({"stat": stat_path, "delta": delta})
                messages.append(f"⬆️ {stat_path} +{delta}")

        return {
            "success": True,
            "gold": new_gold,
            "xp": xp_gained,
            "items": items_added,
            "stat_changes": stat_changes,
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_price(item: Item, base_multiplier: float = 1.0) -> int:
        """Calculate a fair price for an item based on rarity."""
        rarity_prices = {
            Rarity.COMMON: 1,
            Rarity.UNCOMMON: 2,
            Rarity.RARE: 5,
            Rarity.EPIC: 10,
            Rarity.LEGENDARY: 25,
        }
        rarity_mult = rarity_prices.get(item.rarity, 1)

        # Base price from stats bonus
        stat_value = sum(abs(v) for v in item.stats_bonus.values()) * 10

        price = int((50 * rarity_mult + stat_value) * base_multiplier)
        return max(1, price)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shops": {sid: s.to_dict() for sid, s in self.shops.items()},
            "quest_rewards": {qid: r.to_dict() for qid, r in self.quest_rewards.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EconomyManager":
        mgr = cls()
        for sid, sdata in data.get("shops", {}).items():
            mgr.shops[sid] = Shop.from_dict(sdata)
        for qid, qdata in data.get("quest_rewards", {}).items():
            mgr.quest_rewards[qid] = QuestReward(**qdata)
        return mgr
