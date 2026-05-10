"""
Combat System
Turn-based combat engine with skills, equipment bonuses, and status effects.
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .stats import CharacterStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CombatAction(Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    SPECIAL_ATTACK = "special_attack"
    USE_ITEM = "use_item"
    FLEE = "flee"
    WAIT = "wait"


class CombatPhase(Enum):
    INITIATIVE = "initiative"
    PLAYER_TURN = "player_turn"
    ENEMY_TURN = "enemy_turn"
    END_PHASE = "end_phase"
    RESOLVED = "resolved"


class CombatResult(Enum):
    VICTORY = "victory"
    DEFEAT = "defeat"
    RETREAT = "retreat"
    DRAW = "draw"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DamageResult:
    """Result of a single damage calculation."""
    raw_damage: int
    final_damage: int
    is_critical: bool = False
    is_miss: bool = False
    is_evasion: bool = False
    message: str = ""


@dataclass
class CombatTurnResult:
    """Result of a single combat turn."""
    attacker_id: str
    defender_id: str
    action: CombatAction
    damage_result: Optional[DamageResult] = None
    defender_hp_after: int = 0
    defender_alive: bool = True
    message: str = ""
    xp_gained: int = 0


@dataclass
class Combatant:
    """A participant in combat."""
    character_id: str
    name: str
    stats: CharacterStats
    defending: bool = False        # True if using DEFEND action this turn
    buffs: Dict[str, int] = field(default_factory=dict)
    debuffs: Dict[str, int] = field(default_factory=dict)


@dataclass
class CombatSession:
    """An active combat encounter."""
    session_id: str
    player: Combatant
    enemies: List[Combatant] = field(default_factory=list)
    turn_number: int = 0
    phase: CombatPhase = CombatPhase.INITIATIVE
    result: Optional[CombatResult] = None
    initiative_order: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    def add_enemy(self, enemy: Combatant) -> None:
        self.enemies.append(enemy)

    def get_alive_enemies(self) -> List[Combatant]:
        return [e for e in self.enemies if e.stats.is_alive]

    def is_active(self) -> bool:
        return (self.result is None and
                self.phase != CombatPhase.RESOLVED and
                self.player.stats.is_alive and
                len(self.get_alive_enemies()) > 0)


# ---------------------------------------------------------------------------
# Combat Engine
# ---------------------------------------------------------------------------

class CombatSystem:
    """Turn-based combat engine."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Initiative
    # ------------------------------------------------------------------

    def roll_initiative(self, session: CombatSession) -> List[str]:
        """Determine turn order based on speed + luck."""
        all_combatants = [session.player] + session.enemies
        speed_order = sorted(
            all_combatants,
            key=lambda c: (c.stats.combat.speed + c.stats.luck * 0.1),
            reverse=True,
        )
        order = [c.character_id for c in speed_order]
        session.initiative_order = order
        session.phase = CombatPhase.PLAYER_TURN
        session.log.append(f"⚔️ 戰鬥開始！先手順序: {', '.join(order)}")
        return order

    # ------------------------------------------------------------------
    # Attack resolution
    # ------------------------------------------------------------------

    def calculate_damage(
        self,
        attacker: Combatant,
        defender: Combatant,
        action: CombatAction = CombatAction.ATTACK,
    ) -> DamageResult:
        """Calculate damage from attacker → defender."""

        # --- Miss check ---
        if random.random() > attacker.stats.combat.hit_rate:
            return DamageResult(
                raw_damage=0,
                final_damage=0,
                is_miss=True,
                message=f"{attacker.name} 的攻擊落空了！",
            )

        # --- Evasion check ---
        evasion_rate = defender.stats.combat.evasion
        if defender.defending:
            evasion_rate *= 2.0  # Defending doubles evasion
        if random.random() < min(evasion_rate, 0.9):
            return DamageResult(
                raw_damage=0,
                final_damage=0,
                is_evasion=True,
                message=f"{defender.name} 閃開了攻擊！",
            )

        # --- Base damage ---
        base = attacker.stats.combat.attack
        # Modifier by action type
        if action == CombatAction.SPECIAL_ATTACK:
            base = int(base * 1.5)
        elif action == CombatAction.DEFEND:
            base = max(1, base // 3)

        # Apply buffs/debuffs
        base += attacker.buffs.get("attack", 0)
        base -= attacker.debuffs.get("attack", 0)

        # --- Defense reduction ---
        defense = defender.stats.combat.defense
        defense += defender.buffs.get("defense", 0)
        if defender.defending:
            defense = int(defense * 1.5)

        raw_damage = max(1, base - defense // 2 + random.randint(-3, 3))

        # --- Critical check ---
        crit_rate = attacker.stats.combat.crit_rate
        crit_rate += attacker.buffs.get("crit_rate", 0.0)
        is_crit = random.random() < crit_rate
        if is_crit:
            final_damage = int(raw_damage * attacker.stats.combat.crit_multiplier)
        else:
            final_damage = raw_damage

        msg = f"{attacker.name} 對 {defender.name} 造成 {final_damage} 點傷害"
        if is_crit:
            msg = f"💥 {attacker.name} 暴擊了！{final_damage} 點傷害！"
        if defender.defending:
            msg += " （防禦中）"

        return DamageResult(
            raw_damage=raw_damage,
            final_damage=final_damage,
            is_critical=is_crit,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Counter-attack
    # ------------------------------------------------------------------

    def check_counter(self, defender: Combatant, attacker: Combatant) -> Optional[DamageResult]:
        """Check if defender counter-attacks."""
        if random.random() < defender.stats.combat.counter_rate:
            result = self.calculate_damage(defender, attacker, CombatAction.ATTACK)
            if not result.is_miss and not result.is_evasion:
                result.message += f" → {defender.name} 反擊！"
                return result
        return None

    # ------------------------------------------------------------------
    # Process a single turn
    # ------------------------------------------------------------------

    def process_turn(
        self,
        session: CombatSession,
        combatant_id: str,
        action: CombatAction,
        target_id: Optional[str] = None,
    ) -> CombatTurnResult:
        """Execute one combatant's turn."""

        # Find attacker
        attacker = session.player if combatant_id == session.player.character_id else None
        if attacker is None:
            attacker = next((e for e in session.enemies if e.character_id == combatant_id), None)
        if attacker is None:
            return CombatTurnResult(
                attacker_id=combatant_id,
                defender_id="",
                action=action,
                message="⚠️ 找不到戰鬥參與者",
            )

        # Reset defending flag
        attacker.defending = False

        # Handle non-attack actions
        if action == CombatAction.DEFEND:
            attacker.defending = True
            # Small MP regen
            attacker.stats.combat.mp = min(
                attacker.stats.combat.max_mp,
                attacker.stats.combat.mp + 3,
            )
            return CombatTurnResult(
                attacker_id=combatant_id,
                defender_id="",
                action=action,
                message=f"{attacker.name} 採取防禦姿勢，恢復少量魔力。",
            )

        if action == CombatAction.WAIT:
            attacker.stats.survival.stamina = min(
                100, attacker.stats.survival.stamina + 10
            )
            return CombatTurnResult(
                attacker_id=combatant_id,
                defender_id="",
                action=action,
                message=f"{attacker.name} 暫停一回合，恢復體力。",
            )

        if action == CombatAction.FLEE:
            flee_chance = (attacker.stats.combat.speed + attacker.stats.luck) / 200.0
            if random.random() < flee_chance:
                session.result = CombatResult.RETREAT
                session.phase = CombatPhase.RESOLVED
                session.log.append(f"🏃 {attacker.name} 成功撤退！")
                return CombatTurnResult(
                    attacker_id=combatant_id,
                    defender_id="",
                    action=action,
                    message=f"🏃 {attacker.name} 成功脫戰！",
                )
            else:
                return CombatTurnResult(
                    attacker_id=combatant_id,
                    defender_id="",
                    action=action,
                    message=f"❌ {attacker.name} 撤退失敗！",
                )

        # --- Attack actions ---
        # Pick target
        if action in (CombatAction.ATTACK, CombatAction.SPECIAL_ATTACK):
            if target_id:
                target = next((e for e in session.enemies if e.character_id == target_id), None)
                if target is None and combatant_id != session.player.character_id:
                    # Enemy targeting player
                    target = session.player
            else:
                if combatant_id == session.player.character_id:
                    alive = session.get_alive_enemies()
                    target = random.choice(alive) if alive else session.enemies[0]
                else:
                    target = session.player

            damage = self.calculate_damage(attacker, target, action)
            target.stats.combat.hp -= damage.final_damage
            if target.stats.combat.hp <= 0:
                target.stats.combat.hp = 0
                target.stats.is_alive = False
                damage.message += f" 💀 {target.name} 倒下了！"

            session.log.append(damage.message)

            # Counter-check
            counter = self.check_counter(target, attacker)
            if counter and target.stats.is_alive:
                attacker.stats.combat.hp -= counter.final_damage
                if attacker.stats.combat.hp <= 0:
                    attacker.stats.combat.hp = 0
                    attacker.stats.is_alive = False
                session.log.append(counter.message)

            # MP cost for special attacks
            if action == CombatAction.SPECIAL_ATTACK:
                cost = 10
                attacker.stats.combat.mp = max(0, attacker.stats.combat.mp - cost)

            xp = 0
            if combatant_id == session.player.character_id and not target.stats.is_alive:
                xp = 30 + random.randint(0, 20)

            return CombatTurnResult(
                attacker_id=combatant_id,
                defender_id=target.character_id,
                action=action,
                damage_result=damage,
                defender_hp_after=target.stats.combat.hp,
                defender_alive=target.stats.is_alive,
                message=damage.message,
                xp_gained=xp,
            )

        return CombatTurnResult(
            attacker_id=combatant_id,
            defender_id=target_id or "",
            action=action,
            message=f"{attacker.name} 執行 {action.value}",
        )

    # ------------------------------------------------------------------
    # Full combat resolution
    # ------------------------------------------------------------------

    def resolve_combat(
        self,
        session: CombatSession,
        player_actions: List[Tuple[CombatAction, Optional[str]]],
    ) -> List[CombatTurnResult]:
        """
        Resolve a full combat round.

        player_actions: list of (action, target_id) for each player action.
        """
        results = []
        session.turn_number += 1

        # Player actions first
        for action, target_id in player_actions:
            if not session.is_active():
                break
            result = self.process_turn(session, session.player.character_id, action, target_id)
            results.append(result)

        # Enemy AI (simple: attack player)
        if session.is_active():
            for enemy in session.get_alive_enemies():
                if not session.is_active():
                    break
                # Simple AI: 80% attack, 15% special, 5% defend
                roll = random.random()
                if roll < 0.80:
                    enemy_action = CombatAction.ATTACK
                elif roll < 0.95:
                    enemy_action = CombatAction.SPECIAL_ATTACK if enemy.stats.combat.mp >= 10 else CombatAction.ATTACK
                else:
                    enemy_action = CombatAction.DEFEND

                result = self.process_turn(session, enemy.character_id, enemy_action)
                results.append(result)

        # Check end conditions
        if not session.is_active():
            if not session.player.stats.is_alive:
                session.result = CombatResult.DEFEAT
            elif not session.get_alive_enemies():
                session.result = CombatResult.VICTORY
            session.phase = CombatPhase.RESOLVED

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def format_combat_status(
        self, session: CombatSession
    ) -> str:
        """Generate a text summary of current combat state."""
        lines = [f"⚔️ 第 {session.turn_number} 回合"]
        lines.append(f"🧑 玩家 [{session.player.name}] "
                      f"HP:{session.player.stats.combat.hp}/{session.player.stats.combat.max_hp} "
                      f"MP:{session.player.stats.combat.mp}/{session.player.stats.combat.max_mp}")
        for e in session.enemies:
            status = "💀" if not e.stats.is_alive else "👹"
            lines.append(f"  {status} {e.name} HP:{e.stats.combat.hp}/{e.stats.combat.max_hp}")
        return "\n".join(lines)

    def get_available_actions(self, combatant: Combatant) -> List[Dict[str, Any]]:
        """Return available actions for a combatant with descriptions."""
        actions = [
            {
                "action": CombatAction.ATTACK.value,
                "label": "⚔️ 普通攻擊",
                "description": "發動基礎攻擊",
                "mp_cost": 0,
                "available": True,
            },
            {
                "action": CombatAction.SPECIAL_ATTACK.value,
                "label": "💥 特殊攻擊",
                "description": "造成 1.5x 傷害，消耗 10 MP",
                "mp_cost": 10,
                "available": combatant.stats.combat.mp >= 10,
            },
            {
                "action": CombatAction.DEFEND.value,
                "label": "🛡️ 防禦",
                "description": "提高閃避和防禦，恢復少量 MP",
                "mp_cost": 0,
                "available": True,
            },
            {
                "action": CombatAction.FLEE.value,
                "label": "🏃 逃跑",
                "description": "嘗試脫戰 (成功率取決於速度和幸運)",
                "mp_cost": 0,
                "available": True,
            },
            {
                "action": CombatAction.WAIT.value,
                "label": "⏳ 等待",
                "description": "暫停一回合，恢復體力",
                "mp_cost": 0,
                "available": True,
            },
        ]
        return actions
