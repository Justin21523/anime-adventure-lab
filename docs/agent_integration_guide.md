# Agent Integration Guide

## Overview

The Agent Story Integration allows the AI Agent to autonomously modify game state during story turns based on narrative events. This provides a more immersive and dynamic gameplay experience.

## How It Works

### 1. Agent Decision Pipeline

```
Story Turn
  ↓
Narrative Generated (LLM)
  ↓
Agent Intervention Check
  ├─> Keywords detected? (quest, damage, item, etc.)
  ├─> Scene objectives active?
  └─> If YES → Agent Decision Layer
      ↓
Agent Analysis (extract events from narrative)
  ├─> Quest completion?
  ├─> Damage taken?
  ├─> Item acquired?
  ├─> NPC encountered?
  └─> Location discovered?
      ↓
Generate Tool Calls
  ↓
Safety Wrapper Execution
  ├─> Validate parameters (whitelist/blacklist)
  ├─> Snapshot game state
  ├─> Execute tools
  ├─> Rollback on failure
  └─> Audit log
      ↓
Update Game State
  ├─> Modify flags (quest_*, npc_met_*, location_discovered_*)
  ├─> Update stats (HP, MP, level, exp, gold)
  └─> Add items to inventory
      ↓
Track Changes in Memory
```

### 2. Intervention Triggers

The Agent will intervene when:

**Keyword Detection** (2+ keywords in narrative):
- Quest: 獲得, 失去, 完成, 達成, gain, lose, complete, achieve, quest, 任務
- Damage: 傷害, 治療, damage, heal
- Items: 物品, 道具, item
- Level: 等級, 升級, level, level up
- Discovery: 發現, 解鎖, discover, unlock

**Context Detection**:
- Current scene has objectives
- NPC present in scene

### 3. Decision Patterns

The Agent uses rule-based pattern matching to detect events:

#### Pattern 1: Quest Completion
**Narrative**: "你成功完成了龍之試煉"
**Action**: Set `quest_dragon_complete = True`

#### Pattern 2: Damage
**Narrative**: "你受到 30 傷害"
**Action**: Update `hp -= 30` (relative change)

#### Pattern 3: Item Acquisition
**Narrative**: "你獲得了一把劍"
**Action**: Add `sword` to inventory

#### Pattern 4: NPC Encounter
**Narrative**: "你遇到了守護者" (NPC in scene)
**Action**: Set `npc_met_guardian = True`

#### Pattern 5: Location Discovery
**Narrative**: "你進入了黑暗森林"
**Action**: Set `location_discovered_dark_forest = True`

## Configuration

### Enable/Disable Agent

```python
from core.story.engine import StoryEngine

# Enable Agent (default)
engine = StoryEngine(enhanced_mode=True, agent_enabled=True)

# Disable Agent
engine = StoryEngine(enhanced_mode=True, agent_enabled=False)
```

### Adjust Intervention Threshold

```python
from core.agents.story_agent_layer import get_agent_layer

agent_layer = get_agent_layer()
agent_layer.intervention_threshold = 0.8  # Default: 0.7
```

### Enable/Disable at Runtime

```python
agent_layer = get_agent_layer()
agent_layer.enabled = False  # Disable Agent
```

## API Usage

### Story Turn with Agent

```bash
POST /api/story/process_turn
{
  "session_id": "game_20251128_123456_player",
  "player_input": "我攻擊巨龍",
  "use_agent": false,  # Old agent system (kept for compatibility)
  "enrich_with_rag": true
}
```

**Response** (with Agent actions):
```json
{
  "session_id": "game_20251128_123456_player",
  "turn_count": 5,
  "narrative": "你向巨龍發起攻擊，成功擊敗了牠！完成了龍之試煉任務，獲得經驗值100點。",
  "choices": [...],
  "stats": {
    "hp": 80,
    "mp": 50,
    "level": 5,
    "exp": 200
  },
  "inventory": ["sword", "health_potion"],
  "agent_used": true,
  "agent_actions": {
    "decision_type": "story_event_processing",
    "reasoning": "Detected quest completion: quest_dragon_complete; Player gained experience",
    "tool_results": [
      {
        "tool": "modify_world_state",
        "success": true,
        "result": {
          "success": true,
          "modified_flags": {
            "quest_dragon_complete": {"old": false, "new": true}
          }
        }
      },
      {
        "tool": "update_character_state",
        "success": true,
        "result": {
          "success": true,
          "modified_stats": {
            "exp": {"old": 100, "new": 200, "change": 100}
          }
        }
      }
    ],
    "overall_success": true,
    "errors": []
  }
}
```

## Safety Guarantees

### What Agent CAN Do ✅

- Set quest flags: `quest_*`
- Track NPC encounters: `npc_met_*`
- Record location discovery: `location_discovered_*`
- Track item acquisition: `item_acquired_*`
- Trigger story events: `event_*`
- Award achievements: `achievement_*`
- Modify stats within bounds: HP (0-9999), MP (0-9999), Level (1-100)
- Add items to inventory
- Search story memories (RAG)
- Generate scene images (T2I)

### What Agent CANNOT Do ❌

- Modify admin flags: `admin_*` - BLOCKED
- Change system settings: `system_*` - BLOCKED
- Set debug flags: `debug_*` - BLOCKED
- Access internal flags: `_*` - BLOCKED
- Set stats outside bounds (auto-clamped or rejected)
- Break game invariants (automatic rollback)

### Security Layers

1. **Whitelist Validation**: Only allowed flag patterns succeed
2. **Blacklist Filtering**: Forbidden patterns immediately rejected
3. **Bounds Checking**: Numeric constraints enforced
4. **Automatic Rollback**: Failed operations restore previous state
5. **Audit Logging**: Every action logged to JSONL with timestamp

## Monitoring

### Audit Logs

Agent actions are logged to `outputs/agent_audit_logs/YYYY-MM-DD.jsonl`:

```json
{
  "timestamp": "2025-11-28T12:34:56.789",
  "session_id": "game_20251128_123456_player",
  "tool_name": "modify_world_state",
  "params": {"flags": {"quest_dragon_complete": true}},
  "result": {"success": true, "modified_flags": {...}},
  "success": true,
  "error": null
}
```

### Query Audit Logs

```python
from core.monitoring.agent_audit_logger import get_audit_logger

logger = get_audit_logger()

# Query by session
actions = logger.query_actions(session_id="game_20251128_123456_player")

# Query by tool
actions = logger.query_actions(tool_name="modify_world_state")

# Query failures only
actions = logger.query_actions(success=False)

# Get statistics
stats = logger.get_statistics()
print(f"Total actions: {stats['total_actions']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Testing

### Run Agent Integration Tests

```bash
# Standalone tests (no pytest required)
conda run -n ai_env python scripts/test_agent_integration.py

# pytest tests (if environment allows)
conda run -n ai_env pytest tests/test_agent_story_integration.py -v
```

### Manual Testing

```python
import asyncio
from core.agents.story_agent_layer import get_agent_layer

async def test_agent():
    agent_layer = get_agent_layer()

    # Test intervention detection
    should_intervene, reason = await agent_layer.should_agent_intervene(
        "test-session",
        "攻擊敵人",
        "你成功擊敗了敵人，獲得經驗值50點",
        None
    )
    print(f"Should intervene: {should_intervene}, Reason: {reason}")

asyncio.run(test_agent())
```

## Future Enhancements

### LLM-Based Decision Making

Currently uses rule-based pattern matching. Future: use LLM to analyze narrative and decide actions.

```python
# Future implementation
async def make_decision(self, ...):
    # Current: rule-based
    if "完成" in narrative_text:
        decision = AgentDecision(...)

    # Future: LLM-based
    llm_response = await self.llm_adapter.generate(agent_prompt)
    decision = parse_llm_decision(llm_response)
```

### Multi-Agent Scenarios

Support multiple agents (character agents, world agents) working together:

```python
# Future
character_agent = get_agent("character_agent")
world_agent = get_agent("world_agent")

# Coordinate decisions
decisions = await coordinate_agents([character_agent, world_agent], context)
```

### Frontend Agent UI

Display Agent actions in story UI:

```typescript
// Future frontend component
<AgentActionsPanel actions={agentActions}>
  {agentActions.tool_results.map(result => (
    <AgentAction key={result.tool} result={result} />
  ))}
</AgentActionsPanel>
```

## Troubleshooting

### Agent Not Intervening

**Check**:
1. Agent enabled? `agent_layer.enabled = True`
2. Enhanced mode? `engine.enhanced_mode = True`
3. Keywords in narrative? Check intervention log
4. Context has objectives? Check scene.scene_objectives

### Agent Actions Failing

**Check**:
1. Audit logs: `outputs/agent_audit_logs/`
2. Error messages in `agent_actions.errors`
3. Validation failures: whitelist/blacklist patterns
4. Bounds checking: HP/MP/level constraints

### Rollback Occurred

**This is normal!** Agent safety wrapper automatically rolls back failed operations to prevent partial state corruption.

**Check audit logs** to see why the operation failed.

## Best Practices

1. **Monitor Audit Logs**: Regularly check for failures and unexpected behavior
2. **Test Narratives**: Test with various narrative patterns to ensure Agent detects events correctly
3. **Adjust Threshold**: Fine-tune `intervention_threshold` based on your game's needs
4. **Use Safety Wrapper**: Always use safety wrapper, never bypass it
5. **Track Changes**: Use `flags_changed` and `stats_changed` in memory to track Agent modifications

## Summary

The Agent Story Integration provides autonomous, secure, and auditable game state modification based on narrative events. It enhances immersion while maintaining game integrity through multi-layer security validation.

**Key Benefits**:
- ✅ Automatic quest/item/NPC tracking
- ✅ Dynamic stat updates based on narrative
- ✅ Secure with whitelist/blacklist validation
- ✅ Automatic rollback on failure
- ✅ Complete audit trail
- ✅ GPU-safe testing with 100% mocks
- ✅ Easy to enable/disable

**Next Steps**:
- Try it in your story session!
- Monitor audit logs
- Adjust patterns/thresholds as needed
- Consider LLM-based decision making for more complex scenarios
