#!/usr/bin/env python3
"""直接測試 story engine，繞過 API"""
import sys
import asyncio
sys.path.insert(0, '/mnt/c/ai_projects/anime-adventure-lab')

from core.story.engine import get_story_engine

async def test_engine():
    print("=" * 60)
    print("直接測試 Story Engine")
    print("=" * 60)

    try:
        # 獲取 engine
        engine = get_story_engine()
        print("✓ Story engine 獲取成功")

        # 創建 session
        print("\n創建 session...")
        session = engine.create_session(
            player_name="測試玩家",
            persona_id="wise_sage",
            setting="測試環境",
            difficulty="medium"
        )
        print(f"✓ Session 創建成功: {session.session_id}")

        # 處理第一個 turn（這個應該已經在 create_session 中完成了）
        print(f"  Turn count: {session.turn_count}")
        print(f"  Enhanced mode: {engine.enhanced_mode}")
        print(f"  Has context memory: {session.session_id in engine.context_memories}")

        # 嘗試處理第二個 turn
        print("\n處理第二個 turn...")
        result = await engine.process_turn(
            session_id=session.session_id,
            player_input="環顧四周",
            choice_id=None
        )

        print(f"✓ Turn 處理成功!")
        print(f"  Narrative: {result['narrative'][:100]}...")
        print(f"  Choices: {len(result['choices'])}")
        print(f"  Turn count: {result['turn_count']}")

        print("\n" + "=" * 60)
        print("✓ 所有測試通過!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_engine())
    sys.exit(0 if success else 1)
