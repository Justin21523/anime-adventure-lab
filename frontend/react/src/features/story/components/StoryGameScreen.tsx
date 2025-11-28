import { useState } from 'react'
import { useStorySession } from '../hooks/useStorySession'
import { NarrativeDisplay } from './NarrativeDisplay'
import { PlayerInput } from './PlayerInput'
import { CharacterSheet } from './CharacterSheet'
import { SceneVisualizer, SceneVisualizerSkeleton } from './SceneVisualizer'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useUiStore } from '@/stores/uiStore'

interface StoryGameScreenProps {
  sessionId: string
}

export function StoryGameScreen({ sessionId }: StoryGameScreenProps) {
  const { session, isLoading, executeTurn, refetch } = useStorySession(sessionId)
  const { addNotification } = useUiStore()
  const [isExecuting, setIsExecuting] = useState(false)

  const handlePlayerInput = async (input: string) => {
    if (!session) return

    setIsExecuting(true)
    try {
      await executeTurn.mutateAsync({
        session_id: sessionId,
        player_input: input,
      })

      addNotification({
        type: 'success',
        title: '回合執行成功',
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '回合執行失敗',
        message: error instanceof Error ? error.message : '發生未知錯誤',
      })
    } finally {
      setIsExecuting(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-lg font-semibold">加載故事中...</div>
        </div>
      </div>
    )
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="p-8">
          <div className="text-center space-y-4">
            <div className="text-lg font-semibold">找不到故事會話</div>
            <Button onClick={() => window.location.href = '/'}>返回首頁</Button>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* 左側：場景圖像 */}
      <div className="w-96 p-6 border-r border-slate-700 overflow-y-auto">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-white mb-2">場景圖像</h2>
        </div>
        {isExecuting ? (
          <SceneVisualizerSkeleton />
        ) : (
          <SceneVisualizer sceneImage={session.scene_image} showMetadata={true} />
        )}
      </div>

      {/* 中間：敘事區域 */}
      <div className="flex-1 flex flex-col p-6 overflow-hidden">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{session.player_name} 的冒險</h1>
            <p className="text-sm text-slate-400">回合 {session.turn_count}</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              刷新
            </Button>
            <Button variant="outline" size="sm" onClick={() => window.location.href = '/'}>
              離開
            </Button>
          </div>
        </div>

        {/* 敘事顯示 */}
        <div className="flex-1 overflow-hidden mb-4">
          <NarrativeDisplay
            narrative={session.narrative}
            dialogue={session.dialogue}
            choices={session.choices}
          />
        </div>

        {/* 玩家輸入 */}
        <PlayerInput
          onSubmit={handlePlayerInput}
          isExecuting={isExecuting}
          choices={session.choices}
        />
      </div>

      {/* 右側：角色面板 */}
      <div className="w-80 p-6 bg-slate-900/50 border-l border-slate-700">
        <CharacterSheet
          character={session.character}
          inventory={session.inventory}
          flags={session.flags}
        />
      </div>
    </div>
  )
}
