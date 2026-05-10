import { useEffect, useMemo, useRef, useState, lazy, Suspense } from 'react'
import { useStorySession } from '../hooks/useStorySession'
import { useStoryContext } from '../hooks/useStoryContext'
import { NarrativeDisplay } from './NarrativeDisplay'
import { PlayerInput } from './PlayerInput'
import { CharacterSheet } from './CharacterSheet'
import { RelationshipPanel } from './RelationshipPanel'
import { QuestExplorationPanel } from './QuestExplorationPanel'
import { AgentProfilePanel } from './AgentProfilePanel'
import { ReviewQueuePanel } from './ReviewQueuePanel'
import { SceneVisualizer, SceneVisualizerError, SceneVisualizerSkeleton } from './SceneVisualizer'
import { MemoryIndicator } from './MemoryIndicator'
import { TurnTimeline } from './TurnTimeline'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { useUiStore } from '@/stores/uiStore'
import { StoryWorkbenchDialog } from './StoryWorkbenchDialog'
import type { RagMode } from '../types/story.types'
import { useWorld } from '@/features/worlds/hooks/useWorld'
import { usePersona } from '../hooks/usePersonas'
import { useStoryWorldSync } from '../hooks/useStoryWorldSync'
import { useStoryWorldWritebackSuggest } from '../hooks/useStoryWorldWritebackSuggest'
import { JobProgressCard } from '@/features/jobs/components/JobProgressCard'
import { useJob } from '@/features/jobs/hooks/useJob'
import { enqueueReviewQueueItem, loadReviewQueue } from '../lib/reviewQueueStorage'
import type { ReviewQueueItem } from '../types/reviewQueue.types'
import { VisualNovelScreen } from './VisualNovelScreen'
import { Tv } from 'lucide-react'

// Lazy load heavy components for better initial load performance
const RecentMemories = lazy(() => import('./RecentMemories').then(m => ({ default: m.RecentMemories })))
const AgentActionsPanel = lazy(() => import('./AgentActionsPanel').then(m => ({ default: m.AgentActionsPanel })))

interface StoryGameScreenProps {
  sessionId: string
}

export function StoryGameScreen({ sessionId }: StoryGameScreenProps) {
  const { session, isLoading, executeTurn, executeTurnJob, refetch } = useStorySession(sessionId)
  const {
    data: storyContext,
    isLoading: contextLoading,
    error: contextError,
  } = useStoryContext(sessionId)
  const { addNotification } = useUiStore()
  const [isSubmittingTurn, setIsSubmittingTurn] = useState(false)
  const [turnJobId, setTurnJobId] = useState<string | null>(null)
  const [workbenchOpen, setWorkbenchOpen] = useState(false)
  const [workbenchTab, setWorkbenchTab] = useState('world')
  const syncWorldpack = useStoryWorldSync()
  const writebackSuggest = useStoryWorldWritebackSuggest()
  const [autoSync, setAutoSync] = useState(false)
  const [autoReview, setAutoReview] = useState(false)
  const [autoWriteback, setAutoWriteback] = useState(false)
  const [prefillText, setPrefillText] = useState<string | null>(null)
  const [isVisualNovelMode, setIsVisualNovelMode] = useState(true)

  const { data: world } = useWorld(session?.world_id || 'default')
  const { data: persona } = usePersona(session?.persona_id || undefined)

  const playerTemplateName = useMemo(() => {
    const templateId = session?.player_template_id
    if (!templateId || !world?.player_templates?.length) return null
    const found = world.player_templates.find((t) => t.template_id === templateId)
    return found?.name || null
  }, [session?.player_template_id, world?.player_templates])

  useEffect(() => {
    try {
      const key = `story_ui_mode_${sessionId}`
      const raw = localStorage.getItem(key)
      if (raw !== null) {
        setIsVisualNovelMode(raw === 'vn')
      }
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    try {
      const key = `story_ui_mode_${sessionId}`
      localStorage.setItem(key, isVisualNovelMode ? 'vn' : 'classic')
    } catch {
      // ignore
    }
  }, [isVisualNovelMode, sessionId])

  useEffect(() => {
    try {
      const key = `worldstudio_auto_sync_${sessionId}`
      const raw = localStorage.getItem(key)
      if (raw === null) return
      setAutoSync(raw === 'true')
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    try {
      const key = `worldstudio_auto_sync_${sessionId}`
      localStorage.setItem(key, String(autoSync))
    } catch {
      // ignore
    }
  }, [autoSync, sessionId])

  useEffect(() => {
    try {
      const key = `story_auto_review_${sessionId}`
      const raw = localStorage.getItem(key)
      if (raw === null) return
      setAutoReview(raw === 'true')
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    try {
      const key = `story_auto_review_${sessionId}`
      localStorage.setItem(key, String(autoReview))
    } catch {
      // ignore
    }
  }, [autoReview, sessionId])

  useEffect(() => {
    try {
      const key = `story_auto_writeback_${sessionId}`
      const raw = localStorage.getItem(key)
      if (raw === null) return
      setAutoWriteback(raw === 'true')
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    try {
      const key = `story_auto_writeback_${sessionId}`
      localStorage.setItem(key, String(autoWriteback))
    } catch {
      // ignore
    }
  }, [autoWriteback, sessionId])

  const sceneJobEnabled = Boolean(session?.scene_image_job_id) && !session?.scene_image
  const sceneImageJob = useJob(session?.scene_image_job_id || null, {
    enabled: sceneJobEnabled,
    refetchIntervalMs: 2500,
  })

  useEffect(() => {
    const serverTurnJobId = String((session as any)?.turn_job_id || '').trim()
    if (!serverTurnJobId) return
    if (turnJobId) return
    setTurnJobId(serverTurnJobId)
  }, [session, turnJobId])

  const turnJob = useJob(turnJobId, { enabled: Boolean(turnJobId), refetchIntervalMs: 1500 })
  const turnJobRunning = useMemo(() => {
    if (!turnJobId) return false
    const status = String(turnJob.job?.status || '').toLowerCase()
    return status !== 'completed' && status !== 'failed' && status !== 'cancelled'
  }, [turnJob.job?.status, turnJobId])

  const isExecuting = isSubmittingTurn || turnJobRunning

  const autoReviewLastTurnRef = useRef<number | null>(null)
  const autoWritebackLastTurnRef = useRef<number | null>(null)
  useEffect(() => {
    if (!sessionId) return
    const history = (session as any)?.turn_history
    if (!Array.isArray(history) || history.length === 0) return

    const last = history[history.length - 1]
    const lastTurn = Number(last?.turn ?? -1)
    if (!Number.isFinite(lastTurn) || lastTurn < 0) return

    // Keep marker aligned while disabled.
    if (!autoReview) {
      autoReviewLastTurnRef.current = lastTurn
      return
    }

    // On first enable, do not retroactively enqueue old turns.
    if (autoReviewLastTurnRef.current === null) {
      autoReviewLastTurnRef.current = lastTurn
      return
    }

    // Only process newly arrived turns.
    if (lastTurn <= autoReviewLastTurnRef.current) return
    autoReviewLastTurnRef.current = lastTurn

    const artifacts = (last as any)?.artifacts || null
    const diff = (artifacts as any)?.diff || (last as any)?.state_delta || null
    if (!diff) return

    const flags = Array.isArray(diff?.flags) ? diff.flags : []
    const stats = Array.isArray(diff?.stats) ? diff.stats : []
    const invAdded = Array.isArray(diff?.inventory?.added) ? diff.inventory.added : []
    const invRemoved = Array.isArray(diff?.inventory?.removed) ? diff.inventory.removed : []
    const rels = Array.isArray(diff?.relationships) ? diff.relationships : []

    const hasMajorFlag = flags.some((f: any) => {
      const key = String(f?.key || '').trim()
      if (!key) return false
      return key.startsWith('quest_') || key.startsWith('achievement_')
    })
    const hasLevelUp = stats.some((s: any) => {
      const key = String(s?.key || '').trim()
      if (key !== 'level') return false
      const change = Number(s?.change ?? 0)
      return Number.isFinite(change) && change !== 0
    })

    const score = flags.length + stats.length + invAdded.length + invRemoved.length + rels.length
    const isMajor = hasMajorFlag || hasLevelUp || score >= 6
    if (!isMajor) return

    try {
      const existing = loadReviewQueue(sessionId)
      const exists = existing.some((q) => q.kind === 'state_delta' && Number((q as any).turn) === lastTurn)
      if (exists) return

      const item: ReviewQueueItem = {
        kind: 'state_delta',
        id: `sd_${lastTurn}_${Date.now()}`,
        created_at: new Date().toISOString(),
        status: 'pending',
        world_id: String(session?.world_id || '').trim() || 'default',
        turn: lastTurn,
        artifacts: (artifacts as any) || { diff },
      }

      enqueueReviewQueueItem(sessionId, item, 20)
      addNotification({ type: 'success', title: '重大變更已加入審核佇列', message: `Turn ${lastTurn + 1}` })
    } catch {
      // ignore
    }
  }, [addNotification, autoReview, session, sessionId])

  useEffect(() => {
    if (!sessionId) return
    const history = (session as any)?.turn_history
    if (!Array.isArray(history) || history.length === 0) return

    const last = history[history.length - 1]
    const lastTurn = Number(last?.turn ?? -1)
    if (!Number.isFinite(lastTurn) || lastTurn < 0) return

    // Keep marker aligned while disabled.
    if (!autoWriteback) {
      autoWritebackLastTurnRef.current = lastTurn
      return
    }

    // On first enable, do not retroactively enqueue old turns.
    if (autoWritebackLastTurnRef.current === null) {
      autoWritebackLastTurnRef.current = lastTurn
      return
    }

    // Only process newly arrived turns.
    if (lastTurn <= autoWritebackLastTurnRef.current) return
    autoWritebackLastTurnRef.current = lastTurn

    const artifacts = (last as any)?.artifacts || null
    const diff = (artifacts as any)?.diff || (last as any)?.state_delta || null
    if (!diff) return

    const flags = Array.isArray(diff?.flags) ? diff.flags : []
    const stats = Array.isArray(diff?.stats) ? diff.stats : []
    const invAdded = Array.isArray(diff?.inventory?.added) ? diff.inventory.added : []
    const invRemoved = Array.isArray(diff?.inventory?.removed) ? diff.inventory.removed : []
    const rels = Array.isArray(diff?.relationships) ? diff.relationships : []

    const hasMajorFlag = flags.some((f: any) => {
      const key = String(f?.key || '').trim()
      if (!key) return false
      return key.startsWith('quest_') || key.startsWith('achievement_')
    })
    const hasLevelUp = stats.some((s: any) => {
      const key = String(s?.key || '').trim()
      if (key !== 'level') return false
      const change = Number(s?.change ?? 0)
      return Number.isFinite(change) && change !== 0
    })

    const score = flags.length + stats.length + invAdded.length + invRemoved.length + rels.length
    const isMajor = hasMajorFlag || hasLevelUp || score >= 6
    if (!isMajor) return

    if (writebackSuggest.isPending) return

    try {
      const existing = loadReviewQueue(sessionId)
      const exists = existing.some(
        (q) =>
          q.kind === 'world_writeback' &&
          (Number((q as any)?.turn) === lastTurn || String(q.id || '').startsWith(`wb_auto_${lastTurn}_`))
      )
      if (exists) return
    } catch {
      // ignore
    }

    ;(async () => {
      try {
        const res = await writebackSuggest.mutateAsync({
          sessionId,
          request: {
            include_flags: true,
            include_characters: true,
            include_rag_note: true,
            max_new_characters: 10,
          },
        })
        const item: ReviewQueueItem = {
          kind: 'world_writeback',
          id: `wb_auto_${lastTurn}_${Date.now()}`,
          created_at: new Date().toISOString(),
          status: 'pending',
          world_id: String(session?.world_id || '').trim() || 'default',
          turn: lastTurn,
          selection: {
            world_flags: true,
            characters: true,
            rag_note: true,
          },
          response: res as any,
        }
        enqueueReviewQueueItem(sessionId, item, 20)
        addNotification({
          type: 'success',
          title: '已產生世界回寫建議（待審核）',
          message: `flags+${(res as any)?.summary?.flags_added ?? 0} / chars+${(res as any)?.summary?.characters_added ?? 0}`,
        })
      } catch (err) {
        addNotification({
          type: 'error',
          title: '自動回寫建議失敗',
          message: err instanceof Error ? err.message : '未知錯誤',
        })
      }
    })()
  }, [addNotification, autoWriteback, session, sessionId, writebackSuggest])

  useEffect(() => {
    if (!sceneJobEnabled) return
    const status = String(sceneImageJob.job?.status || '')
    if (status === 'completed') {
      void refetch()
    }
  }, [refetch, sceneImageJob.job?.status, sceneJobEnabled])

  const worldSyncedStatus = useMemo(() => {
    const worldUpdatedAt = world?.updated_at
    const appliedAt = session?.worldpack_updated_at
    if (!worldUpdatedAt || !appliedAt) return null
    return String(worldUpdatedAt) === String(appliedAt)
  }, [session?.worldpack_updated_at, world?.updated_at])

  const handleSyncWorldpack = async () => {
    if (!session) return
    try {
      const res = await syncWorldpack.mutateAsync({ sessionId, request: { mode: 'add_only' } })
      addNotification({
        type: 'success',
        title: '世界已同步到本故事',
        message: `flags_added=${res.flags_added.length} / characters_added=${res.characters_added.length}`,
      })
      await refetch()
    } catch (error) {
      addNotification({
        type: 'error',
        title: '同步世界失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  const timelineTurns = useMemo(() => {
    const history = session?.turn_history || []
    if (history && history.length > 0) {
      return history.map((h) => ({
        turn_index: Number(h.turn ?? 0),
        turn: Number(h.turn ?? 0) + 1,
        timestamp: h.timestamp ?? null,
        action: h.player_input,
        result: h.ai_response,
        scene: h.scene_id ?? null,
        choice_id: h.choice_id ?? null,
        enriched_player_input: h.enriched_player_input ?? null,
        rag_mode: h.rag_mode ?? null,
        rerank_mode: h.rerank_mode ?? null,
        rag_query: h.rag_query ?? null,
        knowledge_used: h.knowledge_used ?? null,
        agent_overlay: h.agent_overlay ?? null,
        agent_actions: h.agent_actions ?? null,
        state_delta: h.state_delta ?? null,
        scene_image: h.scene_image ?? null,
        artifacts: h.artifacts ?? null,
        raw: h,
      }))
    }

    const shortTerm = session?.memory_context?.short_term || []
    if (shortTerm && shortTerm.length > 0) {
      return shortTerm.map((t) => ({
        turn_index: Math.max(0, Number(t.turn ?? 0) - 1),
        turn: Number(t.turn ?? 0),
        timestamp: null,
        action: t.action,
        result: t.result,
        scene: t.scene ?? null,
        choice_id: null,
        enriched_player_input: null,
        rag_mode: null,
        rag_query: null,
        knowledge_used: null,
        agent_overlay: null,
        agent_actions: null,
        state_delta: null,
        scene_image: null,
        artifacts: null,
        raw: null,
      }))
    }

    return []
  }, [session?.turn_history, session?.memory_context?.short_term])

  const handlePlayerInput = async (
    input: string,
    options?: {
      choiceId?: string
      ragMode?: RagMode
      rerankMode?: RagMode
      useAgent?: boolean
      includeImage?: boolean
      asyncTurn?: boolean
    }
  ) => {
    if (!session) return

    setIsSubmittingTurn(true)
    try {
      const request = {
        session_id: sessionId,
        player_input: input,
        ...(options?.choiceId ? { choice_id: options.choiceId } : {}),
        ...(options?.ragMode ? { rag_mode: options.ragMode } : {}),
        ...(options?.rerankMode ? { rerank_mode: options.rerankMode } : {}),
        ...(typeof options?.useAgent === 'boolean' ? { use_agent: options.useAgent } : {}),
        ...(typeof options?.includeImage === 'boolean' ? { include_image: options.includeImage } : {}),
      }

      if (options?.asyncTurn) {
        const resp = await executeTurnJob.mutateAsync(request)
        const jobId = String((resp as any)?.job_id || '').trim()
        if (!jobId) throw new Error('缺少 job_id（伺服器未回傳）')
        setTurnJobId(jobId)
        addNotification({ type: 'success', title: '已建立回合任務', message: `job_id: ${jobId}` })
      } else {
        await executeTurn.mutateAsync(request)
        addNotification({ type: 'success', title: '回合執行成功' })
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: '回合執行失敗',
        message: error instanceof Error ? error.message : '發生未知錯誤',
      })
    } finally {
      setIsSubmittingTurn(false)
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

  if (isVisualNovelMode) {
    return (
      <VisualNovelScreen
        session={session}
        sessionId={sessionId}
        isExecuting={isExecuting}
        timelineTurns={timelineTurns}
        onSubmit={handlePlayerInput}
        onRefetch={refetch}
        world={world}
        persona={persona}
        storyContext={storyContext}
        contextLoading={contextLoading}
        contextError={contextError}
        onToggleClassic={() => setIsVisualNovelMode(false)}
      />
    )
  }

  return (
      <div className="flex flex-col lg:flex-row min-h-screen lg:h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* 左側：場景圖像 + 記憶系統 */}
      <div className="w-full lg:w-96 p-6 border-b lg:border-b-0 lg:border-r border-slate-700 overflow-y-auto space-y-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-semibold text-white">介面模式</h2>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setIsVisualNovelMode(true)}
            className="gap-2"
          >
            <Tv className="w-4 h-4" />
            沉浸模式
          </Button>
        </div>
        {/* 場景圖像 */}
        <div>
          <h2 className="text-lg font-semibold text-white mb-2">場景圖像</h2>
	          {(() => {
	            if (session.scene_image) return <SceneVisualizer sceneImage={session.scene_image} showMetadata={true} />
	            if (isExecuting) return <SceneVisualizerSkeleton />

	            const status = String(sceneImageJob.job?.status || '')
	            if (status === 'failed' || status === 'cancelled') return <SceneVisualizerError />
	            if (session.scene_image_job_id) return <SceneVisualizerSkeleton />
	            return null
	          })()}
	        </div>
	        {sceneJobEnabled && session.scene_image_job_id && (
	          <JobProgressCard
	            title="場景圖像任務"
	            jobId={session.scene_image_job_id}
	            job={sceneImageJob.job}
	            isLoading={sceneImageJob.isLoading}
	            error={sceneImageJob.error}
	            cancelling={sceneImageJob.cancelJob.isPending}
	            onCancel={async () => {
	              await sceneImageJob.cancelJob.mutateAsync()
	              await refetch()
	            }}
	            onCompleted={() => void refetch()}
	          />
	        )}

        {/* 記憶狀態指示器 */}
        <div>
          <h2 className="text-lg font-semibold text-white mb-2">記憶狀態</h2>
          <MemoryIndicator memoryStats={session.memory_stats} />
        </div>

        {/* Agent 行動 */}
        {session.agent_actions && session.agent_actions.tool_results.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Agent 行動
            </h2>
            <Suspense fallback={<div className="text-sm text-slate-400">載入中...</div>}>
              <AgentActionsPanel agentActions={session.agent_actions} />
            </Suspense>
          </div>
        )}

        {/* 最近記憶 */}
        {session.memory_context && (
          <div>
            <h2 className="text-lg font-semibold text-white mb-2">最近記憶</h2>
            <Suspense fallback={<div className="text-sm text-slate-400">載入中...</div>}>
              <RecentMemories
                shortTerm={session.memory_context.short_term}
                summaries={session.memory_context.summaries}
              />
            </Suspense>
          </div>
        )}
      </div>

      {/* 中間：敘事區域 */}
      <div className="flex-1 flex flex-col p-6 lg:overflow-hidden">
        <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <h1 className="text-2xl font-bold text-white truncate">{session.player_name} 的冒險</h1>
            <p className="text-sm text-slate-400">回合 {session.turn_count}</p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                世界：{world?.name || session.world_id} ({session.world_id})
              </Badge>
              <Badge variant="outline" className="text-xs">
                玩家風格：{playerTemplateName || session.player_template_id || '未設定'}
              </Badge>
              <Badge variant="outline" className="text-xs">
                敘事者：{persona?.name || session.persona_id || '—'} ({session.persona_id || '—'})
              </Badge>
              <Badge variant="outline" className="text-xs">
                Preset：{session.runtime_preset_id || 'auto'}
              </Badge>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <div className="flex flex-wrap items-center gap-2">
                {world?.updated_at && (
                  <Badge variant="outline" className="text-xs">
                    世界更新：{new Date(world.updated_at).toLocaleString()}
                  </Badge>
                )}
                {session.worldpack_updated_at && (
                  <Badge variant="outline" className="text-xs">
                    已套用：{new Date(session.worldpack_updated_at).toLocaleString()}
                  </Badge>
                )}
                {worldSyncedStatus !== null && (
                  <Badge variant={worldSyncedStatus ? 'default' : 'destructive'} className="text-xs">
                    {worldSyncedStatus ? '已同步' : '需同步'}
                  </Badge>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <Switch checked={autoSync} onCheckedChange={setAutoSync} label="自動同步" />
                <Switch checked={autoReview} onCheckedChange={setAutoReview} label="自動審核" />
                <Switch checked={autoWriteback} onCheckedChange={setAutoWriteback} label="自動回寫建議" />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void handleSyncWorldpack()}
                  disabled={syncWorldpack.isPending}
                >
                  {syncWorldpack.isPending ? '同步中...' : '同步世界'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setWorkbenchTab('world')
                    setWorkbenchOpen(true)
                  }}
                >
                  世界工作室
                </Button>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => setWorkbenchOpen(true)}>
              工作台
            </Button>
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
          {timelineTurns.length > 0 ? (
            <TurnTimeline sessionId={sessionId} turns={timelineTurns} isExecuting={isExecuting} />
          ) : (
            <NarrativeDisplay narrative={session.narrative || ''} choices={session.choices} />
          )}
        </div>

        {/* 玩家輸入 */}
        {turnJobId && (
          <div className="mb-4">
            <JobProgressCard
              title="本回合（Story Turn Job）"
              jobId={turnJobId}
              job={turnJob.job}
              isLoading={turnJob.isLoading}
              error={turnJob.error}
              cancelling={turnJob.cancelJob.isPending}
              onCancel={() => void turnJob.cancelJob.mutate()}
              onCompleted={() => {
                setTurnJobId(null)
                addNotification({ type: 'success', title: '回合任務完成' })
                void refetch()
              }}
              onFailed={(job) => {
                setTurnJobId(null)
                addNotification({
                  type: 'error',
                  title: '回合任務失敗',
                  message: String((job as any)?.error || 'unknown error'),
                })
              }}
              onCancelled={() => {
                setTurnJobId(null)
                addNotification({ type: 'error', title: '回合任務已取消' })
              }}
              hideWhenCompleted={true}
            />
          </div>
        )}
        <PlayerInput
          sessionId={sessionId}
          onSubmit={handlePlayerInput}
          isExecuting={isExecuting}
          choices={session.choices}
          worldId={session.world_id}
          ragAuto={session.rag_auto}
          ragMode={session.rag_mode}
          ragAvailable={session.rag_available}
          enrichWithRag={session.enrich_with_rag}
          ragNextTurn={session.rag_next_turn}
          rerankMode={session.rerank_mode}
          rerankNextTurn={session.rerank_next_turn}
          prefillText={prefillText}
          onPrefillApplied={() => setPrefillText(null)}
        />
      </div>

      {/* 右側：角色面板 */}
      <div className="w-full lg:w-80 p-6 bg-slate-900/50 border-t lg:border-t-0 lg:border-l border-slate-700 overflow-y-auto space-y-4">
        <CharacterSheet
          playerName={session.player_name}
          stats={session.stats}
          inventory={session.inventory}
          flags={session.flags}
        />
        <RelationshipPanel
          context={storyContext}
          isLoading={contextLoading}
          error={contextError}
          worldpack={world || null}
          inventory={session.inventory}
          turnHistory={session.turn_history || null}
          onQuickAction={(text) => setPrefillText(text)}
        />
        <QuestExplorationPanel flags={session.flags} worldpack={world || null} />
        <AgentProfilePanel sessionId={sessionId} lastAgentActions={session.agent_actions || null} />
        <ReviewQueuePanel sessionId={sessionId} worldId={session.world_id} worldpack={world || null} />
      </div>

      <StoryWorkbenchDialog
        open={workbenchOpen}
        onOpenChange={setWorkbenchOpen}
        sessionId={sessionId}
        worldId={session.world_id}
        tab={workbenchTab}
        onTabChange={setWorkbenchTab}
      />
    </div>
  )
}
