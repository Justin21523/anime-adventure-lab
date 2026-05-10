import { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { useUiStore } from '@/stores/uiStore'
import { apiPost } from '@/api/client'
import { useUpdateWorld } from '@/features/worlds/hooks/useWorldMutations'
import { useStoryWorldSync } from '../hooks/useStoryWorldSync'
import { useStoryWorldWritebackSuggest } from '../hooks/useStoryWorldWritebackSuggest'
import type { WorldPack } from '@/features/worlds/types/world.types'
import type { RAGAddDocumentResponse } from '@/features/rag/types/rag.types'
import type { StoryWorldWritebackSuggestRequest, StoryWorldWritebackSuggestResponse } from '../types/story.types'
import { WritebackReviewDialog } from './WritebackReviewDialog'
import { WorldAiReviewDialog } from './WorldAiReviewDialog'
import { StateDeltaReviewDialog } from './StateDeltaReviewDialog'
import type { ReviewQueueItem, WritebackApplySelection } from '../types/reviewQueue.types'
import { loadReviewQueue, saveReviewQueue } from '../lib/reviewQueueStorage'
import { applyWorldPackSelection, computeWorldPackDiff } from '@/features/worlds/utils/worldPackPatch'

function applyWritebackPatch(base: WorldPack, response: StoryWorldWritebackSuggestResponse, selection: WritebackApplySelection): WorldPack {
  const next = JSON.parse(JSON.stringify(base)) as WorldPack
  const patch = (response.patch || {}) as any

  if (selection.world_flags && patch.world_flags && typeof patch.world_flags === 'object') {
    const merged = { ...(next.world_flags || {}) }
    for (const [k, v] of Object.entries(patch.world_flags)) {
      const key = String(k || '').trim()
      if (!key) continue
      merged[key] = Boolean(v)
    }
    next.world_flags = merged
  }

  if (selection.characters && Array.isArray(patch.characters)) {
    const byId = new Map<string, any>()
    for (const c of next.characters || []) {
      const cid = String((c as any)?.character_id || '').trim()
      if (!cid) continue
      byId.set(cid, c)
    }
    for (const c of patch.characters) {
      const cid = String((c as any)?.character_id || '').trim()
      if (!cid) continue
      if (byId.has(cid)) continue
      byId.set(cid, c)
    }
    next.characters = Array.from(byId.values())
  }

  return next
}

interface ReviewQueuePanelProps {
  sessionId: string
  worldId: string
  worldpack?: WorldPack | null
}

export function ReviewQueuePanel({ sessionId, worldId, worldpack }: ReviewQueuePanelProps) {
  const { addNotification } = useUiStore()
  const writebackSuggest = useStoryWorldWritebackSuggest()
  const updateWorld = useUpdateWorld()
  const syncWorldpack = useStoryWorldSync()

  const [queue, setQueue] = useState<ReviewQueueItem[]>([])
  const [dialogOpen, setDialogOpen] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const [options, setOptions] = useState<StoryWorldWritebackSuggestRequest>({
    include_flags: true,
    include_characters: true,
    include_rag_note: true,
    max_new_characters: 10,
  })

  useEffect(() => {
    try {
      setQueue(loadReviewQueue(sessionId))
    } catch {
      setQueue([])
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return

    const handleUpdate = (evt: Event) => {
      try {
        const detail = (evt as any)?.detail || {}
        const target = String(detail.sessionId || '').trim()
        if (target && target !== sessionId) return
      } catch {
        // ignore
      }
      try {
        setQueue(loadReviewQueue(sessionId))
      } catch {
        // ignore
      }
    }

    window.addEventListener('reviewqueue:update', handleUpdate as any)
    return () => window.removeEventListener('reviewqueue:update', handleUpdate as any)
  }, [sessionId])

  useEffect(() => {
    saveReviewQueue(sessionId, queue, 20)
  }, [queue, sessionId])

  const selected = useMemo(() => queue.find((q) => q.id === selectedId) || null, [queue, selectedId])
  const selectedWriteback = selected?.kind === 'world_writeback' ? selected : null
  const selectedWorldAi = selected?.kind === 'world_ai' ? selected : null
  const selectedStateDelta = selected?.kind === 'state_delta' ? selected : null

  const busy = writebackSuggest.isPending || updateWorld.isPending || syncWorldpack.isPending

  const handleSuggest = async () => {
    try {
      const res = await writebackSuggest.mutateAsync({ sessionId, request: options })
      const item: ReviewQueueItem = {
        kind: 'world_writeback',
        id: `wb_${Date.now()}`,
        created_at: new Date().toISOString(),
        status: 'pending',
        world_id: worldId,
        selection: {
          world_flags: Boolean(options.include_flags),
          characters: Boolean(options.include_characters),
          rag_note: Boolean(options.include_rag_note),
        },
        response: res,
      }
      setQueue((prev) => [item, ...prev].slice(0, 20))
      setSelectedId(item.id)
      setDialogOpen(true)
      addNotification({
        type: 'success',
        title: '已生成回寫建議',
        message: `flags+${(res.summary as any)?.flags_added ?? 0} / chars+${(res.summary as any)?.characters_added ?? 0}`,
      })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '生成回寫建議失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const updateWritebackSelection = (itemId: string, selection: WritebackApplySelection) => {
    setQueue((prev) => prev.map((q) => (q.id === itemId && q.kind === 'world_writeback' ? { ...q, selection } : q)))
  }

  const updateWorldAiSelection = (itemId: string, selection: any) => {
    setQueue((prev) => prev.map((q) => (q.id === itemId && q.kind === 'world_ai' ? { ...q, selection } : q)))
  }

  const removeItem = (itemId: string) => {
    setQueue((prev) => prev.filter((q) => q.id !== itemId))
    if (selectedId === itemId) {
      setSelectedId(null)
      setDialogOpen(false)
    }
  }

  const applyWritebackToWorld = async (
    item: Extract<ReviewQueueItem, { kind: 'world_writeback' }>,
    syncAfter: boolean
  ) => {
    if (!worldpack) {
      addNotification({ type: 'error', title: '世界資料尚未載入', message: '請稍後再試' })
      return
    }

    const nextWorld = applyWritebackPatch(worldpack, item.response, item.selection)
    try {
      await updateWorld.mutateAsync({ worldId, world: nextWorld })
      addNotification({ type: 'success', title: '已保存到世界' })

      if (syncAfter) {
        await syncWorldpack.mutateAsync({ sessionId, request: { mode: 'add_only' } })
        addNotification({ type: 'success', title: '已同步到本故事' })
      }

      setQueue((prev) =>
        prev.map((q) =>
          q.id === item.id
            ? { ...q, status: 'applied', applied_at: new Date().toISOString() }
            : q
        )
      )
    } catch (err) {
      addNotification({
        type: 'error',
        title: '保存/同步失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const applyWorldAiToWorld = async (item: Extract<ReviewQueueItem, { kind: 'world_ai' }>, syncAfter: boolean) => {
    if (!worldpack) {
      addNotification({ type: 'error', title: '世界資料尚未載入', message: '請稍後再試' })
      return
    }
    const candidate = (item.response as any)?.worldpack
    if (!candidate) {
      addNotification({ type: 'error', title: '此項目缺少 worldpack', message: 'response.worldpack is missing' })
      return
    }

    const nextWorld = applyWorldPackSelection(worldpack, candidate, item.selection as any)
    try {
      await updateWorld.mutateAsync({ worldId, world: nextWorld })
      addNotification({ type: 'success', title: '已保存到世界' })

      if (syncAfter) {
        await syncWorldpack.mutateAsync({ sessionId, request: { mode: 'add_only' } })
        addNotification({ type: 'success', title: '已同步到本故事' })
      }

      setQueue((prev) =>
        prev.map((q) => (q.id === item.id ? { ...q, status: 'applied', applied_at: new Date().toISOString() } : q))
      )
    } catch (err) {
      addNotification({
        type: 'error',
        title: '保存/同步失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const markStateDeltaReviewed = (itemId: string) => {
    setQueue((prev) =>
      prev.map((q) =>
        q.id === itemId && q.kind === 'state_delta'
          ? { ...q, status: 'applied', applied_at: new Date().toISOString() }
          : q
      )
    )
  }

  const addRagNote = async (item: Extract<ReviewQueueItem, { kind: 'world_writeback' }>) => {
    if (!item.selection.rag_note) {
      addNotification({ type: 'error', title: '未允許寫入 RAG note' })
      return
    }
    const note = item.response.rag_note
    if (!note) {
      addNotification({ type: 'error', title: '此建議沒有 rag_note' })
      return
    }
    const ok = window.confirm(`確認要把本回寫摘要寫入 RAG 知識庫？\\nworld_id=${worldId}`)
    if (!ok) return

    const docId = `story_writeback_${worldId}_${sessionId}_${Date.now()}`
    try {
      const res = await apiPost<RAGAddDocumentResponse, any>('/rag/add', {
        doc_id: docId,
        content: note,
        metadata: {
          world_id: worldId,
          title: `Story writeback ${sessionId}`.trim(),
          tags: ['story_writeback', 'world', worldId, sessionId],
        },
      })
      addNotification({
        type: 'success',
        title: '已寫入 RAG',
        message: `doc_id=${res.doc_id}`,
      })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '寫入 RAG 失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  return (
    <Card className="bg-slate-800/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-lg">審核佇列（世界/故事）</CardTitle>
          <Badge variant="outline" className="text-xs">
            {queue.length}
          </Badge>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          收集世界 AI patch / 世界回寫 / 重大變更（需你確認後才會保存/同步/寫入 RAG）
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3 space-y-3">
          <div className="text-xs font-semibold text-slate-300">世界回寫（從故事匯出）</div>
          <div className="flex flex-wrap gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={Boolean(options.include_flags)}
                onCheckedChange={(v) => setOptions((p) => ({ ...p, include_flags: Boolean(v) }))}
              />
              <span className="text-xs text-slate-200">flags</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={Boolean(options.include_characters)}
                onCheckedChange={(v) => setOptions((p) => ({ ...p, include_characters: Boolean(v) }))}
              />
              <span className="text-xs text-slate-200">characters</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={Boolean(options.include_rag_note)}
                onCheckedChange={(v) => setOptions((p) => ({ ...p, include_rag_note: Boolean(v) }))}
              />
              <span className="text-xs text-slate-200">rag_note</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">max_new_characters</span>
              <Input
                className="h-8 w-20"
                type="number"
                min={0}
                max={50}
                value={Number(options.max_new_characters ?? 10)}
                onChange={(e) => setOptions((p) => ({ ...p, max_new_characters: Number(e.target.value || 0) }))}
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => void handleSuggest()} disabled={busy}>
              {writebackSuggest.isPending ? '生成中...' : '生成回寫建議'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const ok = window.confirm('確認要清空此故事的審核佇列？')
                if (!ok) return
                setQueue([])
              }}
              disabled={busy || queue.length === 0}
            >
              清空佇列
            </Button>
          </div>
        </div>

        {queue.length === 0 ? (
          <div className="text-sm text-slate-500">目前沒有待審核項目</div>
        ) : (
          <div className="space-y-3">
            {queue.slice(0, 8).map((q) => {
              let kindLabel = q.kind
              let detailBadges: Array<{ key: string; text: string }> = []

              if (q.kind === 'world_writeback') {
                kindLabel = 'world_writeback'
                const patchFlags = (((q.response as any)?.patch as any)?.world_flags as Record<string, any>) || {}
                const flags = patchFlags && typeof patchFlags === 'object' ? Object.keys(patchFlags).length : 0
                const chars = Array.isArray(((q.response as any)?.patch as any)?.characters)
                  ? (((q.response as any).patch as any).characters as any[]).length
                  : 0
                detailBadges = [
                  ...(q.world_id ? ([{ key: 'world_id', text: `world ${q.world_id}` }] as any) : []),
                  ...(typeof (q as any).turn === 'number' ? ([{ key: 'turn', text: `turn ${(q as any).turn + 1}` }] as any) : []),
                  { key: 'flags', text: `flags +${flags}` },
                  { key: 'chars', text: `chars +${chars}` },
                  ...(((q.response as any)?.rag_note ? [{ key: 'rag', text: 'RAG note' }] : []) as any),
                ]
              } else if (q.kind === 'world_ai') {
                kindLabel = 'world_ai'
                try {
                  const base = worldpack
                  const cand = (q.response as any)?.worldpack
                  if (base && cand) {
                    const diff = computeWorldPackDiff(base, cand)
                    detailBadges = [
                      ...(q.world_id ? ([{ key: 'world_id', text: `world ${q.world_id}` }] as any) : []),
                      { key: 'world', text: `world ${diff.worldFields.length}` },
                      { key: 'flags', text: `flags ${diff.worldFlagsChanged.length}` },
                      { key: 'chars', text: `chars +${diff.charactersAdded.length}/~${diff.charactersUpdated.length}` },
                    ]
                  }
                } catch {
                  detailBadges = []
                }
              } else if (q.kind === 'state_delta') {
                kindLabel = 'state_delta'
                const diff = (q as any)?.artifacts?.diff || {}
                const flagsCount = Array.isArray(diff?.flags) ? diff.flags.length : 0
                const statsCount = Array.isArray(diff?.stats) ? diff.stats.length : 0
                detailBadges = [
                  ...(q.world_id ? ([{ key: 'world_id', text: `world ${q.world_id}` }] as any) : []),
                  {
                    key: 'turn',
                    text: `turn ${typeof (q as any)?.turn === 'number' ? (q as any).turn + 1 : '—'}`,
                  },
                  { key: 'flags', text: `flags ${flagsCount}` },
                  { key: 'stats', text: `stats ${statsCount}` },
                ]
              }
              return (
                <div key={q.id} className="p-3 rounded-lg border border-slate-700 bg-slate-900/40">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-semibold text-slate-100 truncate">
                        {q.status === 'applied' ? '已套用' : '待審核'} • {kindLabel} • {new Date(q.created_at).toLocaleString()}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {detailBadges.map((b) => (
                          <Badge key={b.key} variant="outline" className="text-xs">
                            {b.text}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => {
                          setSelectedId(q.id)
                          setDialogOpen(true)
                        }}
                      >
                        檢視
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => removeItem(q.id)}
                      >
                        移除
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}
            {queue.length > 8 && <div className="text-xs text-slate-500">顯示前 8 筆（共 {queue.length}）</div>}
          </div>
        )}
      </CardContent>

      <WritebackReviewDialog
        open={dialogOpen && selectedWriteback?.kind === 'world_writeback'}
        onOpenChange={setDialogOpen}
        item={selectedWriteback}
        onUpdateSelection={updateWritebackSelection}
        onApplyToWorld={applyWritebackToWorld}
        onAddRagNote={addRagNote}
        onRemove={removeItem}
        busy={busy}
      />

      <WorldAiReviewDialog
        open={dialogOpen && selectedWorldAi?.kind === 'world_ai'}
        onOpenChange={setDialogOpen}
        item={selectedWorldAi}
        worldpack={worldpack || null}
        onUpdateSelection={updateWorldAiSelection}
        onApplyToWorld={applyWorldAiToWorld}
        onRemove={removeItem}
        busy={busy}
      />

      <StateDeltaReviewDialog
        open={dialogOpen && selectedStateDelta?.kind === 'state_delta'}
        onOpenChange={setDialogOpen}
        item={selectedStateDelta}
        onMarkReviewed={markStateDeltaReviewed}
        onRemove={removeItem}
        busy={busy}
      />
    </Card>
  )
}
