import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { SceneVisualizer } from './SceneVisualizer'
import { AgentActionsPanel } from './AgentActionsPanel'
import type { TimelineTurn } from '../types/timeline.types'
import { enqueueReviewQueueItem } from '../lib/reviewQueueStorage'
import type { ReviewQueueItem } from '../types/reviewQueue.types'
import { useUiStore } from '@/stores/uiStore'
import { useJob } from '@/features/jobs/hooks/useJob'

interface TurnInspectorDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  turn: TimelineTurn | null
  sessionId?: string
}

export function TurnInspectorDialog({ open, onOpenChange, turn, sessionId }: TurnInspectorDialogProps) {
  const { addNotification } = useUiStore()
  const title = turn ? `Turn ${turn.turn} 回合檢視器` : '回合檢視器'
  const sceneLabel = turn?.scene ? `Scene: ${turn.scene}` : 'Scene: —'
  const timeLabel = turn?.timestamp ? new Date(turn.timestamp).toLocaleString() : '—'

  const artifacts = turn?.artifacts || null
  const ragBucket = (artifacts as any)?.rag || null
  const agentsBucket = (artifacts as any)?.agents || null
  const t2iBucket = (artifacts as any)?.t2i || null
  const jobBucket = (artifacts as any)?.job || null

  const ragHits = (Array.isArray(ragBucket?.hits) ? ragBucket.hits : [])?.filter(Boolean) || []
  const ragMode = ragBucket?.mode ?? null
  const rerankMode = ragBucket?.rerank_mode ?? null
  const ragQuery = ragBucket?.query ?? null

  const agentActions = agentsBucket?.actions || null
  const stateDelta = (artifacts as any)?.diff || null
  const image = t2iBucket?.scene_image || null
  const sceneImageJobId = String(t2iBucket?.scene_image_job_id || '').trim() || null
  const sceneImageJob = useJob(sceneImageJobId, {
    enabled: Boolean(sceneImageJobId) && !image,
    refetchIntervalMs: 2500,
  })

  const storyPipelineEvents = (Array.isArray(jobBucket?.stage_events) ? jobBucket.stage_events : [])?.filter(Boolean) || []
  const storyPipelineStartedAt = String(jobBucket?.started_at || '').trim() || null

  const t2iPipelineBucket =
    (t2iBucket as any)?.job ||
    (sceneImageJob.job
      ? {
          job_id: (sceneImageJob.job as any)?.job_id,
          job_type: (sceneImageJob.job as any)?.job_type,
          status: (sceneImageJob.job as any)?.status,
          stage: (sceneImageJob.job as any)?.stage,
          stage_message: (sceneImageJob.job as any)?.stage_message,
          progress: (sceneImageJob.job as any)?.progress,
          started_at: (sceneImageJob.job as any)?.started_at || (sceneImageJob.job as any)?.created_at,
          duration_seconds: (sceneImageJob.job as any)?.duration_seconds,
          stage_events: (sceneImageJob.job as any)?.stage_events,
        }
      : null)

  const t2iPipelineEvents =
    (Array.isArray((t2iPipelineBucket as any)?.stage_events) ? (t2iPipelineBucket as any)?.stage_events : [])?.filter(Boolean) || []
  const t2iPipelineStartedAt = String((t2iPipelineBucket as any)?.started_at || '').trim() || null

  const ragRebuildBucket =
    (ragBucket as any)?.rebuild_job || (ragBucket as any)?.maintenance?.rebuild_job || null
  const ragRebuildEvents =
    (Array.isArray((ragRebuildBucket as any)?.stage_events) ? (ragRebuildBucket as any)?.stage_events : [])?.filter(Boolean) || []
  const ragRebuildStartedAt = String((ragRebuildBucket as any)?.started_at || '').trim() || null
  const canEnqueueStateDelta = Boolean(sessionId && turn && artifacts && stateDelta)

  const handleEnqueueStateDelta = () => {
    if (!sessionId || !turn || !artifacts) {
      addNotification({ type: 'error', title: '缺少 session_id', message: '請在 Story 內使用此功能' })
      return
    }
    if (!stateDelta) {
      addNotification({ type: 'error', title: '本回合沒有狀態 diff', message: '沒有可加入審核佇列的 state_delta' })
      return
    }

    const item: ReviewQueueItem = {
      kind: 'state_delta',
      id: `sd_${Date.now()}`,
      created_at: new Date().toISOString(),
      status: 'pending',
      world_id: String((artifacts as any)?.world?.world_id || '').trim() || null,
      turn: Number(turn.turn_index ?? 0),
      artifacts: (artifacts as any) || {},
    }
    enqueueReviewQueueItem(sessionId, item, 20)
    addNotification({ type: 'success', title: '已加入審核佇列', message: `Turn ${item.turn + 1}` })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl w-[95vw] h-[90vh] p-0 overflow-hidden">
        <div className="flex flex-col h-full">
          <div className="p-6 border-b border-slate-700">
            <DialogHeader>
              <DialogTitle>{title}</DialogTitle>
              <DialogDescription>
                {sceneLabel} • {timeLabel}
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {!turn ? (
              <Card className="bg-slate-900/40 border-slate-700">
                <CardContent className="py-10 text-center text-slate-400">未選擇回合</CardContent>
              </Card>
            ) : (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
	                      <div className="flex items-center justify-between gap-2">
	                        <CardTitle className="text-lg">玩家輸入</CardTitle>
	                        <div className="flex flex-wrap gap-2">
	                        {ragMode && (
	                          <Badge variant="outline" className="text-xs">
	                            RAG: {ragMode}
	                          </Badge>
	                        )}
	                        {rerankMode && (
	                          <Badge variant="outline" className="text-xs">
	                            Rerank: {rerankMode}
	                          </Badge>
	                        )}
	                        {agentActions?.contributors?.length ? (
	                          <Badge variant="secondary" className="text-xs">
                            {agentActions.contributors.length} agents
                          </Badge>
                        ) : null}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="text-xs text-slate-500">原始輸入</div>
                      <Textarea value={turn.action} readOnly rows={5} className="font-mono text-xs" />
                      {turn.enriched_player_input && turn.enriched_player_input !== turn.action && (
                        <>
                          <div className="text-xs text-slate-500">實際送進引擎的提示（含 RAG / agent_hint）</div>
                          <Textarea
                            value={turn.enriched_player_input}
                            readOnly
                            rows={5}
                            className="font-mono text-xs"
                          />
                        </>
                      )}
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">敘事回應</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Textarea value={turn.result} readOnly rows={10} className="font-mono text-xs" />
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">RAG 命中</CardTitle>
                        <Badge variant="outline" className="text-xs">
                          {ragHits.length}
                        </Badge>
	                      </div>
	                      <p className="text-xs text-slate-500 mt-1">
	                        query: {ragQuery || '—'}
	                      </p>
	                    </CardHeader>
                    <CardContent className="space-y-3">
                      {ragHits.length === 0 ? (
                        <div className="text-sm text-slate-500">本回合沒有使用知識庫</div>
                      ) : (
                        ragHits.map((hit, idx) => {
                          const meta = (hit as any)?.metadata || {}
                          const tags = Array.isArray(meta.tags) ? meta.tags : []
                          const semantic = (hit as any)?.semantic_score
                          const bm25 = (hit as any)?.bm25_score
                          const combined = (hit as any)?.combined_score
                          const rerank = (hit as any)?.rerank_score
                          return (
                            <div key={idx} className="p-3 rounded-lg border border-slate-700 bg-slate-950/40">
                              <div className="flex flex-wrap items-center gap-2 mb-2">
                                <Badge variant="secondary" className="text-xs">
                                  #{idx + 1}
                                </Badge>
                                {typeof (hit as any)?.score === 'number' && (
                                  <Badge variant="outline" className="text-xs">
                                    score {(hit as any).score.toFixed(3)}
                                  </Badge>
                                )}
                                {typeof semantic === 'number' && (
                                  <Badge variant="outline" className="text-xs">
                                    sem {semantic.toFixed(3)}
                                  </Badge>
                                )}
                                {typeof bm25 === 'number' && (
                                  <Badge variant="outline" className="text-xs">
                                    bm25 {bm25.toFixed(3)}
                                  </Badge>
                                )}
                                {typeof combined === 'number' && (
                                  <Badge variant="outline" className="text-xs">
                                    comb {combined.toFixed(3)}
                                  </Badge>
                                )}
                                {typeof rerank === 'number' && (
                                  <Badge variant="outline" className="text-xs">
                                    rerank {rerank.toFixed(3)}
                                  </Badge>
                                )}
                                {meta.title && (
                                  <Badge variant="outline" className="text-xs">
                                    {meta.title}
                                  </Badge>
                                )}
                                {meta.parent_doc_id && (
                                  <Badge variant="outline" className="text-xs">
                                    doc {String(meta.parent_doc_id).slice(0, 12)}…
                                  </Badge>
                                )}
                                {tags.slice(0, 3).map((t: any) => (
                                  <Badge key={String(t)} variant="secondary" className="text-xs">
                                    {String(t)}
                                  </Badge>
                                ))}
                              </div>
                              <div className="text-sm text-slate-200 whitespace-pre-wrap break-words">
                                {(hit as any)?.content || ''}
                              </div>
                            </div>
                          )
                        })
                      )}
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">狀態 Diff</CardTitle>
                        <div className="flex flex-wrap gap-2">
                          {stateDelta?.flags?.length ? (
                            <Badge variant="outline" className="text-xs">
                              flags {stateDelta.flags.length}
                            </Badge>
                          ) : null}
                          {stateDelta?.stats?.length ? (
                            <Badge variant="outline" className="text-xs">
                              stats {stateDelta.stats.length}
                            </Badge>
                          ) : null}
                          {(stateDelta?.inventory?.added?.length || 0) + (stateDelta?.inventory?.removed?.length || 0) ? (
                            <Badge variant="outline" className="text-xs">
                              inv {(stateDelta?.inventory?.added?.length || 0) + (stateDelta?.inventory?.removed?.length || 0)}
                            </Badge>
                          ) : null}
                          {stateDelta?.relationships?.length ? (
                            <Badge variant="outline" className="text-xs">
                              rel {stateDelta.relationships.length}
                            </Badge>
                          ) : null}
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={handleEnqueueStateDelta}
                            disabled={!canEnqueueStateDelta}
                            title={canEnqueueStateDelta ? '加入 ReviewQueue 供後續審核' : '需要有 diff 才能加入審核佇列'}
                          >
                            加入審核佇列
                          </Button>
                        </div>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        由 turn 前後狀態計算（含 choice/agent 等變更）
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {!stateDelta ? (
                        <div className="text-sm text-slate-500">本回合未提供 diff（可能為舊存檔）</div>
                      ) : (
                        <>
                          {stateDelta.flags && stateDelta.flags.length > 0 && (
                            <DeltaList
                              title="Flags"
                              items={stateDelta.flags.map((f) => ({
                                key: f.key,
                                old: f.old,
                                new: f.new,
                              }))}
                            />
                          )}
                          {stateDelta.stats && stateDelta.stats.length > 0 && (
                            <DeltaList
                              title="Stats"
                              items={stateDelta.stats.map((s) => ({
                                key: s.key,
                                old: s.old,
                                new: s.new,
                                change: s.change ?? undefined,
                              }))}
                            />
                          )}
                          {stateDelta.inventory && (
                            <div className="space-y-2">
                              <div className="text-xs font-semibold text-slate-300">Inventory</div>
                              {(stateDelta.inventory.added || []).length > 0 && (
                                <div className="text-xs text-slate-200">
                                  + {(stateDelta.inventory.added || []).map((i) => `${i.item} x${i.count}`).join(', ')}
                                </div>
                              )}
                              {(stateDelta.inventory.removed || []).length > 0 && (
                                <div className="text-xs text-slate-200">
                                  - {(stateDelta.inventory.removed || []).map((i) => `${i.item} x${i.count}`).join(', ')}
                                </div>
                              )}
                              {(stateDelta.inventory.added || []).length === 0 && (stateDelta.inventory.removed || []).length === 0 && (
                                <div className="text-sm text-slate-500">無變更</div>
                              )}
                            </div>
                          )}
                          {stateDelta.relationships && stateDelta.relationships.length > 0 && (
                            <DeltaList
                              title="Relationships"
                              items={stateDelta.relationships.map((r) => ({
                                key: r.character_id,
                                old: r.old,
                                new: r.new,
                                change: r.change,
                              }))}
                            />
                          )}
                          {(!stateDelta.flags || stateDelta.flags.length === 0) &&
                            (!stateDelta.stats || stateDelta.stats.length === 0) &&
                            (stateDelta.inventory?.added?.length || 0) === 0 &&
                            (stateDelta.inventory?.removed?.length || 0) === 0 &&
                            (!stateDelta.relationships || stateDelta.relationships.length === 0) && (
                              <div className="text-sm text-slate-500">本回合沒有狀態變更</div>
                            )}
                        </>
                      )}
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">Agent 行動</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {agentActions ? (
                        <AgentActionsPanel agentActions={agentActions} />
                      ) : (
                        <div className="text-sm text-slate-500">本回合沒有 agent actions</div>
                      )}
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">圖像資訊</CardTitle>
                        {image ? (
                          <Badge variant="outline" className="text-xs">
                            {image.width}×{image.height}
                          </Badge>
                        ) : null}
                      </div>
                    </CardHeader>
                    <CardContent>
                      {image ? (
                        <SceneVisualizer sceneImage={image} showMetadata={true} />
                      ) : sceneImageJobId ? (
                        <div className="space-y-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant="outline" className="text-xs font-mono">
                              job {sceneImageJobId.slice(0, 10)}…
                            </Badge>
                            {sceneImageJob.job?.status ? (
                              <Badge variant="outline" className="text-xs">
                                {String(sceneImageJob.job.status)}
                              </Badge>
                            ) : null}
                          </div>
                          {typeof sceneImageJob.job?.progress === 'number' ? (
                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-500">progress</span>
                                <span className="text-slate-300">{Number(sceneImageJob.job.progress).toFixed(1)}%</span>
                              </div>
                              <Progress value={Number(sceneImageJob.job.progress)} max={100} />
                            </div>
                          ) : (
                            <div className="text-sm text-slate-500">圖像任務仍在排隊/執行中</div>
                          )}
                        </div>
                      ) : (
                        <div className="text-sm text-slate-500">本回合未生成圖像</div>
                      )}
                    </CardContent>
                  </Card>
                </div>

                <Card className="bg-slate-900/40 border-slate-700">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between gap-2">
                      <CardTitle className="text-lg">本回合流程</CardTitle>
                      {jobBucket?.job_id ? (
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant="outline" className="text-xs font-mono">
                            {String(jobBucket.job_id).slice(0, 10)}…
                          </Badge>
                          {typeof jobBucket?.duration_seconds === 'number' ? (
                            <Badge variant="outline" className="text-xs">
                              {Math.round(Number(jobBucket.duration_seconds))}s
                            </Badge>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                    {storyPipelineStartedAt ? (
                      <p className="text-xs text-slate-500 mt-1">started_at: {storyPipelineStartedAt}</p>
                    ) : null}
                  </CardHeader>
                  <CardContent>
                    {storyPipelineEvents.length || t2iPipelineEvents.length || ragRebuildEvents.length ? (
                      <div className="space-y-5">
                        {storyPipelineEvents.length ? (
                          <div className="space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="text-sm font-semibold text-slate-200">Story Turn</div>
                              {jobBucket?.job_id ? (
                                <Badge variant="outline" className="text-xs font-mono">
                                  {String(jobBucket.job_id).slice(0, 10)}…
                                </Badge>
                              ) : null}
                            </div>
                            {storyPipelineEvents.map((e: any, idx: number) => (
                              <PipelineEventRow
                                key={`story-${String(e?.stage || 'stage')}-${idx}`}
                                event={e}
                                baseTs={storyPipelineEvents[0]?.ts || storyPipelineStartedAt}
                              />
                            ))}
                          </div>
                        ) : null}

                        {t2iPipelineEvents.length ? (
                          <div className="space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="text-sm font-semibold text-slate-200">Scene Image</div>
                              {(t2iPipelineBucket as any)?.job_id ? (
                                <Badge variant="outline" className="text-xs font-mono">
                                  {String((t2iPipelineBucket as any).job_id).slice(0, 10)}…
                                </Badge>
                              ) : null}
                            </div>
                            {t2iPipelineEvents.map((e: any, idx: number) => (
                              <PipelineEventRow
                                key={`t2i-${String(e?.stage || 'stage')}-${idx}`}
                                event={e}
                                baseTs={t2iPipelineEvents[0]?.ts || t2iPipelineStartedAt}
                              />
                            ))}
                          </div>
                        ) : null}

                        {ragRebuildEvents.length ? (
                          <div className="space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="text-sm font-semibold text-slate-200">RAG Rebuild</div>
                              {(ragRebuildBucket as any)?.job_id ? (
                                <Badge variant="outline" className="text-xs font-mono">
                                  {String((ragRebuildBucket as any).job_id).slice(0, 10)}…
                                </Badge>
                              ) : null}
                            </div>
                            {ragRebuildEvents.map((e: any, idx: number) => (
                              <PipelineEventRow
                                key={`ragrebuild-${String(e?.stage || 'stage')}-${idx}`}
                                event={e}
                                baseTs={ragRebuildEvents[0]?.ts || ragRebuildStartedAt}
                              />
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <div className="text-sm text-slate-500">本回合沒有 pipeline artifacts（可能是舊存檔或非 job 模式）</div>
                    )}
                  </CardContent>
                </Card>

                <Card className="bg-slate-900/40 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Raw JSON</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Textarea
                      value={JSON.stringify(turn.raw || turn, null, 2)}
                      readOnly
                      rows={14}
                      className="font-mono text-xs"
                    />
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function PipelineEventRow({ event, baseTs }: { event: any; baseTs?: string | null }) {
  const ts = String(event?.ts || '').trim()
  const stage = String(event?.stage || '').trim()
  const msg = String(event?.message || '').trim()
  const progress = event?.progress

  const base = (() => {
    const raw = String(baseTs || '').trim()
    if (!raw) return null
    const ms = Date.parse(raw)
    return Number.isFinite(ms) ? ms : null
  })()

  const delta = (() => {
    if (!ts || base === null) return null
    const ms = Date.parse(ts)
    if (!Number.isFinite(ms)) return null
    return Math.max(0, (ms - base) / 1000)
  })()

  return (
    <div className="flex items-start justify-between gap-3 rounded-md border border-slate-800 bg-slate-950/20 p-3">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          {stage ? (
            <Badge variant="outline" className="text-[10px] font-mono">
              {stage}
            </Badge>
          ) : null}
          {delta !== null ? (
            <Badge variant="secondary" className="text-[10px] font-mono">
              +{delta.toFixed(1)}s
            </Badge>
          ) : null}
          {typeof progress === 'number' ? (
            <Badge variant="outline" className="text-[10px] font-mono">
              {Number(progress).toFixed(1)}%
            </Badge>
          ) : null}
        </div>
        <div className="text-sm text-slate-200 break-words">{msg || '—'}</div>
        {ts ? <div className="text-[11px] text-slate-500 mt-1">{ts}</div> : null}
      </div>
    </div>
  )
}

function DeltaList({
  title,
  items,
}: {
  title: string
  items: Array<{ key: string; old: any; new: any; change?: number }>
}) {
  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold text-slate-300">{title}</div>
      <div className="space-y-1">
        {items.slice(0, 20).map((it) => (
          <div key={it.key} className="flex items-start justify-between gap-2 text-xs">
            <div className="text-slate-300 min-w-0 flex-1 break-words">{it.key}</div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <Badge variant="outline" className="text-[10px] px-2 py-0.5">
                {String(it.old)}
              </Badge>
              <span className="text-slate-500">→</span>
              <Badge variant="outline" className="text-[10px] px-2 py-0.5">
                {String(it.new)}
              </Badge>
              {typeof it.change === 'number' && it.change !== 0 && (
                <Badge variant={it.change > 0 ? 'default' : 'destructive'} className="text-[10px] px-2 py-0.5">
                  {it.change > 0 ? `+${it.change}` : String(it.change)}
                </Badge>
              )}
            </div>
          </div>
        ))}
        {items.length > 20 && (
          <div className="text-[11px] text-slate-500 pt-1">顯示前 20 筆（共 {items.length}）</div>
        )}
      </div>
    </div>
  )
}
