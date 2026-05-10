import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { WorldPack } from '@/features/worlds/types/world.types'

interface QuestExplorationPanelProps {
  flags: Record<string, any>
  worldpack?: WorldPack | null
}

type QuestStage = 'complete' | 'failed' | 'in_progress' | 'started' | 'flag'

function stageLabel(stage: QuestStage) {
  switch (stage) {
    case 'complete':
      return '完成'
    case 'failed':
      return '失敗'
    case 'in_progress':
      return '進行中'
    case 'started':
      return '已接受'
    default:
      return '狀態'
  }
}

function stageVariant(stage: QuestStage): 'default' | 'outline' | 'secondary' | 'destructive' {
  switch (stage) {
    case 'complete':
      return 'default'
    case 'failed':
      return 'destructive'
    case 'in_progress':
      return 'secondary'
    case 'started':
      return 'outline'
    default:
      return 'outline'
  }
}

function detectStage(flagKey: string): { topic: string; stage: QuestStage } {
  const rest = flagKey.replace(/^quest_/, '')
  const suffixes: Array<{ suffix: string; stage: QuestStage }> = [
    { suffix: '_complete', stage: 'complete' },
    { suffix: '_completed', stage: 'complete' },
    { suffix: '_failed', stage: 'failed' },
    { suffix: '_in_progress', stage: 'in_progress' },
    { suffix: '_started', stage: 'started' },
  ]
  for (const s of suffixes) {
    if (rest.endsWith(s.suffix)) {
      return { topic: rest.slice(0, -s.suffix.length) || rest, stage: s.stage }
    }
  }
  return { topic: rest, stage: 'flag' }
}

function stagePriority(stage: QuestStage) {
  const order: Record<QuestStage, number> = {
    complete: 5,
    failed: 4,
    in_progress: 3,
    started: 2,
    flag: 1,
  }
  return order[stage] || 0
}

export function QuestExplorationPanel({ flags, worldpack }: QuestExplorationPanelProps) {
  const quests = useMemo(() => {
    const map = new Map<string, { topic: string; stage: QuestStage; flags: string[] }>()
    for (const [k, v] of Object.entries(flags || {})) {
      const key = String(k || '').trim()
      if (!key.startsWith('quest_')) continue
      if (!v) continue
      const { topic, stage } = detectStage(key)
      const current = map.get(topic)
      if (!current) {
        map.set(topic, { topic, stage, flags: [key] })
      } else {
        current.flags.push(key)
        if (stagePriority(stage) > stagePriority(current.stage)) current.stage = stage
      }
    }
    return Array.from(map.values()).sort((a, b) => stagePriority(b.stage) - stagePriority(a.stage) || a.topic.localeCompare(b.topic))
  }, [flags])

  const locations = useMemo(() => {
    const out: string[] = []
    for (const [k, v] of Object.entries(flags || {})) {
      const key = String(k || '').trim()
      if (!key.startsWith('location_discovered_')) continue
      if (!v) continue
      out.push(key.replace(/^location_discovered_/, '').replaceAll('_', ' '))
    }
    return out.sort((a, b) => a.localeCompare(b))
  }, [flags])

  const worldTitle = worldpack ? `${worldpack.name} (${worldpack.world_id})` : null

  return (
    <Card className="bg-slate-800/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-lg">任務 / 探索</CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              任務 {quests.length}
            </Badge>
            <Badge variant="outline" className="text-xs">
              地點 {locations.length}
            </Badge>
          </div>
        </div>
        {worldTitle && <p className="text-xs text-slate-500 mt-1">{worldTitle}</p>}
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="quests">
          <TabsList className="w-full">
            <TabsTrigger value="quests">任務日誌</TabsTrigger>
            <TabsTrigger value="explore">探索</TabsTrigger>
          </TabsList>

          <TabsContent value="quests" className="mt-4">
            {quests.length === 0 ? (
              <div className="text-sm text-slate-500">尚無任務紀錄（quest_* flags）</div>
            ) : (
              <div className="space-y-3">
                {quests.slice(0, 30).map((q) => (
                  <div key={q.topic} className="p-3 rounded-lg border border-slate-700 bg-slate-900/40">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-sm font-semibold text-slate-100 truncate">
                          {q.topic.replaceAll('_', ' ') || '任務'}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {q.flags.slice(0, 3).join(', ')}
                          {q.flags.length > 3 ? '…' : ''}
                        </div>
                      </div>
                      <Badge variant={stageVariant(q.stage)} className="text-xs flex-shrink-0">
                        {stageLabel(q.stage)}
                      </Badge>
                    </div>
                  </div>
                ))}
                {quests.length > 30 && (
                  <div className="text-xs text-slate-500">顯示前 30 筆（共 {quests.length}）</div>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="explore" className="mt-4">
            {locations.length === 0 ? (
              <div className="text-sm text-slate-500">尚未探索任何地點（location_discovered_* flags）</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {locations.slice(0, 80).map((loc) => (
                  <Badge key={loc} variant="secondary" className="text-xs">
                    {loc}
                  </Badge>
                ))}
                {locations.length > 80 && (
                  <div className="text-xs text-slate-500">顯示前 80 筆（共 {locations.length}）</div>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

