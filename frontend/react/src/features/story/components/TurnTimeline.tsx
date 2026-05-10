import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { AgentActions } from '../types/story.types'
import { BookOpen, Bot, Flag, Handshake, Heart, Image as ImageIcon, Package, Search, User } from 'lucide-react'
import type { TimelineTurn } from '../types/timeline.types'
import { TurnInspectorDialog } from './TurnInspectorDialog'

interface TurnTimelineProps {
  sessionId: string
  turns: TimelineTurn[]
  isExecuting?: boolean
}

export function TurnTimeline({ sessionId, turns, isExecuting }: TurnTimelineProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [selectedTurn, setSelectedTurn] = useState<TimelineTurn | null>(null)

  const orderedTurns = useMemo(() => {
    return [...(turns || [])].sort((a, b) => (a.turn ?? 0) - (b.turn ?? 0))
  }, [turns])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [orderedTurns.length, isExecuting])

  if (!orderedTurns.length) {
    return (
      <div className="h-full overflow-y-auto space-y-4 p-6 bg-slate-800/50 rounded-lg">
        <Card className="p-6 bg-slate-900/40 border-slate-700">
          <div className="text-sm text-slate-400">尚無回合紀錄</div>
        </Card>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto space-y-6 p-6 bg-slate-800/50 rounded-lg">
      {orderedTurns.map((t) => (
        <div key={`turn-${t.turn}`} className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="text-xs">
                Turn {t.turn}
              </Badge>
              {t.scene && (
                <Badge variant="secondary" className="text-xs">
                  {t.scene}
                </Badge>
              )}
              {t.timestamp && (
                <Badge variant="outline" className="text-xs">
                  {new Date(t.timestamp).toLocaleString()}
                </Badge>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => {
                setSelectedTurn(t)
                setInspectorOpen(true)
              }}
            >
              檢視
            </Button>
          </div>

          <Card className="p-4 bg-slate-900/50 border-slate-700">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-slate-700/50 flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-slate-200" />
              </div>
              <div className="min-w-0">
                <div className="text-xs text-slate-400 mb-1">你</div>
                <div className="text-sm text-slate-100 whitespace-pre-wrap break-words">
                  {t.action}
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-gradient-to-r from-slate-900/60 to-slate-800/40 border-slate-700">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-600/60 to-purple-600/60 flex items-center justify-center flex-shrink-0">
                <BookOpen className="w-4 h-4 text-white" />
              </div>
              <div className="min-w-0">
                <div className="text-xs text-slate-400 mb-1">敘事者</div>
                <div className="text-sm text-slate-200 whitespace-pre-wrap break-words">
                  {t.result}
                </div>
              </div>
            </div>
          </Card>

          {t.agent_actions && t.agent_actions.tool_results?.length > 0 && (
            <AgentDeltaSummary agentActions={t.agent_actions} />
          )}
        </div>
      ))}

      {isExecuting && (
        <Card className="p-4 bg-slate-900/40 border-slate-700">
          <div className="text-sm text-slate-400">生成下一回合中...</div>
        </Card>
      )}

      <div ref={bottomRef} />

      <TurnInspectorDialog
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
        turn={selectedTurn}
        sessionId={sessionId}
      />
    </div>
  )
}

function AgentDeltaSummary({ agentActions }: { agentActions: AgentActions }) {
  const toolResults = agentActions.tool_results || []

  const agentNames = (() => {
    const names = new Set<string>()
    for (const c of agentActions.contributors || []) {
      const id = String(c.agent || '').trim()
      if (!id || id === 'orchestrator') continue
      names.add(id)
    }
    for (const tr of toolResults) {
      const id = String(tr.agent || '').trim()
      if (!id || id === 'orchestrator') continue
      names.add(id)
    }
    return Array.from(names)
  })()

  const flags: string[] = []
  const stats: Array<{ stat: string; change: number }> = []
  const items: Array<{ item: string; quantity: number }> = []
  const relationships: Array<{ character_id: string; change: number; new?: number }> = []
  let ragResultsCount: number | null = null
  let sceneImageGenerated = false

  for (const tr of toolResults) {
    if (!tr?.success) continue
    const data = tr.result || {}
    if (tr.tool === 'modify_world_state') {
      const modified = data.modified_flags || {}
      for (const key of Object.keys(modified)) flags.push(key)
    } else if (tr.tool === 'update_character_state') {
      const modified = data.modified_stats || {}
      for (const [stat, payload] of Object.entries(modified)) {
        const change = Number((payload as any)?.change ?? 0)
        if (Number.isFinite(change) && change !== 0) stats.push({ stat, change })
      }
    } else if (tr.tool === 'add_inventory_item') {
      const item = data.item
      const quantity = Number(data.quantity ?? 1)
      if (item) items.push({ item: String(item), quantity: Number.isFinite(quantity) ? quantity : 1 })
    } else if (tr.tool === 'update_relationship_state') {
      const modified = data.modified_relationships || {}
      for (const [character_id, payload] of Object.entries(modified)) {
        const change = Number((payload as any)?.change ?? 0)
        const newValue = (payload as any)?.new
        if (Number.isFinite(change) && change !== 0) {
          relationships.push({
            character_id: String(character_id),
            change,
            ...(newValue !== undefined ? { new: Number(newValue) } : {}),
          })
        }
      }
    } else if (tr.tool === 'rag_search') {
      if (typeof data.results_count === 'number') ragResultsCount = data.results_count
    } else if (tr.tool === 'generate_scene_image') {
      sceneImageGenerated = true
    }
  }

  const statLabel = (stat: string) => {
    const map: Record<string, string> = {
      health: 'HP',
      energy: 'Energy',
      experience: 'EXP',
      level: 'LV',
      intelligence: 'INT',
      charisma: 'CHA',
      luck: 'LUK',
    }
    return map[stat] || stat.toUpperCase()
  }

  const chips: Array<{ key: string; icon: ReactNode; text: string; variant?: 'default' | 'outline' | 'secondary' | 'destructive' }> = []

  if (agentNames.length > 0) {
    const preview = agentNames.slice(0, 3).join(', ')
    chips.push({
      key: 'agents',
      icon: <Bot className="w-3 h-3" />,
      text: `Agents：${preview}${agentNames.length > 3 ? '…' : ''}`,
      variant: 'secondary',
    })
  }

  if (flags.length > 0) {
    const preview = flags.slice(0, 2).join(', ')
    chips.push({
      key: 'flags',
      icon: <Flag className="w-3 h-3" />,
      text: `Flags +${flags.length}${preview ? `：${preview}${flags.length > 2 ? '…' : ''}` : ''}`,
      variant: 'outline',
    })
  }

  if (stats.length > 0) {
    const preview = stats
      .slice(0, 3)
      .map((s) => `${statLabel(s.stat)} ${s.change > 0 ? '+' : ''}${s.change}`)
      .join(', ')
    chips.push({
      key: 'stats',
      icon: <Heart className="w-3 h-3" />,
      text: preview,
      variant: 'outline',
    })
  }

  if (items.length > 0) {
    const preview = items
      .slice(0, 2)
      .map((i) => `+${i.item} x${i.quantity}`)
      .join(', ')
    chips.push({
      key: 'items',
      icon: <Package className="w-3 h-3" />,
      text: preview + (items.length > 2 ? '…' : ''),
      variant: 'outline',
    })
  }

  if (relationships.length > 0) {
    const preview = relationships
      .slice(0, 2)
      .map((r) => `${r.character_id} ${r.change > 0 ? '+' : ''}${r.change}${typeof r.new === 'number' ? `→${r.new}` : ''}`)
      .join(', ')
    chips.push({
      key: 'rels',
      icon: <Handshake className="w-3 h-3" />,
      text: `關係：${preview}${relationships.length > 2 ? '…' : ''}`,
      variant: 'outline',
    })
  }

  if (ragResultsCount !== null) {
    chips.push({
      key: 'rag',
      icon: <Search className="w-3 h-3" />,
      text: `RAG：${ragResultsCount} 條`,
      variant: 'outline',
    })
  }

  if (sceneImageGenerated) {
    chips.push({
      key: 'image',
      icon: <ImageIcon className="w-3 h-3" />,
      text: '圖像生成',
      variant: 'outline',
    })
  }

  if (chips.length === 0) return null

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold text-slate-300">變更摘要</div>
        <Badge
          variant={agentActions.overall_success ? 'secondary' : 'destructive'}
          className="text-xs"
        >
          {agentActions.overall_success ? 'OK' : '有錯誤'}
        </Badge>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {chips.map((c) => (
          <Badge key={c.key} variant={c.variant || 'outline'} className="text-xs flex items-center gap-1">
            {c.icon}
            <span className="truncate">{c.text}</span>
          </Badge>
        ))}
      </div>
    </div>
  )
}
