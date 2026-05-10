import { useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { StoryContextSnapshot, StoryTurnHistoryEntry } from '../types/story.types'
import type { WorldPack } from '@/features/worlds/types/world.types'

interface RelationshipPanelProps {
  context?: StoryContextSnapshot | null
  isLoading?: boolean
  error?: unknown
  worldpack?: WorldPack | null
  inventory?: string[]
  turnHistory?: StoryTurnHistoryEntry[] | null
  onQuickAction?: (text: string) => void
}

function scoreLabel(score: number) {
  if (score >= 6) return '盟友'
  if (score >= 2) return '友好'
  if (score >= 0) return '中立'
  if (score <= -6) return '敵對'
  return '緊張'
}

export function RelationshipPanel({
  context,
  isLoading,
  error,
  worldpack,
  inventory,
  turnHistory,
  onQuickAction,
}: RelationshipPanelProps) {
  if (isLoading) {
    return (
      <Card className="bg-slate-800/80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">關係（在場角色）</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">載入中...</CardContent>
      </Card>
    )
  }

  if (!context && error) {
    return (
      <Card className="bg-slate-800/80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">關係（在場角色）</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">
          目前無法取得關係資料（可能未啟用 enhanced mode）
        </CardContent>
      </Card>
    )
  }

  const present = (context?.present_characters || []).filter((c) => {
    const cid = String(c.character_id || '').trim()
    if (!cid) return false
    if (cid === 'player' || cid === 'narrator') return false
    return true
  })

  const [expandedId, setExpandedId] = useState<string | null>(null)

  const inventoryOptions = useMemo(() => {
    const list = Array.isArray(inventory) ? inventory : []
    return Array.from(new Set(list.map((i) => String(i).trim()).filter(Boolean))).slice(0, 50)
  }, [inventory])

  const [selectedItemByChar, setSelectedItemByChar] = useState<Record<string, string>>({})

  const getSelectedItem = (characterId: string) => {
    const current = selectedItemByChar[characterId]
    if (current) return current
    return inventoryOptions[0] || ''
  }

  const setSelectedItem = (characterId: string, item: string) => {
    setSelectedItemByChar((prev) => ({ ...prev, [characterId]: item }))
  }

  return (
    <Card className="bg-slate-800/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-lg">關係（在場角色）</CardTitle>
          <Badge variant="outline" className="text-xs">
            {present.length}
          </Badge>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          以本場景在場角色為主（relationship_score：-10..10）
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {present.length === 0 ? (
          <div className="text-sm text-slate-500">此場景沒有其他角色</div>
        ) : (
          present.map((c) => {
            const score = Number(c.relationship_score ?? 0)
            const magnitude = Math.min(10, Math.abs(score))
            const widthPercent = (magnitude / 10) * 50
            const isPositive = score > 0
            const isNegative = score < 0
            const characterId = String(c.character_id || '').trim()
            const isExpanded = expandedId === characterId

            const template = worldpack?.characters?.find((t) => t.character_id === characterId) || null
            const selectedItem = getSelectedItem(characterId)
            const lastInteraction = (() => {
              const history = Array.isArray(turnHistory) ? turnHistory : []
              for (let i = history.length - 1; i >= 0; i -= 1) {
                const h = history[i]
                const hay = `${h.player_input}\n${h.ai_response}`
                if (hay.includes(c.name) || hay.includes(characterId)) {
                  return {
                    turn: Number(h.turn ?? 0) + 1,
                    snippet: String(h.ai_response || '').slice(0, 80),
                  }
                }
              }
              return null
            })()

            return (
              <div key={c.character_id} className="p-3 rounded-lg border border-slate-700 bg-slate-900/40">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <div className="text-sm font-semibold text-slate-100 truncate">{c.name}</div>
                      <Badge variant="secondary" className="text-xs">
                        {c.role}
                      </Badge>
                      {c.current_state && (
                        <Badge variant="outline" className="text-xs">
                          {c.current_state}
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 mt-1 truncate">{c.character_id}</div>
                  </div>

                  <div className="flex items-center gap-2 flex-shrink-0">
                    <Badge
                      variant={isPositive ? 'default' : isNegative ? 'destructive' : 'outline'}
                      className="text-xs"
                      title="relationship_score"
                    >
                      {scoreLabel(score)} {score >= 0 ? `+${score}` : String(score)}
                    </Badge>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => setExpandedId(isExpanded ? null : characterId)}
                    >
                      {isExpanded ? '收起' : '詳情'}
                    </Button>
                  </div>
                </div>

                <div className="mt-3">
                  <div className="relative h-2 w-full rounded-full bg-slate-700 overflow-hidden">
                    <div className="absolute left-1/2 top-0 h-full w-px bg-slate-400/40" />
                    {isPositive && widthPercent > 0 && (
                      <div
                        className="absolute top-0 h-full bg-green-500/80"
                        style={{ left: '50%', width: `${widthPercent}%` }}
                      />
                    )}
                    {isNegative && widthPercent > 0 && (
                      <div
                        className="absolute top-0 h-full bg-red-500/80"
                        style={{ right: '50%', width: `${widthPercent}%` }}
                      />
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="mt-4 space-y-3">
                    {lastInteraction && (
                      <div className="text-xs text-slate-500">
                        最近互動：Turn {lastInteraction.turn}
                        {lastInteraction.snippet ? ` • ${lastInteraction.snippet}${lastInteraction.snippet.length >= 80 ? '…' : ''}` : ''}
                      </div>
                    )}
                    {template ? (
                      <div className="space-y-2">
                        {template.personality_traits?.length > 0 && (
                          <div className="flex flex-wrap gap-2">
                            {template.personality_traits.slice(0, 8).map((t) => (
                              <Badge key={t} variant="secondary" className="text-xs">
                                {t}
                              </Badge>
                            ))}
                          </div>
                        )}

                        {(template.speaking_style || template.background_story) && (
                          <div className="text-xs text-slate-300 space-y-1">
                            {template.speaking_style && (
                              <div>
                                <span className="text-slate-500">說話風格：</span>
                                <span className="text-slate-200">{template.speaking_style}</span>
                              </div>
                            )}
                            {template.background_story && (
                              <div className="text-slate-200 whitespace-pre-wrap break-words">
                                <span className="text-slate-500">背景：</span>
                                {template.background_story}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-slate-500">
                        （此角色不在 worldpack.characters 中，僅顯示當前關係分數）
                      </div>
                    )}

                    <div className="rounded-lg border border-slate-700 bg-slate-950/30 p-3 space-y-3">
                      <div className="text-xs font-semibold text-slate-300">快捷互動（會自動填入輸入框）</div>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            onQuickAction?.(`我走向 ${c.name}，開啟對話並詢問：你知道些什麼線索？`)
                          }
                        >
                          對話
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            onQuickAction?.(`我用嚴厲口氣威脅 ${c.name}：說出真相，否則後果自負。`)
                          }
                        >
                          威脅
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            const item = selectedItem || '（請填入物品）'
                            onQuickAction?.(`我把 ${item} 送給 ${c.name}，並說：希望這能表達我的善意。`)
                          }}
                        >
                          贈送
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            const item = selectedItem || '（請填入物品）'
                            onQuickAction?.(`我想和 ${c.name} 交易：我提供 ${item}，希望換取你手上的情報或物資。`)
                          }}
                        >
                          交易
                        </Button>
                      </div>

                      <div className="flex flex-col gap-2">
                        <div className="text-xs text-slate-500">道具（用於贈送/交易模板）</div>
                        <Select
                          value={selectedItem}
                          onValueChange={(v) => setSelectedItem(characterId, v)}
                          disabled={inventoryOptions.length === 0}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue placeholder={inventoryOptions.length ? '選擇道具' : '背包為空'} />
                          </SelectTrigger>
                          <SelectContent>
                            {inventoryOptions.map((it) => (
                              <SelectItem key={it} value={it}>
                                {it}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          })
        )}
      </CardContent>
    </Card>
  )
}
