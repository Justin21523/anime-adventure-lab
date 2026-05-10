import { useMemo, useState, type ReactNode } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  ChevronDown,
  ChevronRight,
  Flag,
  MapPin,
  Package,
  Sparkles,
  Users,
} from 'lucide-react'

interface CharacterSheetProps {
  playerName: string
  stats: Record<string, any>
  inventory: string[]
  flags: Record<string, any>
}

export function CharacterSheet({ playerName, stats, inventory, flags }: CharacterSheetProps) {
  const level = Number(stats?.level ?? 1)
  const health = Number(stats?.health ?? 100)
  const energy = Number(stats?.energy ?? 100)

  const hpPercentage = Math.max(0, Math.min(100, health))
  const energyPercentage = Math.max(0, Math.min(100, energy))

  const statEntries = Object.entries(stats || {}).filter(
    ([key]) => !['health', 'energy', 'level'].includes(key)
  )

  const categorizedFlags = useMemo(() => {
    const allEntries = Object.entries(flags || {})
    const groups: Record<string, Array<[string, any]>> = {
      quests: [],
      npcs: [],
      locations: [],
      items: [],
      events: [],
      achievements: [],
      other: [],
    }

    const rules: Array<{ key: keyof typeof groups; prefix: string }> = [
      { key: 'quests', prefix: 'quest_' },
      { key: 'npcs', prefix: 'npc_met_' },
      { key: 'locations', prefix: 'location_discovered_' },
      { key: 'items', prefix: 'item_acquired_' },
      { key: 'events', prefix: 'event_' },
      { key: 'achievements', prefix: 'achievement_' },
    ]

    for (const [k, v] of allEntries) {
      const key = String(k || '').trim()
      if (!key) continue
      const rule = rules.find((r) => key.startsWith(r.prefix))
      if (rule) {
        groups[rule.key].push([key, v])
      } else {
        groups.other.push([key, v])
      }
    }

    const sortEntries = (a: [string, any], b: [string, any]) => {
      const aTruthy = Boolean(a[1])
      const bTruthy = Boolean(b[1])
      if (aTruthy !== bTruthy) return aTruthy ? -1 : 1
      return a[0].localeCompare(b[0])
    }

    for (const k of Object.keys(groups)) {
      groups[k].sort(sortEntries)
    }

    return groups
  }, [flags])

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    quests: true,
    npcs: false,
    locations: false,
    items: false,
    events: false,
    achievements: false,
    other: false,
  })

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const totalFlags = Object.keys(flags || {}).length

  return (
    <div className="space-y-4">
      {/* 角色狀態 */}
      <Card className="bg-slate-800/80">
        <CardHeader>
          <CardTitle className="text-lg">{playerName}</CardTitle>
          <p className="text-sm text-slate-400">等級 {level}</p>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* HP */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300">HP</span>
              <span className="text-slate-400">{health} / 100</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full transition-all"
                style={{ width: `${hpPercentage}%` }}
              />
            </div>
          </div>

          {/* Energy */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300">Energy</span>
              <span className="text-slate-400">{energy} / 100</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${energyPercentage}%` }}
              />
            </div>
          </div>

          {/* 屬性 */}
          {statEntries.length > 0 && (
            <div className="pt-2 border-t border-slate-700">
              <h4 className="text-sm font-semibold text-slate-300 mb-2">屬性</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {statEntries.map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-slate-400 capitalize">{key}</span>
                    <span className="text-slate-200">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 物品欄 */}
      <Card className="bg-slate-800/80">
        <CardHeader>
          <CardTitle className="text-lg">物品欄</CardTitle>
        </CardHeader>
        <CardContent>
          {inventory.length === 0 ? (
            <p className="text-sm text-slate-500">空空如也</p>
          ) : (
            <div className="space-y-2">
              {inventory.map((item, idx) => (
                <div key={`${item}-${idx}`} className="p-2 bg-slate-700/30 rounded">
                  <div className="text-sm font-medium text-slate-200">{item}</div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 標記/任務狀態 */}
      {flags && totalFlags > 0 && (
        <Card className="bg-slate-800/80">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-lg">世界狀態</CardTitle>
              <Badge variant="outline" className="text-xs">
                {totalFlags} flags
              </Badge>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              依前綴分類：任務 / NPC / 地點 / 物品 / 事件 / 成就
            </p>
          </CardHeader>
          <CardContent className="space-y-2">
            <FlagSection
              icon={<Flag className="w-4 h-4" />}
              title="任務"
              sectionKey="quests"
              entries={categorizedFlags.quests}
              open={openSections.quests}
              onToggle={toggleSection}
              formatKey={(k) => k}
            />
            <FlagSection
              icon={<Users className="w-4 h-4" />}
              title="已遇見 NPC / 角色"
              sectionKey="npcs"
              entries={categorizedFlags.npcs}
              open={openSections.npcs}
              onToggle={toggleSection}
              formatKey={(k) => k.replace(/^npc_met_/, '').replaceAll('_', ' ')}
            />
            <FlagSection
              icon={<MapPin className="w-4 h-4" />}
              title="已探索地點"
              sectionKey="locations"
              entries={categorizedFlags.locations}
              open={openSections.locations}
              onToggle={toggleSection}
              formatKey={(k) => k.replace(/^location_discovered_/, '').replaceAll('_', ' ')}
            />
            <FlagSection
              icon={<Package className="w-4 h-4" />}
              title="取得物品"
              sectionKey="items"
              entries={categorizedFlags.items}
              open={openSections.items}
              onToggle={toggleSection}
              formatKey={(k) => k.replace(/^item_acquired_/, '').replaceAll('_', ' ')}
            />
            <FlagSection
              icon={<Sparkles className="w-4 h-4" />}
              title="事件 / 成就"
              sectionKey="events"
              entries={[...categorizedFlags.events, ...categorizedFlags.achievements]}
              open={openSections.events}
              onToggle={toggleSection}
              formatKey={(k) => k.replaceAll('_', ' ')}
            />
            <FlagSection
              icon={<ChevronRight className="w-4 h-4" />}
              title="其他"
              sectionKey="other"
              entries={categorizedFlags.other}
              open={openSections.other}
              onToggle={toggleSection}
              formatKey={(k) => k}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function FlagSection({
  icon,
  title,
  sectionKey,
  entries,
  open,
  onToggle,
  formatKey,
}: {
  icon: ReactNode
  title: string
  sectionKey: string
  entries: Array<[string, any]>
  open: boolean
  onToggle: (key: string) => void
  formatKey: (key: string) => string
}) {
  const count = entries.length

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/30">
      <Button
        type="button"
        variant="ghost"
        className="w-full justify-between px-3 py-2 h-auto hover:bg-slate-900/60"
        onClick={() => onToggle(sectionKey)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-slate-300">{icon}</span>
          <span className="text-sm font-semibold text-slate-200 truncate">{title}</span>
          <Badge variant="outline" className="text-xs">
            {count}
          </Badge>
        </div>
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400" />
        )}
      </Button>

      {open && (
        <div className="px-3 pb-3 space-y-1">
          {count === 0 ? (
            <div className="text-xs text-slate-500">尚無紀錄</div>
          ) : (
            <div className="space-y-1">
              {entries.slice(0, 20).map(([key, value]) => {
                const label = formatKey(key)
                const isBool = typeof value === 'boolean'
                const isTruthy = Boolean(value)
                return (
                  <div key={key} className="flex items-start justify-between gap-2 text-xs">
                    <div className="text-slate-300 break-words min-w-0 flex-1">{label}</div>
                    <Badge
                      variant={isTruthy ? 'default' : 'outline'}
                      className="text-[10px] px-2 py-0.5 flex-shrink-0"
                      title={isBool ? undefined : String(value)}
                    >
                      {isBool ? (isTruthy ? '✓' : '—') : String(value)}
                    </Badge>
                  </div>
                )
              })}
              {count > 20 && (
                <div className="text-[11px] text-slate-500 pt-1">
                  顯示前 20 筆（共 {count}）
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
