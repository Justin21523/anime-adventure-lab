import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import type { CharacterState, InventoryItem } from '../types/story.types'

interface CharacterSheetProps {
  character: CharacterState
  inventory: InventoryItem[]
  flags: Record<string, any>
}

export function CharacterSheet({ character, inventory, flags }: CharacterSheetProps) {
  const hpPercentage = (character.hp / character.max_hp) * 100
  const mpPercentage = character.mp && character.max_mp
    ? (character.mp / character.max_mp) * 100
    : 0

  return (
    <div className="space-y-4">
      {/* 角色狀態 */}
      <Card className="bg-slate-800/80">
        <CardHeader>
          <CardTitle className="text-lg">{character.name}</CardTitle>
          <p className="text-sm text-slate-400">等級 {character.level}</p>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* HP */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-300">HP</span>
              <span className="text-slate-400">{character.hp} / {character.max_hp}</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full transition-all"
                style={{ width: `${hpPercentage}%` }}
              />
            </div>
          </div>

          {/* MP (if exists) */}
          {character.mp !== undefined && character.max_mp && (
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-300">MP</span>
                <span className="text-slate-400">{character.mp} / {character.max_mp}</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${mpPercentage}%` }}
                />
              </div>
            </div>
          )}

          {/* 屬性 */}
          {character.stats && Object.keys(character.stats).length > 0 && (
            <div className="pt-2 border-t border-slate-700">
              <h4 className="text-sm font-semibold text-slate-300 mb-2">屬性</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(character.stats).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-slate-400 capitalize">{key}</span>
                    <span className="text-slate-200">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 狀態效果 */}
          {character.status_effects && character.status_effects.length > 0 && (
            <div className="pt-2 border-t border-slate-700">
              <h4 className="text-sm font-semibold text-slate-300 mb-2">狀態</h4>
              <div className="flex flex-wrap gap-1">
                {character.status_effects.map((effect, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-2 py-1 bg-purple-900/30 text-purple-300 rounded"
                  >
                    {effect}
                  </span>
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
              {inventory.map((item) => (
                <div
                  key={item.id}
                  className="flex items-start justify-between p-2 bg-slate-700/30 rounded"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-slate-200">{item.name}</div>
                    {item.description && (
                      <div className="text-xs text-slate-400 mt-0.5">{item.description}</div>
                    )}
                  </div>
                  <div className="text-sm text-slate-400 ml-2">x{item.quantity}</div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 標記/任務狀態 */}
      {flags && Object.keys(flags).length > 0 && (
        <Card className="bg-slate-800/80">
          <CardHeader>
            <CardTitle className="text-lg">狀態標記</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 text-sm">
              {Object.entries(flags).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-slate-400">{key}</span>
                  <span className="text-slate-200">{String(value)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
