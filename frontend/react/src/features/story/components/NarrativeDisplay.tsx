import { Card } from '@/components/ui/card'
import type { StoryChoice } from '../types/story.types'

interface NarrativeDisplayProps {
  narrative: string
  dialogue?: string
  choices?: StoryChoice[]
}

export function NarrativeDisplay({ narrative, dialogue, choices }: NarrativeDisplayProps) {
  return (
    <div className="h-full overflow-y-auto space-y-4 p-6 bg-slate-800/50 rounded-lg">
      {/* 主要敘事 */}
      <div className="prose prose-invert max-w-none">
        <p className="text-slate-200 leading-relaxed whitespace-pre-wrap">
          {narrative}
        </p>
      </div>

      {/* 對話 */}
      {dialogue && (
        <Card className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border-blue-500/30 p-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold flex-shrink-0">
              💬
            </div>
            <div className="flex-1">
              <p className="text-blue-200 italic">"{dialogue}"</p>
            </div>
          </div>
        </Card>
      )}

      {/* 選項提示 */}
      {choices && choices.length > 0 && (
        <Card className="bg-slate-700/30 border-slate-600 p-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">可選動作：</h4>
          <ul className="space-y-1">
            {choices.map((choice) => (
              <li key={choice.choice_id} className="text-sm text-slate-400">
                • {choice.text}
                {choice.description && (
                  <span className="text-slate-500 ml-2">({choice.description})</span>
                )}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
