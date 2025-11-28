import { useState } from 'react'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import type { StoryChoice } from '../types/story.types'

interface PlayerInputProps {
  onSubmit: (input: string) => Promise<void>
  isExecuting: boolean
  choices?: StoryChoice[]
}

export function PlayerInput({ onSubmit, isExecuting, choices }: PlayerInputProps) {
  const [input, setInput] = useState('')

  const handleSubmit = async () => {
    if (!input.trim() || isExecuting) return

    await onSubmit(input)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleChoiceClick = (choice: StoryChoice) => {
    setInput(choice.text)
  }

  return (
    <Card className="p-4 bg-slate-800/80">
      <Textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="描述你的行動... (Ctrl+Enter 提交)"
        className="mb-3 min-h-[100px] bg-slate-900/50"
        disabled={isExecuting}
      />

      <div className="flex gap-2 flex-wrap">
        <Button
          onClick={handleSubmit}
          disabled={!input.trim() || isExecuting}
          className="flex-shrink-0"
        >
          {isExecuting ? '執行中...' : '執行'}
        </Button>

        {choices?.map((choice) => (
          <Button
            key={choice.id}
            variant="outline"
            size="sm"
            onClick={() => handleChoiceClick(choice)}
            disabled={isExecuting}
            title={choice.description}
          >
            {choice.text}
          </Button>
        ))}
      </div>

      <p className="text-xs text-slate-500 mt-2">
        提示：使用 Ctrl+Enter 快速提交，或點擊選項按鈕快速填入
      </p>
    </Card>
  )
}
