import { useEffect, useMemo, useState } from 'react'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { StoryChoice } from '../types/story.types'
import type { RagMode } from '../types/story.types'

interface TurnSubmitOptions {
  choiceId?: string
  ragMode?: RagMode
  rerankMode?: RagMode
  useAgent?: boolean
  includeImage?: boolean
  asyncTurn?: boolean
}

interface PlayerInputProps {
  sessionId: string
  onSubmit: (input: string, options?: TurnSubmitOptions) => Promise<void>
  isExecuting: boolean
  choices?: StoryChoice[]
  worldId?: string
  ragAuto?: boolean | null
  ragMode?: RagMode | null
  ragAvailable?: boolean | null
  enrichWithRag?: boolean | null
  ragNextTurn?: boolean | null
  rerankMode?: RagMode | null
  rerankNextTurn?: boolean | null
  prefillText?: string | null
  onPrefillApplied?: () => void
}

export function PlayerInput({
  sessionId,
  onSubmit,
  isExecuting,
  choices,
  worldId,
  ragAuto,
  ragMode: serverRagModeProp,
  ragAvailable,
  enrichWithRag,
  ragNextTurn,
  rerankMode: serverRerankModeProp,
  rerankNextTurn,
  prefillText,
  onPrefillApplied,
}: PlayerInputProps) {
  const [input, setInput] = useState('')
  const [ragModeDirty, setRagModeDirty] = useState(false)
  const [rerankModeDirty, setRerankModeDirty] = useState(false)

  const storageKey = useMemo(() => `story:${sessionId}:controls`, [sessionId])

  const serverRagMode = useMemo<RagMode>(() => {
    if (serverRagModeProp === 'auto' || serverRagModeProp === 'on' || serverRagModeProp === 'off') {
      return serverRagModeProp
    }
    const isAuto = ragAuto !== false
    if (isAuto) return 'auto'
    return enrichWithRag ? 'on' : 'off'
  }, [serverRagModeProp, ragAuto, enrichWithRag])

  const serverRerankMode = useMemo<RagMode>(() => {
    if (serverRerankModeProp === 'auto' || serverRerankModeProp === 'on' || serverRerankModeProp === 'off') {
      return serverRerankModeProp
    }
    return 'auto'
  }, [serverRerankModeProp])

  const [ragMode, setRagMode] = useState<RagMode>(serverRagMode)
  const [rerankMode, setRerankMode] = useState<RagMode>(serverRerankMode)
  const [useAgent, setUseAgent] = useState(false)
  const [includeImage, setIncludeImage] = useState(true)
  const [asyncTurn, setAsyncTurn] = useState(true)

  useEffect(() => {
    // Load controls per-session (best-effort)
    try {
      const raw = localStorage.getItem(storageKey)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (parsed?.ragMode === 'auto' || parsed?.ragMode === 'on' || parsed?.ragMode === 'off') {
        setRagMode(parsed.ragMode)
        setRagModeDirty(true)
      }
      if (parsed?.rerankMode === 'auto' || parsed?.rerankMode === 'on' || parsed?.rerankMode === 'off') {
        setRerankMode(parsed.rerankMode)
        setRerankModeDirty(true)
      }
      if (typeof parsed?.useAgent === 'boolean') setUseAgent(parsed.useAgent)
      if (typeof parsed?.includeImage === 'boolean') setIncludeImage(parsed.includeImage)
      if (typeof parsed?.asyncTurn === 'boolean') setAsyncTurn(parsed.asyncTurn)
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  useEffect(() => {
    // Keep UI in sync with server unless user has made a local override.
    if (!ragModeDirty) {
      setRagMode(serverRagMode)
      return
    }
    if (ragMode === serverRagMode) {
      setRagModeDirty(false)
    }
  }, [serverRagMode, ragMode, ragModeDirty])

  useEffect(() => {
    if (!rerankModeDirty) {
      setRerankMode(serverRerankMode)
      return
    }
    if (rerankMode === serverRerankMode) {
      setRerankModeDirty(false)
    }
  }, [serverRerankMode, rerankMode, rerankModeDirty])

  useEffect(() => {
    try {
      localStorage.setItem(
        storageKey,
        JSON.stringify({ ragMode, rerankMode, useAgent, includeImage, asyncTurn })
      )
    } catch {
      // ignore
    }
  }, [storageKey, ragMode, rerankMode, useAgent, includeImage, asyncTurn])

  useEffect(() => {
    if (!prefillText || !String(prefillText).trim()) return
    const text = String(prefillText).trim()
    setInput((prev) => {
      const current = String(prev || '').trim()
      if (!current) return text
      if (current.includes(text)) return prev
      return `${prev}\n${text}`
    })
    onPrefillApplied?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillText])

  const effectiveNextTurnRag = useMemo(() => {
    if (ragMode === 'off') return false
    if (ragMode === 'on') return true
    return Boolean(ragAvailable)
  }, [ragMode, ragAvailable])

  const effectiveNextTurnRerank = useMemo(() => {
    if (ragMode === 'off') return false
    if (rerankMode === 'off') return false
    if (rerankMode === 'on') return true
    return Boolean(rerankNextTurn)
  }, [ragMode, rerankMode, rerankNextTurn])

  const handleSubmit = async () => {
    if (!input.trim() || isExecuting) return

    await onSubmit(input, { ragMode, rerankMode, useAgent, includeImage, asyncTurn })
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleChoiceClick = async (choice: StoryChoice) => {
    if (isExecuting) return
    if (choice.can_choose === false) return

    await onSubmit(choice.text, { choiceId: choice.choice_id, ragMode, rerankMode, useAgent, includeImage, asyncTurn })
    setInput('')
  }

  return (
    <Card className="p-4 bg-slate-800/80">
      <div className="flex flex-col gap-3 mb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {worldId && (
              <Badge variant="outline" className="text-xs">
                World: {worldId}
              </Badge>
            )}
            <Badge
              variant={ragMode === 'auto' ? 'secondary' : ragMode === 'on' ? 'default' : 'outline'}
              className="text-xs"
              title={ragModeDirty ? '此設定將在下一回合送出後生效' : undefined}
            >
              RAG: {ragMode}
              {ragModeDirty ? '（待套用）' : ''}
            </Badge>
            <Badge
              variant={rerankMode === 'auto' ? 'secondary' : rerankMode === 'on' ? 'default' : 'outline'}
              className="text-xs"
              title={rerankModeDirty ? '此設定將在下一回合送出後生效' : undefined}
            >
              Rerank: {rerankMode}
              {rerankModeDirty ? '（待套用）' : ''}
            </Badge>
            {ragModeDirty ? (
              <Badge variant={effectiveNextTurnRag ? 'default' : 'outline'} className="text-xs">
                下一回合（待套用）：{effectiveNextTurnRag ? '會用知識庫' : '不會用'}
              </Badge>
            ) : typeof ragNextTurn === 'boolean' ? (
              <Badge variant={ragNextTurn ? 'default' : 'outline'} className="text-xs">
                下一回合：{ragNextTurn ? '會用知識庫' : '不會用'}
              </Badge>
            ) : (
              <Badge variant={effectiveNextTurnRag ? 'default' : 'outline'} className="text-xs">
                下一回合：{effectiveNextTurnRag ? '會用知識庫' : '不會用'}
              </Badge>
            )}
            <Badge variant={effectiveNextTurnRerank ? 'default' : 'outline'} className="text-xs">
              下一回合 rerank：{effectiveNextTurnRerank ? '開' : '關'}
            </Badge>
            {ragMode !== 'off' && typeof ragAvailable === 'boolean' && !ragAvailable && (
              <Badge variant="destructive" className="text-xs">
                此世界尚無 RAG 文件
              </Badge>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">RAG 模式</span>
              <Select
                value={ragMode}
                onValueChange={(value) => {
                  const next = value as RagMode
                  setRagMode(next)
                  setRagModeDirty(true)
                }}
                disabled={isExecuting}
              >
                <SelectTrigger className="h-8 w-[140px] text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto（依 world_id）</SelectItem>
                  <SelectItem value="on">on（強制）</SelectItem>
                  <SelectItem value="off">off（關閉）</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Rerank</span>
              <Select
                value={rerankMode}
                onValueChange={(value) => {
                  const next = value as RagMode
                  setRerankMode(next)
                  setRerankModeDirty(true)
                }}
                disabled={isExecuting}
              >
                <SelectTrigger className="h-8 w-[140px] text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto（依 world）</SelectItem>
                  <SelectItem value="on">on（強制）</SelectItem>
                  <SelectItem value="off">off（關閉）</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Switch checked={asyncTurn} onCheckedChange={setAsyncTurn} label="非同步回合" disabled={isExecuting} />
            <Switch checked={useAgent} onCheckedChange={setUseAgent} label="AI 助手" disabled={isExecuting} />
            <Switch
              checked={includeImage}
              onCheckedChange={setIncludeImage}
              label="生成圖像"
              disabled={isExecuting}
            />
          </div>
        </div>
      </div>

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
            key={choice.choice_id}
            variant="outline"
            size="sm"
            onClick={() => void handleChoiceClick(choice)}
            disabled={isExecuting || choice.can_choose === false}
            title={choice.description || `${choice.type || 'action'}${choice.difficulty ? ` / ${choice.difficulty}` : ''}`}
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
