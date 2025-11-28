/**
 * Agent Actions Panel
 * Displays Agent autonomous actions taken during story turns
 */

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { AgentActions, AgentToolResult } from '../types/story.types'
import {
  Bot,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Undo2,
  Flag,
  Heart,
  Package,
  Search,
  Image as ImageIcon
} from 'lucide-react'

interface AgentActionsPanelProps {
  agentActions: AgentActions
  className?: string
}

export function AgentActionsPanel({ agentActions, className = '' }: AgentActionsPanelProps) {
  const [expanded, setExpanded] = useState(true)

  if (!agentActions || agentActions.tool_results.length === 0) {
    return null
  }

  const successCount = agentActions.tool_results.filter(r => r.success).length
  const totalCount = agentActions.tool_results.length

  return (
    <Card className={`p-4 border-2 ${agentActions.overall_success ? 'border-green-500/50' : 'border-red-500/50'} ${className}`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between hover:bg-accent/50 rounded p-2 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Bot className={`w-5 h-5 ${agentActions.overall_success ? 'text-green-500' : 'text-red-500'}`} />
          <span className="font-semibold">Agent 自主行動</span>
          <Badge variant={agentActions.overall_success ? 'default' : 'destructive'}>
            {successCount}/{totalCount}
          </Badge>
        </div>
        {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="mt-3 space-y-3">
          {/* Reasoning */}
          <div className="text-sm bg-muted/50 p-3 rounded">
            <div className="font-medium mb-1 text-xs text-muted-foreground">決策推理</div>
            <div className="text-foreground">{agentActions.reasoning}</div>
          </div>

          {/* Tool Results */}
          <div className="space-y-2">
            {agentActions.tool_results.map((result, index) => (
              <AgentToolResultCard key={index} result={result} index={index} />
            ))}
          </div>

          {/* Errors */}
          {agentActions.errors.length > 0 && (
            <div className="bg-destructive/10 border border-destructive/30 rounded p-3">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-destructive" />
                <span className="font-medium text-sm text-destructive">錯誤</span>
              </div>
              <ul className="text-sm space-y-1">
                {agentActions.errors.map((error, index) => (
                  <li key={index} className="text-destructive/90">• {error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

interface AgentToolResultCardProps {
  result: AgentToolResult
  index: number
}

function AgentToolResultCard({ result, index }: AgentToolResultCardProps) {
  const [showDetails, setShowDetails] = useState(false)

  const toolIcon = getToolIcon(result.tool)
  const toolName = getToolDisplayName(result.tool)
  const toolDescription = getToolDescription(result)

  return (
    <div className={`border rounded p-3 ${result.success ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
      {/* Tool Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 flex-1">
          <div className={`mt-0.5 ${result.success ? 'text-green-500' : 'text-red-500'}`}>
            {toolIcon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-sm">{toolName}</span>
              {result.success ? (
                <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
              )}
              {result.rollback_performed && (
                <Badge variant="outline" className="text-xs gap-1">
                  <Undo2 className="w-3 h-3" />
                  已回滾
                </Badge>
              )}
            </div>
            <div className="text-sm text-muted-foreground">
              {toolDescription}
            </div>
          </div>
        </div>

        {/* Show Details Button */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showDetails ? '隱藏' : '詳情'}
        </button>
      </div>

      {/* Error Message */}
      {!result.success && result.error && (
        <div className="mt-2 text-sm text-red-600 bg-red-50 dark:bg-red-950/20 p-2 rounded">
          錯誤: {result.error}
        </div>
      )}

      {/* Detailed Result */}
      {showDetails && result.result && (
        <div className="mt-3 text-xs bg-muted/50 p-2 rounded">
          <pre className="whitespace-pre-wrap break-words text-muted-foreground">
            {JSON.stringify(result.result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// Helper Functions ------------------------------------------------------------

function getToolIcon(toolName: string) {
  switch (toolName) {
    case 'modify_world_state':
      return <Flag className="w-4 h-4" />
    case 'update_character_state':
      return <Heart className="w-4 h-4" />
    case 'add_inventory_item':
      return <Package className="w-4 h-4" />
    case 'rag_search':
      return <Search className="w-4 h-4" />
    case 'generate_scene_image':
      return <ImageIcon className="w-4 h-4" />
    default:
      return <Bot className="w-4 h-4" />
  }
}

function getToolDisplayName(toolName: string): string {
  const names: Record<string, string> = {
    'modify_world_state': '修改世界狀態',
    'update_character_state': '更新角色狀態',
    'add_inventory_item': '添加物品',
    'rag_search': '記憶搜索',
    'generate_scene_image': '生成場景圖像'
  }
  return names[toolName] || toolName
}

function getToolDescription(result: AgentToolResult): string {
  if (!result.success) {
    return '執行失敗'
  }

  if (!result.result) {
    return '執行成功'
  }

  const data = result.result

  switch (result.tool) {
    case 'modify_world_state':
      if (data.modified_flags) {
        const flags = Object.keys(data.modified_flags)
        return `修改了 ${flags.length} 個標記: ${flags.slice(0, 2).join(', ')}${flags.length > 2 ? '...' : ''}`
      }
      return '世界狀態已更新'

    case 'update_character_state':
      if (data.modified_stats) {
        const stats = Object.keys(data.modified_stats)
        const changes = stats.map(stat => {
          const change = data.modified_stats[stat]?.change || 0
          return `${stat.toUpperCase()} ${change > 0 ? '+' : ''}${change}`
        })
        return changes.join(', ')
      }
      return '角色狀態已更新'

    case 'add_inventory_item':
      if (data.item && data.quantity) {
        return `獲得 ${data.item} x${data.quantity}`
      }
      return '物品已添加'

    case 'rag_search':
      if (data.results_count !== undefined) {
        return `找到 ${data.results_count} 條相關記憶`
      }
      return '記憶搜索完成'

    case 'generate_scene_image':
      return '場景圖像已生成'

    default:
      return '執行成功'
  }
}
