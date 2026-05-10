import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import { useToolExecutionMonitor } from '../hooks/useToolExecutionMonitor'
import type { AgentTaskResult, AgentTaskStep } from '../types/agent.types'

interface ToolExecutionPanelProps {
  taskId?: string
  onComplete?: (result: AgentTaskResult) => void
  onError?: (error: string) => void
}

/**
 * Real-time tool execution monitoring panel
 * Displays execution progress, steps, and tool calls
 */
export function ToolExecutionPanel({
  taskId,
  onComplete,
  onError,
}: ToolExecutionPanelProps) {
  const [notifications, setNotifications] = useState<string[]>([])

  const { taskResult, isRunning, isCompleted, isFailed } = useToolExecutionMonitor({
    taskId,
    enabled: !!taskId,
    pollingInterval: 2000,
    onStatusChange: (status) => {
      setNotifications((prev) => [...prev, `Status changed to: ${status}`])
    },
    onComplete,
    onError,
  })

  useEffect(() => {
    // Clear notifications when task ID changes
    if (taskId) {
      setNotifications([])
    }
  }, [taskId])

  if (!taskId || !taskResult) {
    return null
  }

  const progress = taskResult.total_iterations > 0
    ? (taskResult.steps.length / taskResult.total_iterations) * 100
    : 0

  const getStatusColor = (status: AgentTaskResult['status']) => {
    switch (status) {
      case 'running':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'completed':
        return 'bg-green-500/20 text-green-300 border-green-500/30'
      case 'failed':
        return 'bg-red-500/20 text-red-300 border-red-500/30'
      default:
        return 'bg-slate-500/20 text-slate-300 border-slate-500/30'
    }
  }

  const getStatusIcon = (status: AgentTaskResult['status']) => {
    switch (status) {
      case 'running':
        return '⏳'
      case 'completed':
        return '✅'
      case 'failed':
        return '❌'
      default:
        return '❓'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString()
    } catch {
      return timestamp
    }
  }

  const renderStep = (step: AgentTaskStep, index: number) => {
    const isLast = index === taskResult.steps.length - 1
    const isToolCall = step.action && step.action !== 'Final Answer'

    return (
      <div key={step.step_number} className="relative">
        {/* Timeline connector */}
        {!isLast && (
          <div className="absolute left-6 top-12 bottom-0 w-0.5 bg-slate-700" />
        )}

        <div className="flex gap-4">
          {/* Step number badge */}
          <div
            className={cn(
              'flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center font-semibold z-10',
              isRunning && isLast
                ? 'bg-primary text-white animate-pulse'
                : 'bg-slate-800 text-slate-300 border-2 border-slate-700'
            )}
          >
            {step.step_number}
          </div>

          {/* Step content */}
          <div className="flex-1 pb-8">
            <div className="bg-slate-800/50 rounded-lg p-4 space-y-3">
              {/* Timestamp */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  {formatTimestamp(step.timestamp)}
                </span>
                {isToolCall && (
                  <Badge variant="outline" className="text-xs">
                    🛠️ Tool Call
                  </Badge>
                )}
              </div>

              {/* Thought */}
              {step.thought && (
                <div>
                  <div className="text-xs font-semibold text-slate-400 mb-1">
                    💭 Thought
                  </div>
                  <p className="text-sm text-slate-200">{step.thought}</p>
                </div>
              )}

              {/* Action */}
              {step.action && (
                <div>
                  <div className="text-xs font-semibold text-slate-400 mb-1">
                    ⚡ Action
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="outline"
                      className="font-mono text-xs bg-primary/10"
                    >
                      {step.action}
                    </Badge>
                  </div>
                </div>
              )}

              {/* Action Input */}
              {step.action_input && (
                <div>
                  <div className="text-xs font-semibold text-slate-400 mb-1">
                    📥 Input
                  </div>
                  <pre className="text-xs text-slate-300 bg-slate-900 rounded p-2 overflow-x-auto">
                    {typeof step.action_input === 'string'
                      ? step.action_input
                      : JSON.stringify(step.action_input, null, 2)}
                  </pre>
                </div>
              )}

              {/* Observation */}
              {step.observation && (
                <div>
                  <div className="text-xs font-semibold text-slate-400 mb-1">
                    👁️ Observation
                  </div>
                  <p className="text-sm text-slate-200">{step.observation}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Card className="border-primary/30">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            執行監控
            {isRunning && (
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            )}
          </CardTitle>
          <Badge className={getStatusColor(taskResult.status)}>
            {getStatusIcon(taskResult.status)} {taskResult.status.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Progress bar */}
        {isRunning && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">執行進度</span>
              <span className="text-primary font-medium">
                {taskResult.steps.length} / {taskResult.total_iterations} 步驟
              </span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {/* Task info */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-slate-800/30 rounded-lg">
          <div>
            <div className="text-xs text-slate-400">Task ID</div>
            <div className="text-sm font-mono text-slate-200 truncate">
              {taskResult.task_id}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">工具使用</div>
            <div className="flex flex-wrap gap-1 mt-1">
              {taskResult.tools_used.length > 0 ? (
                taskResult.tools_used.map((tool) => (
                  <Badge key={tool} variant="outline" className="text-xs">
                    {tool}
                  </Badge>
                ))
              ) : (
                <span className="text-xs text-slate-500">無</span>
              )}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">開始時間</div>
            <div className="text-sm text-slate-200">
              {formatTimestamp(taskResult.created_at)}
            </div>
          </div>
          {taskResult.completed_at && (
            <div>
              <div className="text-xs text-slate-400">完成時間</div>
              <div className="text-sm text-slate-200">
                {formatTimestamp(taskResult.completed_at)}
              </div>
            </div>
          )}
        </div>

        {/* Execution steps timeline */}
        {taskResult.steps.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold mb-4 flex items-center gap-2">
              <span>執行步驟</span>
              <Badge variant="outline" className="text-xs">
                {taskResult.steps.length}
              </Badge>
            </h4>
            <div className="space-y-0">
              {taskResult.steps.map((step, index) => renderStep(step, index))}
            </div>
          </div>
        )}

        {/* Final result */}
        {taskResult.result && isCompleted && (
          <div className="p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
            <h4 className="font-semibold text-green-400 mb-2 flex items-center gap-2">
              ✅ 任務完成
            </h4>
            <p className="text-sm text-slate-200 whitespace-pre-wrap">
              {taskResult.result}
            </p>
          </div>
        )}

        {/* Error message */}
        {taskResult.error && isFailed && (
          <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
            <h4 className="font-semibold text-red-400 mb-2 flex items-center gap-2">
              ❌ 執行失敗
            </h4>
            <p className="text-sm text-red-300">{taskResult.error}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
