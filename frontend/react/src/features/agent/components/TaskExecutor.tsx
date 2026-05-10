import { useState } from 'react'
import { useAgentTask } from '../hooks/useAgentTask'
import { useAgentTools } from '../hooks/useAgentTools'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ToolSelector } from './ToolSelector'
import { ToolParameterForm } from './ToolParameterForm'
import { ToolExecutionPanel } from './ToolExecutionPanel'
import { useUiStore } from '@/stores/uiStore'

export function TaskExecutor() {
  const [taskDescription, setTaskDescription] = useState('')
  const [maxIterations, setMaxIterations] = useState(5)
  const [selectedTools, setSelectedTools] = useState<string[]>([])
  const [toolParameters, setToolParameters] = useState<Record<string, Record<string, any>>>({})
  const [showToolSelector, setShowToolSelector] = useState(false)
  const [showParameterForm, setShowParameterForm] = useState(false)
  const [currentTaskId, setCurrentTaskId] = useState<string | undefined>()

  const agentTask = useAgentTask()
  const { data: toolsData } = useAgentTools()
  const { addNotification } = useUiStore()

  const handleExecute = async () => {
    if (!taskDescription.trim()) return

    try {
      const result = await agentTask.mutateAsync({
        task_description: taskDescription,
        max_iterations: maxIterations,
        tools: selectedTools.length > 0 ? selectedTools : undefined,
        context: Object.keys(toolParameters).length > 0 ? { tool_parameters: toolParameters } : undefined,
      })

      // Set current task ID for monitoring
      if (result.task_id) {
        setCurrentTaskId(result.task_id)
      }

      addNotification({
        type: 'success',
        title: '任務已啟動',
        message: `Task ID: ${result.task_id}`,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '任務啟動失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  const handleTaskComplete = () => {
    addNotification({
      type: 'success',
      title: '任務已完成',
      message: '所有步驟執行完成',
    })
  }

  const handleTaskError = (error: string) => {
    addNotification({
      type: 'error',
      title: '任務執行失敗',
      message: error,
    })
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>執行 Agent 任務</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label>任務描述</Label>
            <Textarea
              value={taskDescription}
              onChange={(e) => setTaskDescription(e.target.value)}
              placeholder="描述你想讓 Agent 完成的任務..."
              rows={4}
              className="mt-1"
            />
          </div>

          <div>
            <Label>最大迭代次數</Label>
            <Input
              type="number"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value) || 5)}
              min={1}
              max={20}
              className="mt-1"
            />
          </div>

          {/* Tool selection toggle */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-700">
            <div>
              <Label>工具選擇</Label>
              <p className="text-xs text-slate-400 mt-1">
                {selectedTools.length === 0
                  ? '未選擇（Agent 將自動選擇工具）'
                  : `已選擇 ${selectedTools.length} 個工具`}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowToolSelector(!showToolSelector)}
            >
              {showToolSelector ? '隱藏工具選擇' : '選擇工具'}
            </Button>
          </div>

          {/* Parameter configuration toggle */}
          {selectedTools.length > 0 && (
            <div className="flex items-center justify-between pt-2 border-t border-slate-700">
              <div>
                <Label>參數配置</Label>
                <p className="text-xs text-slate-400 mt-1">
                  {Object.keys(toolParameters).length === 0
                    ? '未配置（使用默認參數）'
                    : `已配置 ${Object.keys(toolParameters).length} 個工具的參數`}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowParameterForm(!showParameterForm)}
              >
                {showParameterForm ? '隱藏參數配置' : '配置參數'}
              </Button>
            </div>
          )}

          <Button
            onClick={handleExecute}
            disabled={!taskDescription.trim() || agentTask.isPending}
            className="w-full"
          >
            {agentTask.isPending ? '執行中...' : '執行任務'}
          </Button>
        </CardContent>
      </Card>

      {/* Tool selector */}
      {showToolSelector && (
        <ToolSelector
          selectedTools={selectedTools}
          onToolsChange={setSelectedTools}
        />
      )}

      {/* Parameter configuration form */}
      {showParameterForm && selectedTools.length > 0 && toolsData && (
        <ToolParameterForm
          tools={toolsData.tools}
          selectedToolNames={selectedTools}
          parameters={toolParameters}
          onParametersChange={setToolParameters}
        />
      )}

      {/* Tool execution monitoring panel */}
      {currentTaskId && (
        <ToolExecutionPanel
          taskId={currentTaskId}
          onComplete={handleTaskComplete}
          onError={handleTaskError}
        />
      )}

      {/* Legacy execution results (fallback) */}
      {!currentTaskId && agentTask.data && (
        <Card>
          <CardHeader>
            <CardTitle>執行結果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 最終結果 */}
            {agentTask.data.result && (
              <div className="p-4 bg-green-900/20 border border-green-500/30 rounded">
                <h4 className="font-semibold text-green-400 mb-2">任務完成</h4>
                <p className="text-sm text-slate-200 whitespace-pre-wrap">
                  {agentTask.data.result}
                </p>
              </div>
            )}

            {/* 使用的工具 */}
            <div>
              <h4 className="font-semibold mb-2">使用的工具</h4>
              <div className="flex flex-wrap gap-2">
                {agentTask.data.tools_used.map((tool) => (
                  <span
                    key={tool}
                    className="px-2 py-1 bg-blue-900/30 text-blue-300 rounded text-sm"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>

            {/* 執行步驟 */}
            <div>
              <h4 className="font-semibold mb-2">
                執行步驟 ({agentTask.data.steps.length})
              </h4>
              <div className="space-y-2">
                {agentTask.data.steps.map((step) => (
                  <div
                    key={step.step_number}
                    className="p-3 bg-slate-800/50 rounded"
                  >
                    <div className="flex items-start gap-2">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/20 text-primary flex items-center justify-center text-xs font-semibold">
                        {step.step_number}
                      </span>
                      <div className="flex-1 space-y-1 text-sm">
                        <div>
                          <span className="text-slate-400">思考: </span>
                          <span className="text-slate-200">{step.thought}</span>
                        </div>
                        <div>
                          <span className="text-slate-400">行動: </span>
                          <span className="text-blue-300">{step.action}</span>
                        </div>
                        <div>
                          <span className="text-slate-400">觀察: </span>
                          <span className="text-slate-200">{step.observation}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 錯誤信息 */}
            {agentTask.data.error && (
              <div className="p-4 bg-red-900/20 border border-red-500/30 rounded">
                <h4 className="font-semibold text-red-400 mb-2">錯誤</h4>
                <p className="text-sm text-red-300">{agentTask.data.error}</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 執行錯誤 */}
      {agentTask.error && (
        <Card className="border-red-500/50">
          <CardContent className="py-4">
            <p className="text-red-400 text-sm">
              執行失敗：{agentTask.error instanceof Error ? agentTask.error.message : '未知錯誤'}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
