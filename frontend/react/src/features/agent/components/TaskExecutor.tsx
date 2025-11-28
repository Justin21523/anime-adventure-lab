import { useState } from 'react'
import { useAgentTask } from '../hooks/useAgentTask'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export function TaskExecutor() {
  const [taskDescription, setTaskDescription] = useState('')
  const [maxIterations, setMaxIterations] = useState(5)
  const agentTask = useAgentTask()

  const handleExecute = async () => {
    if (!taskDescription.trim()) return

    await agentTask.mutateAsync({
      task_description: taskDescription,
      max_iterations: maxIterations,
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

          <Button
            onClick={handleExecute}
            disabled={!taskDescription.trim() || agentTask.isPending}
            className="w-full"
          >
            {agentTask.isPending ? '執行中...' : '執行任務'}
          </Button>
        </CardContent>
      </Card>

      {/* 執行結果 */}
      {agentTask.data && (
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
