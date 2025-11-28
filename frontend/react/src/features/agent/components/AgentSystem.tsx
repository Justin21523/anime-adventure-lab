import { TaskExecutor } from './TaskExecutor'
import { ToolBrowser } from './ToolBrowser'
import { Button } from '@/components/ui/button'

export function AgentSystem() {
  return (
    <div className="container mx-auto p-8 space-y-6">
      {/* 頁面標題 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Agent 智能體系統</h1>
          <p className="text-slate-400 mt-1">執行複雜任務和工具調用</p>
        </div>
        <Button variant="outline" onClick={() => window.location.href = '/'}>
          返回首頁
        </Button>
      </div>

      {/* 兩欄布局 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <TaskExecutor />
        </div>
        <div>
          <ToolBrowser />
        </div>
      </div>
    </div>
  )
}
