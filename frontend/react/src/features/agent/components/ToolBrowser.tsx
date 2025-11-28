import { useState } from 'react'
import { useAgentTools } from '../hooks/useAgentTools'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'

export function ToolBrowser() {
  const { data, isLoading } = useAgentTools()
  const [searchQuery, setSearchQuery] = useState('')

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">加載工具列表中...</p>
        </CardContent>
      </Card>
    )
  }

  const tools = data?.tools || []
  const filteredTools = tools.filter(
    (tool) =>
      tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const categories = data?.categories || []

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>可用工具 ({tools.length})</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 搜索框 */}
        <Input
          placeholder="搜索工具..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        {/* 分類標籤 */}
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => (
              <span
                key={category}
                className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs"
              >
                {category}
              </span>
            ))}
          </div>
        )}

        {/* 工具列表 */}
        {filteredTools.length === 0 ? (
          <p className="text-center text-slate-400 py-8">沒有找到相關工具</p>
        ) : (
          <div className="space-y-2">
            {filteredTools.map((tool) => (
              <div
                key={tool.name}
                className="p-4 bg-slate-800/50 rounded hover:bg-slate-800 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-10 h-10 rounded bg-primary/20 text-primary flex items-center justify-center text-xl">
                    🛠️
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold">{tool.name}</h4>
                      {tool.category && (
                        <span className="text-xs px-2 py-0.5 bg-blue-900/30 text-blue-300 rounded">
                          {tool.category}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-400 mb-2">{tool.description}</p>
                    {Object.keys(tool.parameters).length > 0 && (
                      <div className="text-xs text-slate-500">
                        <span className="font-semibold">參數：</span>
                        {Object.keys(tool.parameters).join(', ')}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
