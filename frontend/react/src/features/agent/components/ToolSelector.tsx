import { useState, useMemo } from 'react'
import { useAgentTools } from '../hooks/useAgentTools'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import type { AgentTool } from '../types/agent.types'

interface ToolSelectorProps {
  selectedTools: string[]
  onToolsChange: (tools: string[]) => void
  maxSelection?: number
}

/**
 * Tool selector component with multi-select capability
 * Used for selecting which tools the agent can use during task execution
 */
export function ToolSelector({ selectedTools, onToolsChange, maxSelection }: ToolSelectorProps) {
  const { data, isLoading } = useAgentTools()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const tools = data?.tools || []
  const categories = data?.categories || []

  // Filter tools based on search and category
  const filteredTools = useMemo(() => {
    return tools.filter((tool) => {
      const matchesSearch =
        tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tool.description.toLowerCase().includes(searchQuery.toLowerCase())

      const matchesCategory =
        !selectedCategory || tool.category === selectedCategory

      return matchesSearch && matchesCategory
    })
  }, [tools, searchQuery, selectedCategory])

  // Handle tool selection toggle
  const handleToggleTool = (toolName: string) => {
    const isSelected = selectedTools.includes(toolName)

    if (isSelected) {
      // Remove tool
      onToolsChange(selectedTools.filter((t) => t !== toolName))
    } else {
      // Add tool (check max selection limit)
      if (maxSelection && selectedTools.length >= maxSelection) {
        return // Don't add if max reached
      }
      onToolsChange([...selectedTools, toolName])
    }
  }

  // Select all filtered tools
  const handleSelectAll = () => {
    const allFilteredToolNames = filteredTools.map((t) => t.name)
    const newSelection = Array.from(new Set([...selectedTools, ...allFilteredToolNames]))

    if (maxSelection) {
      onToolsChange(newSelection.slice(0, maxSelection))
    } else {
      onToolsChange(newSelection)
    }
  }

  // Clear all selections
  const handleClearAll = () => {
    onToolsChange([])
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">加載工具列表中...</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>
            選擇工具
            {maxSelection && (
              <span className="ml-2 text-sm font-normal text-slate-400">
                ({selectedTools.length}/{maxSelection})
              </span>
            )}
            {!maxSelection && selectedTools.length > 0 && (
              <span className="ml-2 text-sm font-normal text-slate-400">
                ({selectedTools.length} 已選)
              </span>
            )}
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectAll}
              disabled={filteredTools.length === 0}
            >
              全選
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearAll}
              disabled={selectedTools.length === 0}
            >
              清空
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Search box */}
        <Input
          placeholder="搜索工具..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        {/* Category filters */}
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={cn(
                'px-3 py-1 rounded text-sm transition-colors',
                !selectedCategory
                  ? 'bg-primary text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              )}
            >
              全部
            </button>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={cn(
                  'px-3 py-1 rounded text-sm transition-colors',
                  selectedCategory === category
                    ? 'bg-primary text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                )}
              >
                {category}
              </button>
            ))}
          </div>
        )}

        {/* Tools list with checkboxes */}
        {filteredTools.length === 0 ? (
          <p className="text-center text-slate-400 py-8">沒有找到相關工具</p>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {filteredTools.map((tool) => {
              const isSelected = selectedTools.includes(tool.name)
              const isDisabled =
                maxSelection &&
                !isSelected &&
                selectedTools.length >= maxSelection

              return (
                <div
                  key={tool.name}
                  onClick={() => !isDisabled && handleToggleTool(tool.name)}
                  className={cn(
                    'p-4 rounded cursor-pointer transition-colors',
                    isSelected
                      ? 'bg-primary/20 border-2 border-primary'
                      : 'bg-slate-800/50 border-2 border-transparent hover:bg-slate-800',
                    isDisabled && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <div className="flex items-start gap-3">
                    {/* Checkbox */}
                    <Checkbox
                      checked={isSelected}
                      disabled={!!isDisabled}
                      onCheckedChange={() => handleToggleTool(tool.name)}
                      className="mt-1"
                    />

                    {/* Tool icon */}
                    <div className="flex-shrink-0 w-10 h-10 rounded bg-primary/20 text-primary flex items-center justify-center text-xl">
                      🛠️
                    </div>

                    {/* Tool details */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold truncate">{tool.name}</h4>
                        {tool.category && (
                          <span className="text-xs px-2 py-0.5 bg-blue-900/30 text-blue-300 rounded flex-shrink-0">
                            {tool.category}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-slate-400 mb-2 line-clamp-2">
                        {tool.description}
                      </p>
                      {Object.keys(tool.parameters).length > 0 && (
                        <div className="text-xs text-slate-500">
                          <span className="font-semibold">參數：</span>
                          <span className="truncate">
                            {Object.keys(tool.parameters).join(', ')}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Selected tools summary */}
        {selectedTools.length > 0 && (
          <div className="pt-4 border-t border-slate-700">
            <Label className="text-sm font-semibold mb-2 block">
              已選工具 ({selectedTools.length})
            </Label>
            <div className="flex flex-wrap gap-2">
              {selectedTools.map((toolName) => (
                <span
                  key={toolName}
                  className="px-3 py-1 bg-primary/20 text-primary rounded-full text-sm flex items-center gap-2"
                >
                  {toolName}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleToggleTool(toolName)
                    }}
                    className="hover:text-primary-light transition-colors"
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
