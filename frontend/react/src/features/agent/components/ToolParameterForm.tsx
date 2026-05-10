import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import type { AgentTool } from '../types/agent.types'

interface ToolParameterFormProps {
  tools: AgentTool[]
  selectedToolNames: string[]
  parameters: Record<string, Record<string, any>>
  onParametersChange: (parameters: Record<string, Record<string, any>>) => void
}

/**
 * Dynamic parameter configuration form for selected tools
 * Renders appropriate input fields based on parameter schema
 */
export function ToolParameterForm({
  tools,
  selectedToolNames,
  parameters,
  onParametersChange,
}: ToolParameterFormProps) {
  const [localParameters, setLocalParameters] = useState<Record<string, Record<string, any>>>(parameters)

  // Get selected tools with their parameter schemas
  const selectedTools = tools.filter((tool) => selectedToolNames.includes(tool.name))

  // Update local state when external parameters change
  useEffect(() => {
    setLocalParameters(parameters)
  }, [parameters])

  // Handle parameter value change
  const handleParameterChange = (toolName: string, paramName: string, value: any) => {
    const updatedParams = {
      ...localParameters,
      [toolName]: {
        ...(localParameters[toolName] || {}),
        [paramName]: value,
      },
    }
    setLocalParameters(updatedParams)
    onParametersChange(updatedParams)
  }

  // Clear parameters for a specific tool
  const handleClearToolParams = (toolName: string) => {
    const updatedParams = { ...localParameters }
    delete updatedParams[toolName]
    setLocalParameters(updatedParams)
    onParametersChange(updatedParams)
  }

  // Infer parameter type from schema or value
  const getParameterType = (paramSchema: any): string => {
    if (typeof paramSchema === 'object' && paramSchema !== null) {
      if (paramSchema.type) return paramSchema.type
      if (paramSchema.enum) return 'enum'
      if (paramSchema.properties) return 'object'
    }
    return 'string'
  }

  // Get parameter description
  const getParameterDescription = (paramSchema: any): string => {
    if (typeof paramSchema === 'object' && paramSchema?.description) {
      return paramSchema.description
    }
    return ''
  }

  // Check if parameter is required
  const isParameterRequired = (paramSchema: any): boolean => {
    if (typeof paramSchema === 'object' && paramSchema?.required !== undefined) {
      return paramSchema.required
    }
    return false
  }

  // Get default value
  const getDefaultValue = (paramSchema: any, type: string): any => {
    if (typeof paramSchema === 'object' && paramSchema?.default !== undefined) {
      return paramSchema.default
    }

    switch (type) {
      case 'number':
      case 'integer':
        return 0
      case 'boolean':
        return false
      case 'array':
        return []
      case 'object':
        return {}
      default:
        return ''
    }
  }

  // Render appropriate input field based on parameter type
  const renderParameterInput = (tool: AgentTool, paramName: string, paramSchema: any) => {
    const type = getParameterType(paramSchema)
    const description = getParameterDescription(paramSchema)
    const required = isParameterRequired(paramSchema)
    const currentValue = localParameters[tool.name]?.[paramName] ?? getDefaultValue(paramSchema, type)

    const inputId = `${tool.name}-${paramName}`

    switch (type) {
      case 'boolean':
        return (
          <div key={paramName} className="flex items-center space-x-2">
            <Checkbox
              id={inputId}
              checked={currentValue}
              onCheckedChange={(checked) =>
                handleParameterChange(tool.name, paramName, checked)
              }
            />
            <Label htmlFor={inputId} className="cursor-pointer">
              {paramName}
              {required && <span className="text-red-400 ml-1">*</span>}
            </Label>
            {description && (
              <span className="text-xs text-slate-400">({description})</span>
            )}
          </div>
        )

      case 'number':
      case 'integer':
        return (
          <div key={paramName} className="space-y-1">
            <Label htmlFor={inputId}>
              {paramName}
              {required && <span className="text-red-400 ml-1">*</span>}
            </Label>
            {description && (
              <p className="text-xs text-slate-400 mb-1">{description}</p>
            )}
            <Input
              id={inputId}
              type="number"
              value={currentValue}
              onChange={(e) =>
                handleParameterChange(
                  tool.name,
                  paramName,
                  type === 'integer' ? parseInt(e.target.value) || 0 : parseFloat(e.target.value) || 0
                )
              }
              placeholder={`Enter ${paramName}`}
              step={type === 'integer' ? 1 : 0.1}
            />
          </div>
        )

      case 'enum':
        const enumValues = (paramSchema.enum || []) as string[]
        return (
          <div key={paramName} className="space-y-1">
            <Label htmlFor={inputId}>
              {paramName}
              {required && <span className="text-red-400 ml-1">*</span>}
            </Label>
            {description && (
              <p className="text-xs text-slate-400 mb-1">{description}</p>
            )}
            <Select
              value={currentValue}
              onValueChange={(value) =>
                handleParameterChange(tool.name, paramName, value)
              }
            >
              <SelectTrigger id={inputId}>
                <SelectValue placeholder={`Select ${paramName}`} />
              </SelectTrigger>
              <SelectContent>
                {enumValues.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )

      case 'array':
      case 'object':
        return (
          <div key={paramName} className="space-y-1">
            <Label htmlFor={inputId}>
              {paramName}
              {required && <span className="text-red-400 ml-1">*</span>}
              <Badge variant="outline" className="ml-2 text-xs">
                {type === 'array' ? 'JSON Array' : 'JSON Object'}
              </Badge>
            </Label>
            {description && (
              <p className="text-xs text-slate-400 mb-1">{description}</p>
            )}
            <Textarea
              id={inputId}
              value={
                typeof currentValue === 'string'
                  ? currentValue
                  : JSON.stringify(currentValue, null, 2)
              }
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value)
                  handleParameterChange(tool.name, paramName, parsed)
                } catch {
                  // Keep as string if invalid JSON
                  handleParameterChange(tool.name, paramName, e.target.value)
                }
              }}
              placeholder={`Enter ${type} as JSON`}
              rows={4}
              className="font-mono text-sm"
            />
          </div>
        )

      default:
        // String or text
        return (
          <div key={paramName} className="space-y-1">
            <Label htmlFor={inputId}>
              {paramName}
              {required && <span className="text-red-400 ml-1">*</span>}
            </Label>
            {description && (
              <p className="text-xs text-slate-400 mb-1">{description}</p>
            )}
            <Input
              id={inputId}
              type="text"
              value={currentValue}
              onChange={(e) =>
                handleParameterChange(tool.name, paramName, e.target.value)
              }
              placeholder={`Enter ${paramName}`}
            />
          </div>
        )
    }
  }

  if (selectedTools.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">請先選擇工具以配置參數</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>工具參數配置</CardTitle>
          <Badge variant="outline">
            {selectedTools.length} 個工具
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {selectedTools.map((tool) => {
          const paramEntries = Object.entries(tool.parameters)
          const hasParams = paramEntries.length > 0

          return (
            <div
              key={tool.name}
              className="p-4 border border-slate-700 rounded-lg space-y-4"
            >
              {/* Tool header */}
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-primary">{tool.name}</h4>
                    {tool.category && (
                      <Badge variant="outline" className="text-xs">
                        {tool.category}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-slate-400">{tool.description}</p>
                </div>
                {hasParams && localParameters[tool.name] && (
                  <button
                    onClick={() => handleClearToolParams(tool.name)}
                    className="text-xs text-slate-400 hover:text-primary transition-colors"
                  >
                    重置
                  </button>
                )}
              </div>

              {/* Parameters */}
              {hasParams ? (
                <div className="space-y-3 pl-4 border-l-2 border-primary/30">
                  {paramEntries.map(([paramName, paramSchema]) =>
                    renderParameterInput(tool, paramName, paramSchema)
                  )}
                </div>
              ) : (
                <p className="text-sm text-slate-500 italic">此工具無需參數</p>
              )}
            </div>
          )
        })}

        {/* Summary */}
        <div className="pt-4 border-t border-slate-700">
          <p className="text-xs text-slate-400">
            提示：參數會隨著工具一起傳遞給 Agent。未填寫的參數將使用默認值。
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
