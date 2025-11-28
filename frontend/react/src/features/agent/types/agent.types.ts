/**
 * Agent system types
 */

export interface AgentTool {
  name: string
  description: string
  parameters: Record<string, any>
  category?: string
}

export interface AgentTaskRequest {
  task_description: string
  max_iterations?: number
  tools?: string[]
  context?: Record<string, any>
}

export interface AgentTaskStep {
  step_number: number
  thought: string
  action: string
  action_input: any
  observation: string
  timestamp: string
}

export interface AgentTaskResult {
  task_id: string
  status: 'running' | 'completed' | 'failed'
  result?: string
  steps: AgentTaskStep[]
  tools_used: string[]
  total_iterations: number
  error?: string
  created_at: string
  completed_at?: string
}

export interface AgentToolsResponse {
  tools: AgentTool[]
  categories: string[]
  total: number
}
