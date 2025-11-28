import { useMutation } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import type { AgentTaskRequest, AgentTaskResult } from '../types/agent.types'

/**
 * Hook for executing agent tasks
 */
export function useAgentTask() {
  return useMutation({
    mutationFn: async (request: AgentTaskRequest) => {
      const response = await apiPost<AgentTaskResult>('/agent/task', {
        task_description: request.task_description,
        max_iterations: request.max_iterations || 5,
        tools: request.tools,
        context: request.context,
      })
      return response
    },
  })
}
