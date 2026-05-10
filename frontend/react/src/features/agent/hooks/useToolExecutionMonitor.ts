import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { AgentTaskResult } from '../types/agent.types'

interface UseToolExecutionMonitorOptions {
  taskId?: string
  enabled?: boolean
  pollingInterval?: number
  onStatusChange?: (status: AgentTaskResult['status']) => void
  onComplete?: (result: AgentTaskResult) => void
  onError?: (error: string) => void
}

/**
 * Hook for monitoring tool execution in real-time
 * Polls the backend for task status and updates
 */
export function useToolExecutionMonitor({
  taskId,
  enabled = true,
  pollingInterval = 2000, // Poll every 2 seconds
  onStatusChange,
  onComplete,
  onError,
}: UseToolExecutionMonitorOptions) {
  const previousStatusRef = useRef<AgentTaskResult['status'] | undefined>(undefined)

  const {
    data: taskResult,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: CACHE_KEYS.agent.task(taskId || ''),
    queryFn: async () => {
      if (!taskId) return null
      const response = await apiGet<AgentTaskResult>(`/agent/task/${taskId}`)
      return response
    },
    enabled: enabled && !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data
      // Stop polling if task is completed or failed
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false
      }
      return pollingInterval
    },
    staleTime: 0, // Always fetch fresh data
  })

  // Monitor status changes and trigger callbacks
  useEffect(() => {
    if (!taskResult) return

    const currentStatus = taskResult.status

    // Trigger status change callback
    if (currentStatus !== previousStatusRef.current) {
      onStatusChange?.(currentStatus)
      previousStatusRef.current = currentStatus
    }

    // Trigger completion callback
    if (currentStatus === 'completed') {
      onComplete?.(taskResult)
    }

    // Trigger error callback
    if (currentStatus === 'failed' && taskResult.error) {
      onError?.(taskResult.error)
    }
  }, [taskResult, onStatusChange, onComplete, onError])

  return {
    taskResult,
    isLoading,
    error,
    refetch,
    isRunning: taskResult?.status === 'running',
    isCompleted: taskResult?.status === 'completed',
    isFailed: taskResult?.status === 'failed',
  }
}
