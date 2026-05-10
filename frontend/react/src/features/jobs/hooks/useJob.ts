import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { JobCancelResponse, JobRecord } from '../types/job.types'

function isTerminalStatus(status: string): boolean {
  const s = String(status || '').toLowerCase()
  return s === 'completed' || s === 'failed' || s === 'cancelled'
}

export interface UseJobOptions {
  enabled?: boolean
  refetchIntervalMs?: number
  stopWhenTerminal?: boolean
}

export function useJob(jobId?: string | null, options?: UseJobOptions) {
  const queryClient = useQueryClient()
  const id = String(jobId || '').trim()
  const enabled = Boolean(id) && (options?.enabled ?? true)
  const refetchIntervalMs = options?.refetchIntervalMs ?? 2000
  const stopWhenTerminal = options?.stopWhenTerminal ?? true

  const query = useQuery({
    queryKey: CACHE_KEYS.jobs.job(id),
    queryFn: async () => {
      return apiGet<JobRecord>(`/jobs/${id}`, { retry: false })
    },
    enabled,
    refetchInterval: (q) => {
      if (!enabled) return false
      if (!refetchIntervalMs) return false
      const data = q.state.data as any
      const status = String(data?.status || '')
      if (stopWhenTerminal && status && isTerminalStatus(status)) return false
      return refetchIntervalMs
    },
  })

  const cancelJob = useMutation({
    mutationFn: async () => {
      if (!id) throw new Error('jobId is required')
      return apiDelete<JobCancelResponse>(`/jobs/${id}`, { retry: false })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.jobs.job(id) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.jobs.list() })
    },
  })

  return {
    job: query.data ?? null,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch: query.refetch,
    cancelJob,
  }
}

