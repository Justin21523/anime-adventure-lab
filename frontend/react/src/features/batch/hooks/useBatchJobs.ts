import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { BatchJob, BatchJobListResponse, BatchSubmitRequest } from '../types/batch.types'

/**
 * Hook for managing batch jobs
 */
export function useBatchJobs() {
  const queryClient = useQueryClient()

  // Fetch all jobs
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: CACHE_KEYS.batch.jobs(),
    queryFn: async () => {
      const response = await apiGet<BatchJobListResponse>('/batch/list')
      return response
    },
    refetchInterval: 3000, // Poll every 3 seconds for active jobs
    staleTime: 1000,
  })

  // Submit new job
  const submitJob = useMutation({
    mutationFn: async (request: BatchSubmitRequest) => {
      const response = await apiPost<BatchJob>('/batch/submit', request)
      return response
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.batch.jobs() })
    },
  })

  // Cancel job
  const cancelJob = useMutation({
    mutationFn: async (jobId: string) => {
      await apiPost(`/batch/cancel/${jobId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.batch.jobs() })
    },
  })

  return {
    jobs: data?.jobs || [],
    total: data?.total || 0,
    isLoading,
    error,
    refetch,
    submitJob,
    cancelJob,
  }
}
