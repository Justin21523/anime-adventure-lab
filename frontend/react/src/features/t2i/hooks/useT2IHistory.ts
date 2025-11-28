import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { T2IHistoryResponse } from '../types/t2i.types'

/**
 * Hook for fetching T2I generation history
 * Can be filtered by session_id to show images for a specific story session
 */
export function useT2IHistory(sessionId?: string, limit: number = 20) {
  return useQuery({
    queryKey: CACHE_KEYS.t2i.history(sessionId),
    queryFn: async () => {
      const params = new URLSearchParams()
      if (sessionId) params.append('session_id', sessionId)
      params.append('limit', limit.toString())

      const response = await apiGet<T2IHistoryResponse>(
        `/t2i/history?${params.toString()}`
      )
      return response
    },
    staleTime: 30_000, // 30 seconds
  })
}
