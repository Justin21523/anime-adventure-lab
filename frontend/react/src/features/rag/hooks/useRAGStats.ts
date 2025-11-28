import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { RAGStats } from '../types/rag.types'

/**
 * Hook for fetching RAG statistics
 */
export function useRAGStats(worldId?: string) {
  return useQuery({
    queryKey: CACHE_KEYS.rag.stats(worldId),
    queryFn: async () => {
      const url = worldId ? `/rag/stats?world_id=${worldId}` : '/rag/stats'
      const response = await apiGet<RAGStats>(url)
      return response
    },
    staleTime: 30_000, // 30 seconds
  })
}
