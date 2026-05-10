import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'

export interface RAGRebuildResponse {
  success: boolean
  job_id?: string
  status?: string
  message?: string
  time_taken_seconds?: number
  documents_processed?: number
}

export interface RAGClearWorldResponse {
  success: boolean
  world_id: string
  documents_removed: number
  chunks_removed: number
  message?: string
}

export function useRAGMaintenance(worldId?: string) {
  const queryClient = useQueryClient()

  const rebuildIndex = useMutation({
    mutationFn: async () => {
      return apiPost<RAGRebuildResponse>('/rag/rebuild')
    },
    onSuccess: () => {
      // Defer invalidation until the async job completes (handled by UI via /jobs polling).
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
    },
  })

  const clearWorld = useMutation({
    mutationFn: async () => {
      if (!worldId) {
        throw new Error('worldId is required')
      }
      return apiDelete<RAGClearWorldResponse>(`/rag/worlds/${worldId}/documents`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(worldId) })
    },
  })

  return { rebuildIndex, clearWorld }
}
