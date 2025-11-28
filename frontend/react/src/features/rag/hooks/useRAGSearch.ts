import { useMutation } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import type { RAGSearchRequest, RAGSearchResponse } from '../types/rag.types'

/**
 * Hook for RAG search functionality
 */
export function useRAGSearch() {
  return useMutation({
    mutationFn: async (request: RAGSearchRequest) => {
      const response = await apiPost<RAGSearchResponse>('/rag/search', {
        query: request.query,
        world_id: request.world_id || 'default',
        top_k: request.top_k || 5,
        score_threshold: request.score_threshold,
      })
      return response
    },
  })
}
