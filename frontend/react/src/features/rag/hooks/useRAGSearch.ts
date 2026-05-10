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
        filters: request.world_id ? { world_id: request.world_id } : undefined,
        parameters: {
          top_k: request.top_k || 5,
          ...(request.min_score !== undefined ? { min_score: request.min_score } : {}),
        },
      })
      return response
    },
  })
}
