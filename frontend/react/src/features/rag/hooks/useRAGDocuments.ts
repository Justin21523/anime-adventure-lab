import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost, apiUploadFile } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { RAGDocument, RAGDocumentListResponse, RAGUploadRequest } from '../types/rag.types'

/**
 * Hook for managing RAG documents
 */
export function useRAGDocuments(worldId?: string) {
  const queryClient = useQueryClient()

  // Fetch all documents
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: CACHE_KEYS.rag.documents(worldId),
    queryFn: async () => {
      const url = worldId ? `/rag/documents?world_id=${worldId}` : '/rag/documents'
      const response = await apiGet<RAGDocumentListResponse>(url)
      return response
    },
    staleTime: 60_000, // 1 minute
  })

  // Upload document
  const uploadDocument = useMutation({
    mutationFn: async ({ file, world_id, metadata }: RAGUploadRequest) => {
      const response = await apiUploadFile<RAGDocument>(
        '/rag/upload',
        file,
        {
          world_id: world_id || 'default',
          ...(metadata && { metadata: JSON.stringify(metadata) }),
        }
      )
      return response
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(worldId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
    },
  })

  // Delete document
  const deleteDocument = useMutation({
    mutationFn: async (docId: string) => {
      await apiPost(`/rag/document/${docId}/delete`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(worldId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
    },
  })

  return {
    documents: data?.documents || [],
    total: data?.total || 0,
    isLoading,
    error,
    refetch,
    uploadDocument,
    deleteDocument,
  }
}
