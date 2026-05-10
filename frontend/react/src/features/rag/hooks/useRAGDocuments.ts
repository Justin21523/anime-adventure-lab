import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiGet, apiUploadFile, apiUploadFiles } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { RAGDocumentListResponse, RAGUploadBatchRequest, RAGUploadJobResponse, RAGUploadRequest } from '../types/rag.types'

/**
 * Hook for managing RAG documents
 */
export function useRAGDocuments(worldId?: string) {
  const queryClient = useQueryClient()
  const [uploadProgress, setUploadProgress] = useState<number>(0)

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
    mutationFn: async ({ file, world_id, tags }: RAGUploadRequest) => {
      const response = await apiUploadFile<RAGUploadJobResponse>(
        '/rag/upload_job',
        file,
        {
          world_id: world_id || 'default',
          ...(tags && tags.trim() ? { tags: tags.trim() } : {}),
        },
        // Progress callback
        (progress) => {
          setUploadProgress(Math.round(progress))
        }
      )
      return response
    },
    onSuccess: () => {
      setUploadProgress(0) // Reset progress
    },
    onError: () => {
      setUploadProgress(0) // Reset progress on error
    },
  })

  // Batch upload documents (multi-file or zip)
  const uploadDocumentsBatch = useMutation({
    mutationFn: async ({ files, world_id, tags }: RAGUploadBatchRequest) => {
      const response = await apiUploadFiles<RAGUploadJobResponse>(
        '/rag/upload_batch_job',
        files,
        {
          world_id: world_id || 'default',
          ...(tags && tags.trim() ? { tags: tags.trim() } : {}),
        },
        (progress) => {
          setUploadProgress(Math.round(progress))
        }
      )
      return response
    },
    onSuccess: () => {
      setUploadProgress(0)
    },
    onError: () => {
      setUploadProgress(0)
    },
  })

  // Delete document
  const deleteDocument = useMutation({
    mutationFn: async (docId: string) => {
      await apiDelete(`/rag/documents/${docId}`)
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
    uploadDocumentsBatch,
    uploadProgress,
    deleteDocument,
  }
}
