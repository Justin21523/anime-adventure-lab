/**
 * RAG feature types
 */

export interface RAGDocument {
  doc_id: string
  filename: string
  world_id: string
  chunk_count: number
  metadata?: Record<string, any>
  created_at: string
}

export interface RAGSearchResult {
  doc_id: string
  chunk_id: string
  content: string
  score: number
  metadata?: Record<string, any>
}

export interface RAGStats {
  total_documents: number
  total_chunks: number
  total_vectors: number
  index_size_mb: number
  world_id: string
}

export interface RAGUploadRequest {
  file: File
  world_id?: string
  metadata?: Record<string, any>
}

export interface RAGSearchRequest {
  query: string
  world_id?: string
  top_k?: number
  score_threshold?: number
}

export interface RAGSearchResponse {
  results: RAGSearchResult[]
  query: string
  total_results: number
  search_time_ms: number
}

export interface RAGDocumentListResponse {
  documents: RAGDocument[]
  total: number
  world_id?: string
}
