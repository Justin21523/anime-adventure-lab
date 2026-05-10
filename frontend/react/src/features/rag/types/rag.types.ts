/**
 * RAG feature types
 */

export interface RAGDocumentInfo {
  doc_id: string
  title: string
  chunks: number
  total_chars: number
  created_at?: string | null
  metadata: Record<string, any>
}

export interface RAGAddDocumentResponse {
  doc_id: string
  added: boolean
  chunks_created?: number | null
  success?: boolean
  timestamp?: string
}

export interface RAGSearchResult {
  doc_id: string
  content: string
  score: number
  rank: number
  metadata: Record<string, any>
}

export interface RAGStats {
  total_documents: number
  total_chunks: number
  index_size: number
  embedding_model: string
  embedding_dim: number
  model_loaded: boolean
  success?: boolean
  timestamp?: string
}

export interface RAGUploadRequest {
  file: File
  world_id?: string
  tags?: string
}

export interface RAGUploadBatchRequest {
  files: File[]
  world_id?: string
  tags?: string
}

export interface RAGUploadJobResponse {
  success: boolean
  job_id: string
  status?: string
  [key: string]: any
}

export interface RAGSearchRequest {
  query: string
  world_id?: string
  top_k?: number
  min_score?: number
}

export interface RAGSearchResponse {
  results: RAGSearchResult[]
  query: string
  total_found: number
  success?: boolean
  timestamp?: string
}

export interface RAGDocumentListResponse {
  documents: RAGDocumentInfo[]
  total: number
  limit: number
  offset: number
  success?: boolean
}
