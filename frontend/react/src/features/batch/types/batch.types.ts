/**
 * Batch processing types
 */

export type BatchJobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type BatchJobType = 't2i' | 'caption' | 'embedding' | 'training' | 'rag_indexing'

export interface BatchJob {
  job_id: string
  job_type: BatchJobType
  status: BatchJobStatus
  progress?: {
    current: number
    total: number
    percentage: number
  }
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  result_path?: string
  metadata?: Record<string, any>
}

export interface BatchJobListResponse {
  jobs: BatchJob[]
  total: number
}

export interface BatchSubmitRequest {
  job_type: BatchJobType
  config: Record<string, any>
  priority?: number
}

export interface BatchJobResult {
  job_id: string
  status: string
  results: any[]
  output_path?: string
}
