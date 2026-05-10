export type JobStatus =
  | 'queued'
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | string

export interface JobRecord {
  job_id: string
  status: JobStatus
  progress?: number
  job_type?: string
  created_at?: string
  started_at?: string
  finished_at?: string
  cancelled_at?: string
  error?: string
  result?: any
  payload?: any
  [key: string]: any
}

export interface JobCancelResponse {
  job_id: string
  cancelled: boolean
  revoked?: boolean
  job?: JobRecord
  [key: string]: any
}

