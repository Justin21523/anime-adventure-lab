/**
 * Custom API error class that maps backend exceptions
 * Mirrors the backend exception hierarchy from core/exceptions.py
 */

export interface ApiErrorResponse {
  error_code?: string
  code?: string
  message?: string
  detail?: string
  details?: unknown
  errors?: unknown
  request_id?: string
  status_code?: number
}

export class AppError extends Error {
  public readonly errorCode: string
  public readonly details?: unknown
  public readonly statusCode: number
  public readonly requestId?: string

  constructor(response: ApiErrorResponse) {
    super(response.message || response.detail || 'Request failed')
    this.name = 'AppError'
    this.errorCode = response.error_code || response.code || 'UNKNOWN_ERROR'
    this.details = response.details || response.errors
    this.statusCode = response.status_code || 500
    this.requestId = response.request_id
  }

  /**
   * Check if error is a specific type
   */
  is(errorCode: string): boolean {
    return this.errorCode === errorCode
  }

  /**
   * Get user-friendly error message
   */
  getUserMessage(): string {
    const messages: Record<string, string> = {
      MODEL_NOT_FOUND: '模型未找到，請檢查配置',
      MODEL_LOAD_ERROR: '模型加載失敗',
      INFERENCE_ERROR: '推理過程出錯',
      RAG_ENGINE_ERROR: 'RAG 引擎錯誤',
      DOCUMENT_NOT_FOUND: '文檔未找到',
      EMBEDDING_ERROR: '嵌入生成失敗',
      BATCH_JOB_ERROR: '批次任務執行失敗',
      STORY_ENGINE_ERROR: '故事引擎錯誤',
      AGENT_EXECUTION_ERROR: 'Agent 執行失敗',
      VALIDATION_ERROR: '輸入驗證失敗',
      RESOURCE_NOT_FOUND: '資源未找到',
      PERMISSION_DENIED: '權限不足',
    }

    return messages[this.errorCode] || this.message
  }
}

/**
 * Type guard for API error response
 */
export function isApiError(error: unknown): error is ApiErrorResponse {
  return (
    !!error &&
    typeof error === 'object' &&
    ('error_code' in error || 'code' in error) &&
    ('message' in error || 'detail' in error)
  )
}

/**
 * Handle axios error and convert to AppError
 */
export function handleApiError(error: unknown): AppError {
  const candidate = error as {
    response?: { data?: unknown; status?: number }
    message?: string
  }
  if (candidate.response?.data && isApiError(candidate.response.data)) {
    return new AppError({
      ...candidate.response.data,
      status_code: candidate.response.status,
    })
  }

  // Fallback for non-API errors
  return new AppError({
    error_code: 'UNKNOWN_ERROR',
    message: candidate.message || 'An unknown error occurred',
    status_code: candidate.response?.status || 500,
  })
}
