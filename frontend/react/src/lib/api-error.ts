/**
 * Custom API error class that maps backend exceptions
 * Mirrors the backend exception hierarchy from core/exceptions.py
 */

export interface ApiErrorResponse {
  error_code: string
  message: string
  details?: any
  status_code?: number
}

export class AppError extends Error {
  public readonly errorCode: string
  public readonly details?: any
  public readonly statusCode: number

  constructor(response: ApiErrorResponse) {
    super(response.message)
    this.name = 'AppError'
    this.errorCode = response.error_code
    this.details = response.details
    this.statusCode = response.status_code || 500
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
export function isApiError(error: any): error is ApiErrorResponse {
  return (
    error &&
    typeof error === 'object' &&
    'error_code' in error &&
    'message' in error
  )
}

/**
 * Handle axios error and convert to AppError
 */
export function handleApiError(error: any): AppError {
  if (error.response?.data && isApiError(error.response.data)) {
    return new AppError(error.response.data)
  }

  // Fallback for non-API errors
  return new AppError({
    error_code: 'UNKNOWN_ERROR',
    message: error.message || 'An unknown error occurred',
    status_code: error.response?.status || 500,
  })
}
