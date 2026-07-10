import axios, { type AxiosRequestConfig, type AxiosError } from 'axios'
import { AppError, handleApiError } from '@/lib/api-error'
import { logger } from '@/utils/logger'
import { retryWithBackoff, type RetryConfig, createRetryConfigWithRateLimitRespect } from '@/lib/retry'

/**
 * Base API client configuration
 */
const apiClient = axios.create({
  // Use relative URL to leverage Vite proxy and avoid CORS issues in development
  // In production, VITE_API_BASE should be set to the actual API URL
  baseURL: import.meta.env.VITE_API_BASE || '/api/v1',
  timeout: 30000,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Request interceptor - add auth tokens, logging, etc.
 */
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    const method = config.method?.toUpperCase() || 'GET'
    if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
      const csrf = document.cookie
        .split('; ')
        .find((entry) => entry.startsWith('saga_csrf='))
        ?.split('=', 2)[1]
      if (csrf) {
        config.headers['X-CSRF-Token'] = decodeURIComponent(csrf)
      }
    }

    // Log request with structured logger
    logger.logApiRequest(
      method,
      config.url || '',
      {
        baseURL: config.baseURL,
      }
    )

    return config
  },
  (error) => {
    logger.error('API request failed to send', {
      error: error.message,
      url: error.config?.url,
    })
    return Promise.reject(error)
  }
)

/**
 * Response interceptor - handle errors, logging
 */
apiClient.interceptors.response.use(
  (response) => {
    // Log successful response
    logger.logApiResponse(
      response.config.method?.toUpperCase() || 'GET',
      response.config.url || '',
      response.status
    )

    return response
  },
  (error: AxiosError) => {
    const status = error.response?.status || 0
    const url = error.config?.url || 'unknown'
    const method = error.config?.method?.toUpperCase() || 'GET'

    // Log API error with full context
    logger.logApiResponse(method, url, status, {
      error: error.message,
      data: error.response?.data,
    })

    // Convert to AppError
    const appError = handleApiError(error)

    // Handle specific error codes
    if (appError.statusCode === 401) {
      // Unauthorized - clear auth and redirect to login
      logger.warn('Unauthorized access - clearing auth token', {
        url,
        method,
      })
      localStorage.removeItem('auth_token')
      // window.location.href = '/login'
    } else if (appError.statusCode >= 500) {
      // Server errors - log for monitoring
      logger.error('Server error encountered', {
        url,
        method,
        status,
        message: appError.message,
      })
    } else if (appError.statusCode >= 400) {
      // Client errors - log as warning
      logger.warn('Client error in API request', {
        url,
        method,
        status,
        message: appError.message,
      })
    }

    return Promise.reject(appError)
  }
)

/**
 * Type-safe API call wrapper
 * Will be used with openapi-typescript generated types
 */
export interface ApiResponse<T> {
  data: T
  status: number
  statusText: string
}

/**
 * Extended axios config with retry options
 */
export interface ApiRequestConfig extends AxiosRequestConfig {
  /**
   * Retry configuration for this request
   * Set to false to disable retry
   */
  retry?: RetryConfig | false
}

/**
 * GET request with automatic retry
 */
export async function apiGet<T>(
  url: string,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  // If retry is explicitly disabled, make request without retry
  if (retry === false) {
    const response = await apiClient.get<T>(url, axiosConfig)
    return response.data
  }

  // Use retry with backoff
  const retryConfig = createRetryConfigWithRateLimitRespect(retry || {})

  return retryWithBackoff(async () => {
    const response = await apiClient.get<T>(url, axiosConfig)
    return response.data
  }, retryConfig)
}

/**
 * POST request with automatic retry
 */
export async function apiPost<T, D = unknown>(
  url: string,
  data?: D,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  // Mutations are not retried unless the caller explicitly opts in. Retrying
  // session/job creation without an idempotency key can duplicate state.
  if (retry === undefined || retry === false) {
    const response = await apiClient.post<T>(url, data, axiosConfig)
    return response.data
  }

  // Use retry with backoff
  const retryConfig = createRetryConfigWithRateLimitRespect(retry || {})

  return retryWithBackoff(async () => {
    const response = await apiClient.post<T>(url, data, axiosConfig)
    return response.data
  }, retryConfig)
}

/**
 * PUT request with automatic retry
 */
export async function apiPut<T, D = unknown>(
  url: string,
  data?: D,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  if (retry === undefined || retry === false) {
    const response = await apiClient.put<T>(url, data, axiosConfig)
    return response.data
  }

  // Use retry with backoff
  const retryConfig = createRetryConfigWithRateLimitRespect(retry || {})

  return retryWithBackoff(async () => {
    const response = await apiClient.put<T>(url, data, axiosConfig)
    return response.data
  }, retryConfig)
}

/**
 * PATCH request with automatic retry
 */
export async function apiPatch<T, D = unknown>(
  url: string,
  data?: D,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  if (retry === undefined || retry === false) {
    const response = await apiClient.patch<T>(url, data, axiosConfig)
    return response.data
  }

  const retryConfig = createRetryConfigWithRateLimitRespect(retry || {})

  return retryWithBackoff(async () => {
    const response = await apiClient.patch<T>(url, data, axiosConfig)
    return response.data
  }, retryConfig)
}

/**
 * DELETE request with automatic retry
 */
export async function apiDelete<T>(
  url: string,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  if (retry === undefined || retry === false) {
    const response = await apiClient.delete<T>(url, axiosConfig)
    return response.data
  }

  // Use retry with backoff
  const retryConfig = createRetryConfigWithRateLimitRespect(retry || {})

  return retryWithBackoff(async () => {
    const response = await apiClient.delete<T>(url, axiosConfig)
    return response.data
  }, retryConfig)
}

/**
 * File upload helper with automatic retry
 * Note: File uploads generally should not be retried automatically as they can be large
 * and retrying may cause duplicate uploads. Retry is disabled by default but can be enabled.
 */
export async function apiUploadFile<T>(
  url: string,
  file: File,
  additionalData?: Record<string, unknown>,
  onProgress?: (progress: number) => void,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  const formData = new FormData()
  formData.append('file', file)

  // Add additional fields
  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, String(value))
    })
  }

  const uploadConfig: AxiosRequestConfig = {
    ...axiosConfig,
    headers: {
      'Content-Type': 'multipart/form-data',
      ...axiosConfig.headers,
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = (progressEvent.loaded / progressEvent.total) * 100
        onProgress(Math.round(progress))
      }
    },
  }

  // Retry is disabled by default for file uploads
  if (retry === undefined || retry === false) {
    const response = await apiClient.post<T>(url, formData, uploadConfig)
    return response.data
  }

  // Use retry with backoff if explicitly enabled
  const retryConfig = createRetryConfigWithRateLimitRespect(retry)

  return retryWithBackoff(async () => {
    const response = await apiClient.post<T>(url, formData, uploadConfig)
    return response.data
  }, retryConfig)
}

/**
 * Multi-file upload helper with progress reporting.
 * Uses field name `files` (FastAPI UploadFile list).
 */
export async function apiUploadFiles<T>(
  url: string,
  files: File[],
  additionalData?: Record<string, unknown>,
  onProgress?: (progress: number) => void,
  config?: ApiRequestConfig
): Promise<T> {
  const { retry, ...axiosConfig } = config || {}

  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, String(value))
    })
  }

  const uploadConfig: AxiosRequestConfig = {
    ...axiosConfig,
    headers: {
      'Content-Type': 'multipart/form-data',
      ...axiosConfig.headers,
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = (progressEvent.loaded / progressEvent.total) * 100
        onProgress(Math.round(progress))
      }
    },
  }

  if (retry === undefined || retry === false) {
    const response = await apiClient.post<T>(url, formData, uploadConfig)
    return response.data
  }

  const retryConfig = createRetryConfigWithRateLimitRespect(retry)
  return retryWithBackoff(async () => {
    const response = await apiClient.post<T>(url, formData, uploadConfig)
    return response.data
  }, retryConfig)
}

export default apiClient
export { AppError }
