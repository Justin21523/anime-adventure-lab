import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'
import { AppError, handleApiError } from '@/lib/api-error'

/**
 * Base API client configuration
 */
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1',
  timeout: 30000,
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

    // Log request in development
    if (import.meta.env.DEV) {
      console.log('[API Request]', config.method?.toUpperCase(), config.url)
    }

    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

/**
 * Response interceptor - handle errors, logging
 */
apiClient.interceptors.response.use(
  (response) => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.log('[API Response]', response.status, response.config.url)
    }

    return response
  },
  (error) => {
    console.error('[API Error]', error.response?.status, error.response?.data)

    // Convert to AppError
    const appError = handleApiError(error)

    // Handle specific error codes
    if (appError.statusCode === 401) {
      // Unauthorized - clear auth and redirect to login
      localStorage.removeItem('auth_token')
      // window.location.href = '/login'
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
 * GET request
 */
export async function apiGet<T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.get<T>(url, config)
  return response.data
}

/**
 * POST request
 */
export async function apiPost<T, D = any>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.post<T>(url, data, config)
  return response.data
}

/**
 * PUT request
 */
export async function apiPut<T, D = any>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.put<T>(url, data, config)
  return response.data
}

/**
 * DELETE request
 */
export async function apiDelete<T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.delete<T>(url, config)
  return response.data
}

/**
 * File upload helper
 */
export async function apiUploadFile<T>(
  url: string,
  file: File,
  additionalData?: Record<string, any>,
  onProgress?: (progress: number) => void
): Promise<T> {
  const formData = new FormData()
  formData.append('file', file)

  // Add additional fields
  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, String(value))
    })
  }

  const response = await apiClient.post<T>(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = (progressEvent.loaded / progressEvent.total) * 100
        onProgress(Math.round(progress))
      }
    },
  })

  return response.data
}

export default apiClient
export { AppError }
