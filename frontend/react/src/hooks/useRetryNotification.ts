import { useCallback, useRef } from 'react'
import { useToast } from '@/components/ui'
import type { AxiosError } from 'axios'
import type { RetryConfig } from '@/lib/retry'

/**
 * Hook to show toast notifications for retry attempts
 */
export function useRetryNotification() {
  const { toast, dismiss } = useToast()
  const retryToastIdRef = useRef<string | null>(null)

  /**
   * Create a retry config with toast notifications
   */
  const createRetryConfigWithNotification = useCallback(
    (baseConfig: RetryConfig = {}): RetryConfig => {
      return {
        ...baseConfig,
        onRetry: (attemptNumber: number, error: AxiosError, delay: number) => {
          // Call base onRetry if provided
          baseConfig.onRetry?.(attemptNumber, error, delay)

          const maxRetries = baseConfig.maxRetries || 3
          const url = error.config?.url || 'unknown'
          const method = error.config?.method?.toUpperCase() || 'GET'
          const status = error.response?.status

          // Dismiss previous retry toast if exists
          if (retryToastIdRef.current) {
            dismiss(retryToastIdRef.current)
          }

          // Show retry notification
          const { id } = toast({
            title: '正在重試...',
            description: `${method} ${url} 失敗 (${
              status ? `狀態: ${status}` : '網絡錯誤'
            })，正在進行第 ${attemptNumber}/${maxRetries} 次重試...`,
            variant: 'default',
            duration: delay + 1000, // Show for slightly longer than the delay
          })

          retryToastIdRef.current = id
        },
      }
    },
    [toast, dismiss]
  )

  /**
   * Show success notification after successful retry
   */
  const showRetrySuccess = useCallback(
    (attemptNumber: number) => {
      // Dismiss retry toast
      if (retryToastIdRef.current) {
        dismiss(retryToastIdRef.current)
        retryToastIdRef.current = null
      }

      // Show success toast
      toast({
        title: '請求成功',
        description: `重試 ${attemptNumber} 次後成功`,
        variant: 'default',
        duration: 3000,
      })
    },
    [toast, dismiss]
  )

  /**
   * Show failure notification after all retries exhausted
   */
  const showRetryFailure = useCallback(
    (maxRetries: number, error?: string) => {
      // Dismiss retry toast
      if (retryToastIdRef.current) {
        dismiss(retryToastIdRef.current)
        retryToastIdRef.current = null
      }

      // Show error toast
      toast({
        title: '請求失敗',
        description: error || `已重試 ${maxRetries} 次，仍然失敗`,
        variant: 'destructive',
        duration: 5000,
      })
    },
    [toast, dismiss]
  )

  return {
    createRetryConfigWithNotification,
    showRetrySuccess,
    showRetryFailure,
  }
}

/**
 * Example usage:
 *
 * const { createRetryConfigWithNotification, showRetrySuccess, showRetryFailure } = useRetryNotification()
 *
 * try {
 *   const data = await apiGet('/endpoint', {
 *     retry: createRetryConfigWithNotification({
 *       maxRetries: 3,
 *       baseDelay: 1000,
 *     })
 *   })
 *   showRetrySuccess(1) // If needed
 * } catch (error) {
 *   showRetryFailure(3, error.message)
 * }
 */
