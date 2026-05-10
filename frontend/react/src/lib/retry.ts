import { AxiosError } from 'axios'
import { logger } from '@/utils/logger'

/**
 * Retry configuration options
 */
export interface RetryConfig {
  /**
   * Maximum number of retry attempts
   * @default 3
   */
  maxRetries?: number

  /**
   * Base delay in milliseconds for exponential backoff
   * @default 1000
   */
  baseDelay?: number

  /**
   * Maximum delay in milliseconds
   * @default 30000 (30 seconds)
   */
  maxDelay?: number

  /**
   * Backoff multiplier for exponential backoff
   * @default 2
   */
  backoffMultiplier?: number

  /**
   * Whether to add jitter to prevent thundering herd
   * @default true
   */
  enableJitter?: boolean

  /**
   * Custom function to determine if error should be retried
   */
  shouldRetry?: (error: AxiosError) => boolean

  /**
   * Callback when retry attempt is made
   */
  onRetry?: (attemptNumber: number, error: AxiosError, delay: number) => void
}

/**
 * Default retry configuration
 */
export const DEFAULT_RETRY_CONFIG: Required<RetryConfig> = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  enableJitter: true,
  shouldRetry: defaultShouldRetry,
  onRetry: () => {},
}

/**
 * Default retry strategy - retry on network errors and 5xx server errors
 */
export function defaultShouldRetry(error: AxiosError): boolean {
  // Don't retry if no response (request was cancelled)
  if (error.code === 'ECONNABORTED' || error.message === 'canceled') {
    return false
  }

  // Retry on network errors
  if (!error.response) {
    return true
  }

  const status = error.response.status

  // Retry on specific server errors
  if (status >= 500 && status < 600) {
    return true
  }

  // Retry on specific client errors
  if (status === 408 || status === 429) {
    // 408 Request Timeout
    // 429 Too Many Requests
    return true
  }

  // Don't retry on other errors
  return false
}

/**
 * Calculate exponential backoff delay with optional jitter
 */
export function calculateBackoffDelay(
  attemptNumber: number,
  config: Required<RetryConfig>
): number {
  // Calculate exponential delay: baseDelay * (backoffMultiplier ^ attemptNumber)
  const exponentialDelay =
    config.baseDelay * Math.pow(config.backoffMultiplier, attemptNumber)

  // Cap at max delay
  let delay = Math.min(exponentialDelay, config.maxDelay)

  // Add jitter to prevent thundering herd
  if (config.enableJitter) {
    // Random jitter between 0 and delay
    delay = Math.random() * delay
  }

  return Math.floor(delay)
}

/**
 * Sleep utility for async delay
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * Retry wrapper for async functions with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig = {}
): Promise<T> {
  const finalConfig: Required<RetryConfig> = {
    ...DEFAULT_RETRY_CONFIG,
    ...config,
  }

  let lastError: any

  for (let attempt = 0; attempt <= finalConfig.maxRetries; attempt++) {
    try {
      // Try the function
      return await fn()
    } catch (error: any) {
      lastError = error

      // Check if we should retry
      const shouldRetry = finalConfig.shouldRetry(error)

      // If this is the last attempt or we shouldn't retry, throw
      if (attempt === finalConfig.maxRetries || !shouldRetry) {
        logger.warn('Retry exhausted or error not retryable', {
          attempt: attempt + 1,
          maxRetries: finalConfig.maxRetries,
          shouldRetry,
          error: error.message,
        })
        throw error
      }

      // Calculate delay for next retry
      const delay = calculateBackoffDelay(attempt, finalConfig)

      // Log retry attempt
      logger.info('Retrying request', {
        attempt: attempt + 1,
        maxRetries: finalConfig.maxRetries,
        delay,
        error: error.message,
        status: error.response?.status,
      })

      // Call retry callback
      finalConfig.onRetry(attempt + 1, error, delay)

      // Wait before retrying
      await sleep(delay)
    }
  }

  // This should never be reached, but TypeScript needs it
  throw lastError
}

/**
 * Extract retry-after header from response
 * Returns delay in milliseconds
 */
export function getRetryAfterDelay(error: AxiosError): number | null {
  const retryAfter = error.response?.headers['retry-after']

  if (!retryAfter) {
    return null
  }

  // retry-after can be in seconds or HTTP date
  const retryAfterNum = parseInt(retryAfter, 10)

  if (!isNaN(retryAfterNum)) {
    // It's in seconds, convert to milliseconds
    return retryAfterNum * 1000
  }

  // Try to parse as date
  const retryAfterDate = new Date(retryAfter)
  if (!isNaN(retryAfterDate.getTime())) {
    return Math.max(0, retryAfterDate.getTime() - Date.now())
  }

  return null
}

/**
 * Create a retry config that respects Retry-After headers
 */
export function createRetryConfigWithRateLimitRespect(
  baseConfig: RetryConfig = {}
): RetryConfig {
  return {
    ...baseConfig,
    shouldRetry: (error: AxiosError) => {
      // Use custom shouldRetry if provided
      if (baseConfig.shouldRetry && !baseConfig.shouldRetry(error)) {
        return false
      }

      // Default retry logic
      return defaultShouldRetry(error)
    },
    onRetry: (attempt, error, delay) => {
      // Call base onRetry if provided
      baseConfig.onRetry?.(attempt, error, delay)

      // Check for Retry-After header
      const retryAfterDelay = getRetryAfterDelay(error)
      if (retryAfterDelay !== null) {
        logger.info('Respecting Retry-After header', {
          attempt,
          retryAfterDelay,
          originalDelay: delay,
        })
      }
    },
  }
}
