import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { logger } from '@/utils/logger'
import { calculateBackoffDelay } from '@/lib/retry'

/**
 * Connection states for SSE
 */
export type SSEConnectionState =
  | 'idle'           // Not connected
  | 'connecting'     // Initial connection
  | 'connected'      // Successfully connected
  | 'disconnected'   // Disconnected (not reconnecting)
  | 'reconnecting'   // Attempting to reconnect
  | 'failed'         // Failed after max retries

/**
 * SSE Reconnect configuration
 */
export interface SSEReconnectConfig {
  /**
   * Enable automatic reconnection
   * @default true
   */
  enabled?: boolean

  /**
   * Maximum number of reconnect attempts
   * @default Infinity (keep trying forever)
   */
  maxAttempts?: number

  /**
   * Base delay for exponential backoff (ms)
   * @default 1000
   */
  baseDelay?: number

  /**
   * Maximum delay between reconnect attempts (ms)
   * @default 30000 (30 seconds)
   */
  maxDelay?: number

  /**
   * Backoff multiplier
   * @default 2
   */
  backoffMultiplier?: number

  /**
   * Enable jitter for backoff
   * @default true
   */
  enableJitter?: boolean

  /**
   * Pause reconnect when page is hidden
   * @default true
   */
  pauseOnHidden?: boolean

  /**
   * Callback when reconnect attempt starts
   */
  onReconnectAttempt?: (attempt: number, delay: number) => void

  /**
   * Callback when reconnection succeeds
   */
  onReconnectSuccess?: (attempt: number) => void

  /**
   * Callback when all reconnect attempts exhausted
   */
  onReconnectFailed?: () => void
}

/**
 * SSE options with reconnect support
 */
export interface UseSSEWithReconnectOptions<T> {
  /**
   * SSE endpoint URL
   */
  url: string

  /**
   * Request body (for POST requests)
   */
  body?: any

  /**
   * Additional headers
   */
  headers?: Record<string, string>

  /**
   * Whether to enable the connection
   * @default true
   */
  enabled?: boolean

  /**
   * Reconnect configuration
   */
  reconnect?: SSEReconnectConfig

  /**
   * Callback when a message is received
   */
  onMessage?: (data: T) => void

  /**
   * Callback when an error occurs
   */
  onError?: (error: Error) => void

  /**
   * Callback when the stream completes normally
   */
  onComplete?: () => void

  /**
   * Callback when connection state changes
   */
  onStateChange?: (state: SSEConnectionState) => void
}

/**
 * SSE hook return value
 */
export interface UseSSEWithReconnectReturn<T> {
  /**
   * Latest received data
   */
  data: T | null

  /**
   * Current connection state
   */
  state: SSEConnectionState

  /**
   * Whether currently connected
   */
  isConnected: boolean

  /**
   * Whether currently reconnecting
   */
  isReconnecting: boolean

  /**
   * Current error (if any)
   */
  error: Error | null

  /**
   * Current reconnect attempt number (0 if not reconnecting)
   */
  reconnectAttempt: number

  /**
   * Manually start the connection
   */
  start: () => void

  /**
   * Stop the connection (disables auto-reconnect)
   */
  stop: () => void

  /**
   * Manually trigger reconnect
   */
  reconnect: () => void
}

/**
 * Default reconnect configuration
 */
const DEFAULT_RECONNECT_CONFIG: Required<SSEReconnectConfig> = {
  enabled: true,
  maxAttempts: Infinity,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  enableJitter: true,
  pauseOnHidden: true,
  onReconnectAttempt: () => {},
  onReconnectSuccess: () => {},
  onReconnectFailed: () => {},
}

/**
 * Enhanced SSE hook with automatic reconnection
 *
 * @example
 * ```tsx
 * const { data, state, isConnected, reconnectAttempt } = useSSEWithReconnect({
 *   url: '/api/stream',
 *   body: { query: 'hello' },
 *   reconnect: {
 *     maxAttempts: 5,
 *     baseDelay: 2000,
 *   },
 *   onMessage: (msg) => console.log(msg),
 * })
 * ```
 */
export function useSSEWithReconnect<T = any>(
  options: UseSSEWithReconnectOptions<T>
): UseSSEWithReconnectReturn<T> {
  const {
    url,
    body,
    headers = {},
    enabled = true,
    reconnect: reconnectConfig = {},
    onMessage,
    onError,
    onComplete,
    onStateChange,
  } = options

  // Merge reconnect config with defaults
  const config = { ...DEFAULT_RECONNECT_CONFIG, ...reconnectConfig }

  // State
  const [data, setData] = useState<T | null>(null)
  const [state, setState] = useState<SSEConnectionState>('idle')
  const [error, setError] = useState<Error | null>(null)
  const [reconnectAttempt, setReconnectAttempt] = useState(0)

  // Refs
  const controllerRef = useRef<AbortController | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isManualStopRef = useRef(false)
  const isPageHiddenRef = useRef(false)

  /**
   * Update connection state
   */
  const updateState = useCallback(
    (newState: SSEConnectionState) => {
      setState(newState)
      onStateChange?.(newState)
      logger.debug('SSE state changed', { state: newState, url })
    },
    [url, onStateChange]
  )

  /**
   * Clear reconnect timeout
   */
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  /**
   * Stop connection
   */
  const stop = useCallback(() => {
    isManualStopRef.current = true
    clearReconnectTimeout()

    if (controllerRef.current) {
      controllerRef.current.abort()
      controllerRef.current = null
    }

    setReconnectAttempt(0)
    updateState('disconnected')
    logger.info('SSE connection stopped', { url })
  }, [url, updateState, clearReconnectTimeout])

  /**
   * Schedule reconnect
   */
  const scheduleReconnect = useCallback(
    (attempt: number) => {
      // Check if reconnect is enabled
      if (!config.enabled) {
        logger.debug('Reconnect disabled', { url })
        return
      }

      // Check if max attempts reached
      if (attempt >= config.maxAttempts) {
        logger.warn('Max reconnect attempts reached', {
          url,
          attempts: attempt,
          maxAttempts: config.maxAttempts,
        })
        updateState('failed')
        config.onReconnectFailed()
        return
      }

      // Don't reconnect if manually stopped
      if (isManualStopRef.current) {
        logger.debug('Reconnect cancelled (manual stop)', { url })
        return
      }

      // Don't reconnect if page is hidden (if pauseOnHidden is enabled)
      if (config.pauseOnHidden && isPageHiddenRef.current) {
        logger.debug('Reconnect paused (page hidden)', { url })
        return
      }

      // Calculate delay with exponential backoff
      const delay = calculateBackoffDelay(attempt, {
        baseDelay: config.baseDelay,
        maxDelay: config.maxDelay,
        backoffMultiplier: config.backoffMultiplier,
        enableJitter: config.enableJitter,
      } as any)

      logger.info('Scheduling reconnect', {
        url,
        attempt: attempt + 1,
        delay,
        maxAttempts: config.maxAttempts,
      })

      setReconnectAttempt(attempt + 1)
      updateState('reconnecting')
      config.onReconnectAttempt(attempt + 1, delay)

      // Schedule reconnect
      clearReconnectTimeout()
      reconnectTimeoutRef.current = setTimeout(() => {
        connect(attempt + 1)
      }, delay)
    },
    [url, config, updateState, clearReconnectTimeout]
  )

  /**
   * Connect to SSE endpoint
   */
  const connect = useCallback(
    (attempt: number = 0) => {
      // Don't connect if disabled
      if (!enabled) {
        logger.debug('SSE connection disabled', { url })
        return
      }

      // Don't connect if manually stopped
      if (isManualStopRef.current) {
        logger.debug('SSE connection cancelled (manual stop)', { url })
        return
      }

      // Clean up previous connection
      if (controllerRef.current) {
        controllerRef.current.abort()
      }

      // Create new AbortController
      const controller = new AbortController()
      controllerRef.current = controller

      // Update state
      updateState(attempt === 0 ? 'connecting' : 'reconnecting')

      logger.info('Connecting to SSE', {
        url,
        attempt: attempt > 0 ? attempt : undefined,
      })

      // Connect using fetchEventSource
      fetchEventSource(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,

        onopen: async (response) => {
          if (response.ok) {
            logger.info('SSE connection opened', {
              url,
              status: response.status,
            })

            setError(null)
            setReconnectAttempt(0)
            updateState('connected')

            // Notify reconnect success if this was a reconnect
            if (attempt > 0) {
              config.onReconnectSuccess(attempt)
            }
          } else {
            const errorMsg = `SSE connection failed: ${response.status} ${response.statusText}`
            logger.error('SSE connection failed', {
              url,
              status: response.status,
              statusText: response.statusText,
            })
            throw new Error(errorMsg)
          }
        },

        onmessage: (event) => {
          try {
            const parsed = JSON.parse(event.data) as T
            setData(parsed)

            if (onMessage) {
              onMessage(parsed)
            }

            // Check for completion
            if ('done' in (parsed as any) && (parsed as any).done) {
              logger.info('SSE stream completed', { url })
              if (onComplete) {
                onComplete()
              }
              stop()
            }
          } catch (err: any) {
            logger.error('Failed to parse SSE message', {
              url,
              error: err,
              data: event.data,
            })
          }
        },

        onerror: (err) => {
          const error = err instanceof Error ? err : new Error('SSE connection error')

          logger.error('SSE connection error', {
            url,
            error: error.message,
            attempt,
          })

          setError(error)

          if (onError) {
            onError(error)
          }

          // Don't throw - allow reconnect logic to handle it
          // The library will call onclose automatically
        },

        onclose: () => {
          logger.info('SSE connection closed', { url, attempt })

          // If connection was established and then closed unexpectedly, try to reconnect
          if (state === 'connected' && !isManualStopRef.current) {
            logger.info('SSE connection lost, attempting reconnect', { url })
            scheduleReconnect(attempt)
          } else if (!isManualStopRef.current) {
            // Initial connection failed, try to reconnect
            scheduleReconnect(attempt)
          } else {
            updateState('disconnected')
          }
        },
      }).catch((err) => {
        // Only log if not aborted
        if (err.name !== 'AbortError') {
          logger.error('SSE connection error (uncaught)', {
            url,
            error: err.message,
            attempt,
          })
        }
      })
    },
    [url, body, headers, enabled, state, config, onMessage, onError, onComplete, updateState, scheduleReconnect, stop]
  )

  /**
   * Start connection
   */
  const start = useCallback(() => {
    isManualStopRef.current = false
    setReconnectAttempt(0)
    connect(0)
  }, [connect])

  /**
   * Manually trigger reconnect
   */
  const reconnectManual = useCallback(() => {
    logger.info('Manual reconnect triggered', { url })
    setReconnectAttempt(0)
    connect(0)
  }, [url, connect])

  /**
   * Handle page visibility change
   */
  useEffect(() => {
    if (!config.pauseOnHidden) return

    const handleVisibilityChange = () => {
      const isHidden = document.hidden
      isPageHiddenRef.current = isHidden

      logger.debug('Page visibility changed', { hidden: isHidden, url })

      if (!isHidden && state === 'reconnecting') {
        // Resume reconnection when page becomes visible
        logger.info('Page visible, resuming reconnect', { url })
        scheduleReconnect(reconnectAttempt - 1)
      } else if (isHidden) {
        // Pause reconnection when page is hidden
        logger.debug('Page hidden, pausing reconnect', { url })
        clearReconnectTimeout()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [url, state, reconnectAttempt, config.pauseOnHidden, scheduleReconnect, clearReconnectTimeout])

  /**
   * Auto-start on mount if enabled
   */
  useEffect(() => {
    if (enabled) {
      start()
    }

    // Cleanup on unmount
    return () => {
      clearReconnectTimeout()
      if (controllerRef.current) {
        controllerRef.current.abort()
        controllerRef.current = null
      }
    }
  }, [enabled, start, clearReconnectTimeout])

  return {
    data,
    state,
    isConnected: state === 'connected',
    isReconnecting: state === 'reconnecting',
    error,
    reconnectAttempt,
    start,
    stop,
    reconnect: reconnectManual,
  }
}
