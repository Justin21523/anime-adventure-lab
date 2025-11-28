import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchEventSource } from '@microsoft/fetch-event-source'

/**
 * SSE (Server-Sent Events) hook for real-time streaming
 * Used for chat streaming, progress updates, etc.
 */

interface UseSSEOptions<T> {
  url: string
  body?: any
  enabled?: boolean
  onMessage?: (data: T) => void
  onError?: (error: Error) => void
  onComplete?: () => void
}

interface UseSSEReturn<T> {
  data: T | null
  isConnected: boolean
  error: Error | null
  start: () => void
  stop: () => void
}

export function useSSE<T = any>(options: UseSSEOptions<T>): UseSSEReturn<T> {
  const {
    url,
    body,
    enabled = true,
    onMessage,
    onError,
    onComplete,
  } = options

  const [data, setData] = useState<T | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const controllerRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort()
      controllerRef.current = null
      setIsConnected(false)
    }
  }, [])

  const start = useCallback(() => {
    if (!enabled) return

    // Clean up previous connection
    stop()

    // Create new AbortController
    const controller = new AbortController()
    controllerRef.current = controller

    // Connect to SSE endpoint
    fetchEventSource(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,

      onopen: async (response) => {
        if (response.ok) {
          setIsConnected(true)
          setError(null)
        } else {
          throw new Error(`SSE connection failed: ${response.status}`)
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
            if (onComplete) {
              onComplete()
            }
            stop()
          }
        } catch (err) {
          console.error('[SSE] Failed to parse message:', err)
        }
      },

      onerror: (err) => {
        const error = err instanceof Error ? err : new Error('SSE connection error')
        setError(error)
        setIsConnected(false)

        if (onError) {
          onError(error)
        }

        // Throw to stop retrying
        throw error
      },

      onclose: () => {
        setIsConnected(false)
        if (onComplete) {
          onComplete()
        }
      },
    }).catch((err) => {
      // Only log if not aborted
      if (err.name !== 'AbortError') {
        console.error('[SSE] Connection error:', err)
      }
    })
  }, [url, body, enabled, onMessage, onError, onComplete, stop])

  // Auto-start on mount if enabled
  useEffect(() => {
    if (enabled) {
      start()
    }

    // Cleanup on unmount
    return () => {
      stop()
    }
  }, [enabled, start, stop])

  return {
    data,
    isConnected,
    error,
    start,
    stop,
  }
}

/**
 * Chat streaming hook (specialized SSE hook)
 */
interface ChatStreamMessage {
  content: string
  done: boolean
  error?: string
}

export function useChatStream(
  messages: any[],
  enabled: boolean = false
) {
  const [fullContent, setFullContent] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

  const { data, isConnected, error, start, stop } = useSSE<ChatStreamMessage>({
    url: '/api/v1/chat/stream',
    body: { messages },
    enabled: false, // Manual control
    onMessage: (msg) => {
      if (!msg.done) {
        setFullContent((prev) => prev + msg.content)
      }
    },
    onComplete: () => {
      setIsStreaming(false)
    },
    onError: () => {
      setIsStreaming(false)
    },
  })

  const startStream = useCallback(() => {
    setFullContent('')
    setIsStreaming(true)
    start()
  }, [start])

  const stopStream = useCallback(() => {
    setIsStreaming(false)
    stop()
  }, [stop])

  return {
    content: fullContent,
    isStreaming: isStreaming && isConnected,
    error,
    startStream,
    stopStream,
  }
}
