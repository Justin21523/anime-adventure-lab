import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { SSEConnectionState } from '@/hooks/useSSEWithReconnect'

export interface SSEConnectionStatusProps {
  /**
   * Current connection state
   */
  state: SSEConnectionState

  /**
   * Number of reconnect attempts (0 if not reconnecting)
   */
  reconnectAttempt?: number

  /**
   * Error message (if any)
   */
  error?: string | null

  /**
   * Whether to show controls
   */
  showControls?: boolean

  /**
   * Reconnect button handler
   */
  onReconnect?: () => void

  /**
   * Disconnect button handler
   */
  onDisconnect?: () => void

  /**
   * Component variant
   */
  variant?: 'full' | 'compact' | 'minimal'

  /**
   * Additional className
   */
  className?: string
}

/**
 * Get state badge variant
 */
function getStateBadgeVariant(state: SSEConnectionState): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (state) {
    case 'connected':
      return 'default'
    case 'connecting':
    case 'reconnecting':
      return 'secondary'
    case 'failed':
      return 'destructive'
    default:
      return 'outline'
  }
}

/**
 * Get state label
 */
function getStateLabel(state: SSEConnectionState): string {
  switch (state) {
    case 'idle':
      return '閒置'
    case 'connecting':
      return '連接中'
    case 'connected':
      return '已連接'
    case 'disconnected':
      return '已斷開'
    case 'reconnecting':
      return '重新連接中'
    case 'failed':
      return '連接失敗'
    default:
      return '未知'
  }
}

/**
 * Get state icon
 */
function getStateIcon(state: SSEConnectionState) {
  switch (state) {
    case 'connected':
      return (
        <svg
          className="w-4 h-4 text-green-500"
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'connecting':
    case 'reconnecting':
      return (
        <svg
          className="w-4 h-4 text-blue-500 animate-spin"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )
    case 'failed':
      return (
        <svg
          className="w-4 h-4 text-red-500"
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      )
    default:
      return (
        <svg
          className="w-4 h-4 text-slate-500"
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
            clipRule="evenodd"
          />
        </svg>
      )
  }
}

/**
 * Full variant - detailed status display
 */
function FullVariant({
  state,
  reconnectAttempt,
  error,
  showControls,
  onReconnect,
  onDisconnect,
  className,
}: SSEConnectionStatusProps) {
  const isReconnecting = state === 'reconnecting'
  const canReconnect = state === 'disconnected' || state === 'failed'
  const canDisconnect = state === 'connected' || state === 'connecting'

  return (
    <div className={cn('p-4 border border-slate-700 rounded-lg bg-slate-800/50 space-y-3', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {getStateIcon(state)}
          <div>
            <p className="text-sm font-medium text-slate-200">{getStateLabel(state)}</p>
            {isReconnecting && reconnectAttempt && reconnectAttempt > 0 && (
              <p className="text-xs text-slate-400">重試第 {reconnectAttempt} 次</p>
            )}
          </div>
        </div>
        <Badge variant={getStateBadgeVariant(state)}>{getStateLabel(state)}</Badge>
      </div>

      {/* Error message */}
      {error && (state === 'failed' || state === 'disconnected') && (
        <div className="flex items-start gap-2 p-2 text-xs text-red-400 bg-red-900/20 border border-red-500/30 rounded">
          <svg
            className="w-4 h-4 mt-0.5 flex-shrink-0"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Controls */}
      {showControls && (
        <div className="flex items-center gap-2">
          {canReconnect && onReconnect && (
            <Button size="sm" variant="outline" onClick={onReconnect}>
              重新連接
            </Button>
          )}
          {canDisconnect && onDisconnect && (
            <Button size="sm" variant="outline" onClick={onDisconnect}>
              斷開連接
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Compact variant - inline status indicator
 */
function CompactVariant({
  state,
  reconnectAttempt,
  onReconnect,
  className,
}: SSEConnectionStatusProps) {
  const isReconnecting = state === 'reconnecting'
  const canReconnect = state === 'disconnected' || state === 'failed'

  return (
    <div className={cn('inline-flex items-center gap-2', className)}>
      {getStateIcon(state)}
      <span className="text-sm text-slate-300">
        {getStateLabel(state)}
        {isReconnecting && reconnectAttempt && reconnectAttempt > 0 && ` (${reconnectAttempt})`}
      </span>
      {canReconnect && onReconnect && (
        <button
          onClick={onReconnect}
          className="text-xs text-blue-400 hover:text-blue-300 underline"
        >
          重試
        </button>
      )}
    </div>
  )
}

/**
 * Minimal variant - just an icon indicator
 */
function MinimalVariant({ state, className }: SSEConnectionStatusProps) {
  return (
    <div
      className={cn('inline-flex items-center', className)}
      title={getStateLabel(state)}
    >
      {getStateIcon(state)}
    </div>
  )
}

/**
 * SSE Connection Status component
 *
 * Displays the current SSE connection state with optional controls
 *
 * @example
 * ```tsx
 * const { state, reconnectAttempt, error, reconnect, stop } = useSSEWithReconnect({
 *   url: '/api/stream'
 * })
 *
 * <SSEConnectionStatus
 *   state={state}
 *   reconnectAttempt={reconnectAttempt}
 *   error={error?.message}
 *   onReconnect={reconnect}
 *   onDisconnect={stop}
 *   variant="full"
 *   showControls
 * />
 * ```
 */
export function SSEConnectionStatus(props: SSEConnectionStatusProps) {
  const { variant = 'full' } = props

  switch (variant) {
    case 'compact':
      return <CompactVariant {...props} />
    case 'minimal':
      return <MinimalVariant {...props} />
    default:
      return <FullVariant {...props} />
  }
}
