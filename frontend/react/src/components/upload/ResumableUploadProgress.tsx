import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import type { ResumableUploadProgress, UploadState } from '@/hooks/useResumableUpload'

export interface ResumableUploadProgressProps {
  /**
   * Upload state
   */
  state: UploadState

  /**
   * Progress information
   */
  progress: ResumableUploadProgress | null

  /**
   * File name
   */
  fileName?: string

  /**
   * Error message
   */
  error?: string | null

  /**
   * Whether to show control buttons
   */
  showControls?: boolean

  /**
   * Pause handler
   */
  onPause?: () => void

  /**
   * Resume handler
   */
  onResume?: () => void

  /**
   * Cancel handler
   */
  onCancel?: () => void

  /**
   * Additional class name
   */
  className?: string
}

/**
 * Format bytes to human-readable size
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'

  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

/**
 * Format seconds to time string
 */
function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '--:--'

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

/**
 * Get state badge variant
 */
function getStateBadgeVariant(state: UploadState): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (state) {
    case 'uploading':
      return 'default'
    case 'paused':
      return 'secondary'
    case 'completed':
      return 'outline'
    case 'error':
    case 'cancelled':
      return 'destructive'
    default:
      return 'outline'
  }
}

/**
 * Get state label
 */
function getStateLabel(state: UploadState): string {
  switch (state) {
    case 'idle':
      return '準備中'
    case 'uploading':
      return '上傳中'
    case 'paused':
      return '已暫停'
    case 'completed':
      return '已完成'
    case 'error':
      return '上傳失敗'
    case 'cancelled':
      return '已取消'
    default:
      return '未知'
  }
}

/**
 * Resumable upload progress component
 */
export function ResumableUploadProgress({
  state,
  progress,
  fileName,
  error,
  showControls = true,
  onPause,
  onResume,
  onCancel,
  className,
}: ResumableUploadProgressProps) {
  const isUploading = state === 'uploading'
  const isPaused = state === 'paused'
  const isCompleted = state === 'completed'
  const hasError = state === 'error' || state === 'cancelled'

  return (
    <div className={cn('space-y-3 p-4 border border-slate-700 rounded-lg bg-slate-800/50', className)}>
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {fileName && (
            <p className="text-sm font-medium text-slate-200 truncate">{fileName}</p>
          )}
          <div className="flex items-center gap-2 mt-1">
            <Badge variant={getStateBadgeVariant(state)}>{getStateLabel(state)}</Badge>
            {progress && (
              <span className="text-xs text-slate-400">
                {progress.uploadedChunks} / {progress.totalChunks} 塊
              </span>
            )}
          </div>
        </div>

        {/* Control buttons */}
        {showControls && (
          <div className="flex items-center gap-2 ml-4">
            {isUploading && onPause && (
              <Button size="sm" variant="outline" onClick={onPause}>
                暫停
              </Button>
            )}
            {isPaused && onResume && (
              <Button size="sm" variant="outline" onClick={onResume}>
                繼續
              </Button>
            )}
            {(isUploading || isPaused) && onCancel && (
              <Button size="sm" variant="outline" onClick={onCancel}>
                取消
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Progress bar */}
      {progress && !hasError && (
        <div className="space-y-1">
          <Progress value={progress.percentage} className="h-2" />
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{progress.percentage.toFixed(1)}%</span>
            <span>
              {formatBytes(progress.uploadedBytes)} / {formatBytes(progress.totalBytes)}
            </span>
          </div>
        </div>
      )}

      {/* Upload stats */}
      {progress && isUploading && (
        <div className="grid grid-cols-3 gap-4 text-xs">
          <div>
            <p className="text-slate-500">上傳速度</p>
            <p className="text-slate-200 font-medium">{formatBytes(progress.speed)}/s</p>
          </div>
          <div>
            <p className="text-slate-500">剩餘時間</p>
            <p className="text-slate-200 font-medium">
              {formatTime(progress.estimatedTimeRemaining)}
            </p>
          </div>
          <div>
            <p className="text-slate-500">上傳中</p>
            <p className="text-slate-200 font-medium">{progress.uploadingChunks} 塊</p>
          </div>
        </div>
      )}

      {/* Completed stats */}
      {isCompleted && progress && (
        <div className="flex items-center gap-2 text-xs text-green-400">
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
          <span>上傳完成 - {formatBytes(progress.totalBytes)}</span>
        </div>
      )}

      {/* Error message */}
      {error && hasError && (
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

      {/* Chunk details (for debugging) */}
      {progress && import.meta.env.DEV && (
        <details className="text-xs text-slate-500">
          <summary className="cursor-pointer hover:text-slate-400">詳細信息</summary>
          <div className="mt-2 space-y-1 pl-4">
            <p>總塊數: {progress.totalChunks}</p>
            <p>已上傳: {progress.uploadedChunks}</p>
            <p>上傳中: {progress.uploadingChunks}</p>
            <p>失敗: {progress.failedChunks}</p>
            <p>速度: {formatBytes(progress.speed)}/s</p>
          </div>
        </details>
      )}
    </div>
  )
}

/**
 * Compact version for inline display
 */
export function ResumableUploadProgressCompact({
  state,
  progress,
  fileName,
  className,
}: Pick<ResumableUploadProgressProps, 'state' | 'progress' | 'fileName' | 'className'>) {
  return (
    <div className={cn('flex items-center gap-3', className)}>
      {/* File icon */}
      <div className="flex-shrink-0">
        <svg
          className="w-8 h-8 text-slate-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
          />
        </svg>
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        {fileName && (
          <p className="text-sm text-slate-200 truncate">{fileName}</p>
        )}
        <div className="flex items-center gap-2 mt-1">
          <Badge variant={getStateBadgeVariant(state)}>
            {getStateLabel(state)}
          </Badge>
          {progress && (
            <span className="text-xs text-slate-400">
              {progress.percentage.toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      {/* Progress indicator */}
      {progress && (
        <div className="flex-shrink-0 w-12 h-12">
          <svg className="transform -rotate-90" width="48" height="48">
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
              className="text-slate-700"
            />
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 20}`}
              strokeDashoffset={`${2 * Math.PI * 20 * (1 - progress.percentage / 100)}`}
              className="text-primary transition-all duration-300"
            />
          </svg>
        </div>
      )}
    </div>
  )
}
