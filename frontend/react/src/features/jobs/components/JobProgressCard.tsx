import { useEffect, useMemo, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import type { JobRecord } from '../types/job.types'

function formatDurationSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return ''
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds - mins * 60)
  return `${mins}m ${secs.toString().padStart(2, '0')}s`
}

function parseTimestampMs(value?: string | null): number | null {
  const s = String(value || '').trim()
  if (!s) return null
  const ms = Date.parse(s)
  return Number.isFinite(ms) ? ms : null
}

function isTerminalStatus(status: string): boolean {
  const s = String(status || '').toLowerCase()
  return s === 'completed' || s === 'failed' || s === 'cancelled'
}

function statusBadgeVariant(status: string): 'default' | 'outline' | 'secondary' | 'destructive' {
  const s = String(status || '').toLowerCase()
  if (s === 'completed') return 'default'
  if (s === 'failed') return 'destructive'
  if (s === 'cancelled') return 'secondary'
  return 'outline'
}

export interface JobProgressCardProps {
  title?: string
  jobId: string
  job?: JobRecord | null
  isLoading?: boolean
  error?: unknown
  onCancel?: () => void | Promise<void>
  cancelling?: boolean
  onCompleted?: (job: JobRecord) => void
  onFailed?: (job: JobRecord) => void
  onCancelled?: (job: JobRecord) => void
  hideWhenCompleted?: boolean
}

export function JobProgressCard({
  title = 'Job',
  jobId,
  job,
  isLoading,
  error,
  onCancel,
  cancelling,
  onCompleted,
  onFailed,
  onCancelled,
  hideWhenCompleted,
}: JobProgressCardProps) {
  const status = String(job?.status || (isLoading ? 'loading' : 'unknown'))
  const progress = useMemo(() => {
    const v = Number(job?.progress ?? 0)
    return Number.isFinite(v) ? Math.max(0, Math.min(100, v)) : 0
  }, [job?.progress])

  const stage = String((job as any)?.stage || '').trim()
  const stageMessage = String((job as any)?.stage_message || '').trim()

  const elapsedText = useMemo(() => {
    const durationSeconds = Number((job as any)?.duration_seconds)
    if (Number.isFinite(durationSeconds) && durationSeconds > 0) return formatDurationSeconds(durationSeconds)

    const startMs = parseTimestampMs(job?.started_at) ?? parseTimestampMs(job?.created_at)
    if (!startMs) return ''
    const nowMs = Date.now()
    const elapsedSeconds = Math.max(0, (nowMs - startMs) / 1000)
    return formatDurationSeconds(elapsedSeconds)
  }, [job])

  const terminal = isTerminalStatus(status)
  const shouldHide = Boolean(hideWhenCompleted) && String(status).toLowerCase() === 'completed'

  const lastTerminalStatusRef = useRef<string | null>(null)
  useEffect(() => {
    if (!job) return
    const s = String(job.status || '')
    if (!isTerminalStatus(s)) return
    if (lastTerminalStatusRef.current === s) return
    lastTerminalStatusRef.current = s

    const lowered = s.toLowerCase()
    if (lowered === 'completed') onCompleted?.(job)
    else if (lowered === 'failed') onFailed?.(job)
    else if (lowered === 'cancelled') onCancelled?.(job)
  }, [job, onCancelled, onCompleted, onFailed])

  if (!jobId || shouldHide) return null

  const jobError = (job as any)?.error
  const errorText =
    jobError ? String(jobError) : error instanceof Error ? error.message : error ? String(error) : null

  return (
    <Card className="bg-slate-900/40 border-slate-700">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm">{title}</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="text-[10px] font-mono">
              {jobId.slice(0, 10)}…
            </Badge>
            <Badge variant={statusBadgeVariant(status)} className="text-xs">
              {status}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {(stageMessage || stage || elapsedText) && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="text-slate-300">
              {stageMessage || (stage ? `stage: ${stage}` : '')}
            </div>
            {elapsedText && <div className="text-slate-500">{terminal ? `耗時 ${elapsedText}` : `已執行 ${elapsedText}`}</div>}
          </div>
        )}

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-500">progress</span>
            <span className="text-slate-300">{progress.toFixed(1)}%</span>
          </div>
          <Progress value={progress} max={100} />
        </div>

        {errorText && (
          <div className="text-xs text-red-300 rounded-md border border-red-900/40 bg-red-950/20 p-3 whitespace-pre-wrap break-words">
            {errorText}
          </div>
        )}

        {onCancel && !terminal && (
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              className="h-7 px-2 text-xs"
              onClick={() => void onCancel()}
              disabled={Boolean(cancelling)}
            >
              {cancelling ? '取消中...' : '取消任務'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
