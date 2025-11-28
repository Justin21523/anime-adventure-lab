import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { formatTimestamp } from '@/lib/utils'
import type { BatchJob } from '../types/batch.types'

interface BatchJobCardProps {
  job: BatchJob
  onCancel?: (jobId: string) => void
  onDownload?: (jobId: string) => void
}

const statusColors = {
  pending: 'text-slate-400 bg-slate-700/30',
  running: 'text-blue-400 bg-blue-900/30',
  completed: 'text-green-400 bg-green-900/30',
  failed: 'text-red-400 bg-red-900/30',
  cancelled: 'text-orange-400 bg-orange-900/30',
}

const statusLabels = {
  pending: '等待中',
  running: '執行中',
  completed: '已完成',
  failed: '失敗',
  cancelled: '已取消',
}

const jobTypeLabels = {
  t2i: '文生圖',
  caption: '圖像標註',
  embedding: '向量生成',
  training: '模型訓練',
  rag_indexing: 'RAG 索引',
}

export function BatchJobCard({ job, onCancel, onDownload }: BatchJobCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">
            {jobTypeLabels[job.job_type] || job.job_type}
          </CardTitle>
          <span
            className={`text-xs px-2 py-1 rounded ${statusColors[job.status]}`}
          >
            {statusLabels[job.status]}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* 進度條 */}
        {job.progress && job.status === 'running' && (
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-400">進度</span>
              <span className="text-slate-300">
                {job.progress.current} / {job.progress.total} ({job.progress.percentage.toFixed(1)}%)
              </span>
            </div>
            <Progress value={job.progress.percentage} max={100} />
          </div>
        )}

        {/* 時間信息 */}
        <div className="text-xs text-slate-400 space-y-1">
          <div>創建時間: {formatTimestamp(job.created_at)}</div>
          {job.started_at && (
            <div>開始時間: {formatTimestamp(job.started_at)}</div>
          )}
          {job.completed_at && (
            <div>完成時間: {formatTimestamp(job.completed_at)}</div>
          )}
        </div>

        {/* 錯誤消息 */}
        {job.error_message && (
          <div className="p-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-400">
            {job.error_message}
          </div>
        )}

        {/* 操作按鈕 */}
        <div className="flex gap-2">
          {job.status === 'running' && onCancel && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onCancel(job.job_id)}
              className="text-orange-400"
            >
              取消
            </Button>
          )}

          {job.status === 'completed' && job.result_path && onDownload && (
            <Button
              size="sm"
              onClick={() => onDownload(job.job_id)}
            >
              下載結果
            </Button>
          )}
        </div>

        {/* Job ID */}
        <div className="text-xs text-slate-600">
          ID: {job.job_id.slice(0, 16)}...
        </div>
      </CardContent>
    </Card>
  )
}
