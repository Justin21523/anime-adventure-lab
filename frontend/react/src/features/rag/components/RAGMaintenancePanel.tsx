import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useUiStore } from '@/stores/uiStore'
import { useRAGMaintenance } from '../hooks/useRAGMaintenance'
import { useJob } from '@/features/jobs/hooks/useJob'
import { JobProgressCard } from '@/features/jobs/components/JobProgressCard'
import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { CACHE_KEYS } from '@/config/query.config'

interface RAGMaintenancePanelProps {
  worldId: string
}

export function RAGMaintenancePanel({ worldId }: RAGMaintenancePanelProps) {
  const { addNotification } = useUiStore()
  const { rebuildIndex, clearWorld } = useRAGMaintenance(worldId)
  const queryClient = useQueryClient()
  const [rebuildJobId, setRebuildJobId] = useState<string | null>(null)
  const rebuildJob = useJob(rebuildJobId, { enabled: Boolean(rebuildJobId), refetchIntervalMs: 2000 })

  const rebuildInProgress = useMemo(() => {
    if (!rebuildJobId) return false
    const status = String(rebuildJob.job?.status || '').toLowerCase()
    return status !== 'completed' && status !== 'failed' && status !== 'cancelled'
  }, [rebuildJob.job?.status, rebuildJobId])

  const handleRebuild = async () => {
    try {
      const res = await rebuildIndex.mutateAsync()
      const jobId = res.job_id ? String(res.job_id) : ''
      if (jobId) {
        setRebuildJobId(jobId)
        addNotification({ type: 'success', title: '已建立索引重建任務', message: `job_id: ${jobId}` })
        return
      }
      addNotification({ type: res.success ? 'success' : 'error', title: '索引重建完成', message: res.message || '—' })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '索引重建失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const handleClearWorld = async () => {
    if (!confirm(`確定要清空 world_id=${worldId} 的所有知識庫文件嗎？此操作會移除所有 chunks 並重建索引。`)) {
      return
    }
    try {
      const res = await clearWorld.mutateAsync()
      addNotification({
        type: res.success ? 'success' : 'error',
        title: '已清空世界知識庫',
        message: res.message || `移除 ${res.documents_removed} 文件 / ${res.chunks_removed} chunks`,
      })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '清空失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>維護</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-slate-500">
          提示：清空/重建會影響檢索；大型知識庫可能需要一些時間。
        </p>
        <div className="flex flex-col sm:flex-row gap-2">
          <Button
            variant="outline"
            onClick={() => void handleRebuild()}
            disabled={rebuildIndex.isPending || clearWorld.isPending || rebuildInProgress}
          >
            {rebuildIndex.isPending ? '提交中...' : rebuildInProgress ? '重建中...' : '索引重建'}
          </Button>
          <Button
            variant="outline"
            onClick={() => void handleClearWorld()}
            disabled={clearWorld.isPending || rebuildIndex.isPending || rebuildInProgress}
            className="text-red-300 border-red-700/50 hover:bg-red-900/20"
          >
            {clearWorld.isPending ? '清空中...' : '清空此世界'}
          </Button>
        </div>

        {rebuildJobId && (
          <JobProgressCard
            title="RAG 索引重建"
            jobId={rebuildJobId}
            job={rebuildJob.job}
            isLoading={rebuildJob.isLoading}
            error={rebuildJob.error}
            cancelling={rebuildJob.cancelJob.isPending}
            onCancel={() => void rebuildJob.cancelJob.mutate()}
            onCompleted={() => {
              queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
              queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(worldId) })
              addNotification({ type: 'success', title: '索引重建完成' })
            }}
            onFailed={(job) => {
              addNotification({
                type: 'error',
                title: '索引重建失敗',
                message: String((job as any)?.error || 'unknown error'),
              })
            }}
            onCancelled={() => addNotification({ type: 'error', title: '索引重建已取消' })}
          />
        )}
      </CardContent>
    </Card>
  )
}
