import { useBatchJobs } from '../hooks/useBatchJobs'
import { BatchJobCard } from './BatchJobCard'
import { Button } from '@/components/ui/button'
import { useUiStore } from '@/stores/uiStore'

export function BatchMonitor() {
  const { jobs, isLoading, cancelJob, refetch } = useBatchJobs()
  const { addNotification } = useUiStore()

  const handleCancel = async (jobId: string) => {
    try {
      await cancelJob.mutateAsync(jobId)
      addNotification({
        type: 'success',
        title: '任務已取消',
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '取消失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  const handleDownload = (jobId: string) => {
    window.open(`/api/v1/batch/download/${jobId}`, '_blank')
  }

  const activeJobs = jobs.filter(j => j.status === 'running' || j.status === 'pending')
  const completedJobs = jobs.filter(j => j.status === 'completed')
  const failedJobs = jobs.filter(j => j.status === 'failed' || j.status === 'cancelled')

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <p className="text-slate-400">加載中...</p>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-8 space-y-6">
      {/* 頁面標題 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">批次任務監控</h1>
          <p className="text-slate-400 mt-1">
            總計 {jobs.length} 個任務 (進行中: {activeJobs.length}, 完成: {completedJobs.length}, 失敗: {failedJobs.length})
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            刷新
          </Button>
          <Button variant="outline" onClick={() => window.location.href = '/'}>
            返回首頁
          </Button>
        </div>
      </div>

      {/* 進行中的任務 */}
      {activeJobs.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">進行中 ({activeJobs.length})</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {activeJobs.map((job) => (
              <BatchJobCard
                key={job.job_id}
                job={job}
                onCancel={handleCancel}
              />
            ))}
          </div>
        </div>
      )}

      {/* 已完成的任務 */}
      {completedJobs.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">已完成 ({completedJobs.length})</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {completedJobs.map((job) => (
              <BatchJobCard
                key={job.job_id}
                job={job}
                onDownload={handleDownload}
              />
            ))}
          </div>
        </div>
      )}

      {/* 失敗/取消的任務 */}
      {failedJobs.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">失敗/取消 ({failedJobs.length})</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {failedJobs.map((job) => (
              <BatchJobCard key={job.job_id} job={job} />
            ))}
          </div>
        </div>
      )}

      {/* 空狀態 */}
      {jobs.length === 0 && (
        <div className="text-center py-16">
          <p className="text-slate-400 text-lg">還沒有任何批次任務</p>
        </div>
      )}
    </div>
  )
}
