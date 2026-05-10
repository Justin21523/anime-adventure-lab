import React from 'react'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { useUiStore } from '@/stores/uiStore'
import { useBatchJobs } from '@/features/batch/hooks/useBatchJobs'
import { BatchJobCard } from '@/features/batch/components/BatchJobCard'
import { 
  Terminal, 
  Globe, 
  Image as ImageIcon, 
  LayoutGrid, 
  Loader2 
} from 'lucide-react'

// Static imports for reliability
import { WorldStudioPanel } from '@/features/worlds/components/WorldStudioPanel'
import { T2IGenerator } from '@/features/t2i/components/T2IGenerator'
import { ImageGallery } from '@/features/t2i/components/ImageGallery'
import { TaskExecutor } from '@/features/agent/components/TaskExecutor'
import { ToolBrowser } from '@/features/agent/components/ToolBrowser'

function WorkbenchTabTrigger({ value, icon, label }: { value: string, icon: React.ReactNode, label: string }) {
  return (
    <TabsTrigger 
      value={value} 
      className="flex items-center gap-2 px-6 py-2.5 rounded-xl data-[state=active]:bg-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg transition-all text-slate-400 hover:text-slate-100"
    >
      {icon}
      <span className="font-bold tracking-wide">{label}</span>
    </TabsTrigger>
  )
}

interface StoryWorkbenchDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  worldId: string
  tab?: string
  onTabChange?: (value: string) => void
}

function BatchJobsPanel() {
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

  const activeJobs = jobs.filter((j) => j.status === 'running' || j.status === 'pending')
  const completedJobs = jobs.filter((j) => j.status === 'completed')
  const failedJobs = jobs.filter((j) => j.status === 'failed' || j.status === 'cancelled')

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <p className="text-slate-400">加載中...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">批次任務</h3>
          <p className="text-sm text-slate-400">
            總計 {jobs.length}（進行中: {activeJobs.length}，完成: {completedJobs.length}，失敗/取消: {failedJobs.length}）
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          刷新
        </Button>
      </div>

      {activeJobs.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-slate-200">進行中</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {activeJobs.map((job) => (
              <BatchJobCard key={job.job_id} job={job} onCancel={handleCancel} />
            ))}
          </div>
        </div>
      )}

      {completedJobs.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-slate-200">已完成</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {completedJobs.map((job) => (
              <BatchJobCard key={job.job_id} job={job} onDownload={handleDownload} />
            ))}
          </div>
        </div>
      )}

      {failedJobs.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-slate-200">失敗 / 取消</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {failedJobs.map((job) => (
              <BatchJobCard key={job.job_id} job={job} />
            ))}
          </div>
        </div>
      )}

      {jobs.length === 0 && (
        <div className="text-center py-16">
          <p className="text-slate-400 text-lg">還沒有任何批次任務</p>
        </div>
      )}
    </div>
  )
}

export function StoryWorkbenchDialog({
  open,
  onOpenChange,
  sessionId,
  worldId,
  tab,
  onTabChange,
}: StoryWorkbenchDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[90vw] w-[1200px] h-[90vh] p-0 overflow-hidden bg-slate-950 border-slate-800 shadow-2xl flex flex-col">
        {/* Header - Fixed */}
        <div className="px-8 py-6 border-b border-white/5 bg-indigo-950/10 flex-shrink-0">
          <DialogHeader>
            <div className="flex items-center gap-3">
               <Terminal className="w-7 h-7 text-indigo-400" />
               <DialogTitle className="text-2xl font-bold text-white">故事工作台</DialogTitle>
            </div>
            <DialogDescription className="text-slate-400 mt-1">
              Session: {sessionId.slice(0, 8)}... / World: {worldId}
            </DialogDescription>
          </DialogHeader>
        </div>

        {/* Content Area - Scrollable */}
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden text-slate-100">
          <Tabs defaultValue="world" value={tab} onValueChange={onTabChange} className="h-full flex flex-col">
            <div className="px-8 py-2 border-b border-white/5 bg-slate-900/20">
              <TabsList className="bg-transparent border-none gap-2 h-12">
                <WorkbenchTabTrigger value="world" icon={<Globe className="w-4 h-4" />} label="世界工作室" />
                <WorkbenchTabTrigger value="t2i" icon={<ImageIcon className="w-4 h-4" />} label="場景視覺" />
                <WorkbenchTabTrigger value="batch" icon={<LayoutGrid className="w-4 h-4" />} label="批次任務" />
                <WorkbenchTabTrigger value="agent" icon={<Terminal className="w-4 h-4" />} label="Agent 控制" />
              </TabsList>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <div className="p-8">
                <TabsContent value="world" className="mt-0 outline-none">
                   <WorldStudioPanel worldId={worldId} sessionId={sessionId} />
                </TabsContent>

                <TabsContent value="t2i" className="mt-0 outline-none">
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
                    <T2IGenerator sessionId={sessionId} />
                    <ImageGallery sessionId={sessionId} />
                  </div>
                </TabsContent>

                <TabsContent value="batch" className="mt-0 outline-none">
                  <BatchJobsPanel />
                </TabsContent>

                <TabsContent value="agent" className="mt-0 outline-none">
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
                    <TaskExecutor />
                    <ToolBrowser />
                  </div>
                </TabsContent>
              </div>
            </div>
          </Tabs>
        </div>
      </DialogContent>
      {/* ... styles ... */}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(79, 70, 229, 0.4);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(79, 70, 229, 0.6);
        }
      `}</style>
    </Dialog>
  )
}
