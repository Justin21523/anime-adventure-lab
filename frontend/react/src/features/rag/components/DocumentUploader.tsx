import { useCallback, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useRAGDocuments } from '../hooks/useRAGDocuments'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useUiStore } from '@/stores/uiStore'
import { cn } from '@/lib/utils'
import { CACHE_KEYS } from '@/config/query.config'
import { useJob } from '@/features/jobs/hooks/useJob'
import { JobProgressCard } from '@/features/jobs/components/JobProgressCard'

interface DocumentUploaderProps {
  worldId?: string
}

export function DocumentUploader({ worldId }: DocumentUploaderProps) {
  const { uploadDocument, uploadDocumentsBatch, uploadProgress } = useRAGDocuments(worldId)
  const { addNotification } = useUiStore()
  const queryClient = useQueryClient()
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [tags, setTags] = useState('')
  const [ingestJobId, setIngestJobId] = useState<string | null>(null)

  const ingestJob = useJob(ingestJobId, { enabled: Boolean(ingestJobId), refetchIntervalMs: 2000 })
  const ingestInProgress = useMemo(() => {
    if (!ingestJobId) return false
    const status = String(ingestJob.job?.status || '').toLowerCase()
    return status !== 'completed' && status !== 'failed' && status !== 'cancelled'
  }, [ingestJob.job?.status, ingestJobId])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      setSelectedFiles(files)
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFiles(Array.from(files))
    }
  }

  const handleRemoveFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  const handleUpload = async () => {
    if (!selectedFiles.length) return

    try {
      const resp =
        selectedFiles.length > 1
          ? await uploadDocumentsBatch.mutateAsync({
              files: selectedFiles,
              world_id: worldId,
              tags,
            })
          : selectedFiles[0].name.toLowerCase().endsWith('.zip')
            ? await uploadDocumentsBatch.mutateAsync({
                files: selectedFiles,
                world_id: worldId,
                tags,
              })
            : await uploadDocument.mutateAsync({
                file: selectedFiles[0],
                world_id: worldId,
                tags,
              })

      const jobId = String((resp as any)?.job_id || '').trim()
      addNotification({
        type: 'success',
        title: '已建立上傳任務',
        message: jobId
          ? `job_id: ${jobId.slice(0, 10)}…（${selectedFiles.length} 個檔案）`
          : '（伺服器未回傳 job_id）',
      })

      if (jobId) setIngestJobId(jobId)

      setSelectedFiles([])
      setTags('')
    } catch (error) {
      addNotification({
        type: 'error',
        title: '上傳失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>上傳文檔</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {ingestJobId && (
          <JobProgressCard
            title="RAG 文件處理"
            jobId={ingestJobId}
            job={ingestJob.job}
            isLoading={ingestJob.isLoading}
            error={ingestJob.error}
            cancelling={ingestJob.cancelJob.isPending}
            onCancel={() => void ingestJob.cancelJob.mutate()}
            onCompleted={() => {
              queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(worldId) })
              queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(worldId) })
              addNotification({ type: 'success', title: '已寫入知識庫', message: `world_id=${worldId || 'default'}` })
              setIngestJobId(null)
            }}
            onFailed={(job) => {
              addNotification({
                type: 'error',
                title: '知識庫寫入失敗',
                message: String((job as any)?.error || 'unknown error'),
              })
              setIngestJobId(null)
            }}
            onCancelled={() => {
              addNotification({ type: 'error', title: '任務已取消' })
              setIngestJobId(null)
            }}
            hideWhenCompleted={true}
          />
        )}

        {/* 拖放區域 */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
            isDragging
              ? 'border-primary bg-primary/5'
              : 'border-slate-600 hover:border-slate-500'
          )}
        >
          <div className="space-y-2">
            <div className="text-4xl">📄</div>
            <p className="text-sm text-slate-400">
              {isDragging
                ? '放開以上傳文件'
                : '拖放文件到此處，或點擊選擇文件'}
            </p>
            <p className="text-xs text-slate-500">
              支持: .txt, .md, .pdf, .docx, .csv, .json, .zip（zip 會自動解壓後批次匯入）
            </p>
          </div>
        </div>

        {/* 文件選擇器 */}
        <div className="flex gap-2">
          <Input
            type="file"
            onChange={handleFileSelect}
            multiple
            accept=".txt,.md,.markdown,.pdf,.docx,.csv,.tsv,.json,.xml,.html,.htm,.yml,.yaml,.zip"
            className="flex-1"
          />
        </div>

        {/* Tags */}
        <div className="space-y-1">
          <div className="text-xs text-slate-400">Tags（可選，逗號分隔）</div>
          <Input
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="例如：世界觀, 角色設定, 地理"
          />
        </div>

        {/* 已選文件 */}
        {selectedFiles.length ? (
          <div className="space-y-2">
            <div className="text-xs text-slate-400">
              已選 {selectedFiles.length} 個檔案（共{' '}
              {(selectedFiles.reduce((sum, f) => sum + (f.size || 0), 0) / 1024 / 1024).toFixed(2)} MB）
            </div>
            <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
              {selectedFiles.map((f, idx) => (
                <div key={`${f.name}-${idx}`} className="p-3 bg-slate-800 rounded-lg">
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-sm font-medium truncate">{f.name}</div>
                      <div className="text-xs text-slate-400">
                        {(f.size / 1024).toFixed(2)} KB
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleRemoveFile(idx)}
                      disabled={uploadDocument.isPending || uploadDocumentsBatch.isPending}
                    >
                      移除
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {/* 上傳進度條 */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">上傳中...</span>
              <span className="text-primary font-medium">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
              <div
                className="bg-primary h-full transition-all duration-300 ease-out"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* 上傳按鈕 */}
        <Button
          onClick={handleUpload}
          disabled={
            !selectedFiles.length ||
            uploadDocument.isPending ||
            uploadDocumentsBatch.isPending ||
            ingestInProgress
          }
          className="w-full"
        >
          {uploadDocument.isPending || uploadDocumentsBatch.isPending
            ? '上傳中...'
            : ingestInProgress
              ? '處理中...'
              : '上傳文檔'}
        </Button>
      </CardContent>
    </Card>
  )
}
