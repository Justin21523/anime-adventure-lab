import { useState, useCallback } from 'react'
import { useRAGDocuments } from '../hooks/useRAGDocuments'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useUiStore } from '@/stores/uiStore'
import { cn } from '@/lib/utils'

interface DocumentUploaderProps {
  worldId?: string
}

export function DocumentUploader({ worldId }: DocumentUploaderProps) {
  const { uploadDocument } = useRAGDocuments(worldId)
  const { addNotification } = useUiStore()
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

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
      setSelectedFile(files[0])
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFile(files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    try {
      await uploadDocument.mutateAsync({
        file: selectedFile,
        world_id: worldId,
      })

      addNotification({
        type: 'success',
        title: '上傳成功',
        message: `文檔 "${selectedFile.name}" 已成功上傳`,
      })

      setSelectedFile(null)
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
              支持: .txt, .md, .pdf
            </p>
          </div>
        </div>

        {/* 文件選擇器 */}
        <div className="flex gap-2">
          <Input
            type="file"
            onChange={handleFileSelect}
            accept=".txt,.md,.pdf"
            className="flex-1"
          />
        </div>

        {/* 已選文件 */}
        {selectedFile && (
          <div className="p-3 bg-slate-800 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">{selectedFile.name}</div>
                <div className="text-xs text-slate-400">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </div>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSelectedFile(null)}
              >
                移除
              </Button>
            </div>
          </div>
        )}

        {/* 上傳按鈕 */}
        <Button
          onClick={handleUpload}
          disabled={!selectedFile || uploadDocument.isPending}
          className="w-full"
        >
          {uploadDocument.isPending ? '上傳中...' : '上傳文檔'}
        </Button>
      </CardContent>
    </Card>
  )
}
