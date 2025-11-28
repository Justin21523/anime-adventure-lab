import { useRAGDocuments } from '../hooks/useRAGDocuments'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { formatTimestamp } from '@/lib/utils'
import { useUiStore } from '@/stores/uiStore'

interface DocumentListProps {
  worldId?: string
}

export function DocumentList({ worldId }: DocumentListProps) {
  const { documents, isLoading, deleteDocument } = useRAGDocuments(worldId)
  const { addNotification } = useUiStore()

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`確定要刪除文檔 "${filename}" 嗎？`)) {
      return
    }

    try {
      await deleteDocument.mutateAsync(docId)
      addNotification({
        type: 'success',
        title: '刪除成功',
        message: `文檔 "${filename}" 已刪除`,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '刪除失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">加載中...</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>文檔列表 ({documents.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {documents.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-slate-400">還沒有上傳任何文檔</p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="p-4 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">📄</span>
                      <div>
                        <div className="font-medium">{doc.filename}</div>
                        <div className="text-xs text-slate-400 mt-1">
                          <span>{doc.chunk_count} 個區塊</span>
                          <span className="mx-2">•</span>
                          <span>World: {doc.world_id}</span>
                          <span className="mx-2">•</span>
                          <span>{formatTimestamp(doc.created_at)}</span>
                        </div>
                      </div>
                    </div>

                    {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                      <div className="mt-2 text-xs text-slate-500">
                        {JSON.stringify(doc.metadata)}
                      </div>
                    )}
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(doc.doc_id, doc.filename)}
                    disabled={deleteDocument.isPending}
                    className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                  >
                    刪除
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
