import { useMemo, useState } from 'react'
import { useRAGDocuments } from '../hooks/useRAGDocuments'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { formatTimestamp } from '@/lib/utils'
import { useUiStore } from '@/stores/uiStore'

interface DocumentListProps {
  worldId?: string
}

export function DocumentList({ worldId }: DocumentListProps) {
  const { documents, isLoading, deleteDocument } = useRAGDocuments(worldId)
  const { addNotification } = useUiStore()
  const [tagFilter, setTagFilter] = useState('')

  const activeTags = useMemo(
    () =>
      tagFilter
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean),
    [tagFilter]
  )

  const allTags = useMemo(() => {
    const set = new Set<string>()
    for (const doc of documents) {
      const tags = doc.metadata?.tags
      if (Array.isArray(tags)) {
        for (const t of tags) {
          const v = String(t || '').trim()
          if (v) set.add(v)
        }
      }
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [documents])

  const filteredDocuments = useMemo(() => {
    if (activeTags.length === 0) return documents
    const needles = activeTags.map((t) => t.toLowerCase())
    return documents.filter((doc) => {
      const tags = doc.metadata?.tags
      if (!Array.isArray(tags) || tags.length === 0) return false
      const hay = tags.map((t: any) => String(t || '').trim().toLowerCase()).filter(Boolean)
      return needles.some((n) => hay.includes(n))
    })
  }, [activeTags, documents])

  const handleDelete = async (docId: string, title: string) => {
    if (!confirm(`確定要刪除文檔 "${title}" 嗎？`)) {
      return
    }

    try {
      await deleteDocument.mutateAsync(docId)
      addNotification({
        type: 'success',
        title: '刪除成功',
        message: `文檔 "${title}" 已刪除`,
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
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
          <CardTitle>文檔列表 ({filteredDocuments.length}/{documents.length})</CardTitle>
          <div className="w-full sm:w-64">
            <Input
              value={tagFilter}
              onChange={(e) => setTagFilter(e.target.value)}
              placeholder="標籤篩選（逗號分隔）"
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {documents.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-slate-400">還沒有上傳任何文檔</p>
          </div>
        ) : (
          <div className="space-y-3">
            {allTags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {allTags.slice(0, 24).map((t) => {
                  const selected = activeTags.some((x) => x.toLowerCase() === t.toLowerCase())
                  return (
                    <button
                      key={t}
                      type="button"
                      onClick={() => {
                        if (selected) {
                          setTagFilter(activeTags.filter((x) => x.toLowerCase() !== t.toLowerCase()).join(', '))
                        } else {
                          setTagFilter([...activeTags, t].join(', '))
                        }
                      }}
                      className={`text-xs px-2 py-1 rounded-full border transition-colors ${
                        selected
                          ? 'border-primary bg-primary/10 text-primary'
                          : 'border-slate-700 bg-slate-900/40 text-slate-300 hover:bg-slate-900/70'
                      }`}
                    >
                      {t}
                    </button>
                  )
                })}
                {allTags.length > 24 && (
                  <span className="text-xs text-slate-500">… +{allTags.length - 24}</span>
                )}
              </div>
            )}

            {filteredDocuments.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-slate-400">沒有符合標籤篩選的文檔</p>
              </div>
            ) : (
              filteredDocuments.map((doc) => {
              const displayName = doc.metadata?.original_filename || doc.title || doc.doc_id
              const docWorldId = doc.metadata?.world_id || 'default'
              const createdAt = doc.created_at
              const tags = Array.isArray(doc.metadata?.tags) ? doc.metadata.tags : []

              return (
                <div
                  key={doc.doc_id}
                  className="p-4 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">📄</span>
                        <div>
                          <div className="font-medium">{displayName}</div>
                          <div className="text-xs text-slate-400 mt-1">
                            <span>{doc.chunks} 個區塊</span>
                            <span className="mx-2">•</span>
                            <span>World: {docWorldId}</span>
                            <span className="mx-2">•</span>
                            <span>{createdAt ? formatTimestamp(createdAt) : '-'}</span>
                          </div>
                        </div>
                      </div>

                      {tags.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {tags.slice(0, 12).map((t: any) => (
                            <Badge key={String(t)} variant="secondary" className="text-xs">
                              {String(t)}
                            </Badge>
                          ))}
                          {tags.length > 12 && (
                            <span className="text-xs text-slate-500">… +{tags.length - 12}</span>
                          )}
                        </div>
                      )}
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(doc.doc_id, displayName)}
                      disabled={deleteDocument.isPending}
                      className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                    >
                      刪除
                    </Button>
                  </div>
                </div>
              )
              })
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
