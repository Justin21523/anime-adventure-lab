import { useState } from 'react'
import { useRAGSearch } from '../hooks/useRAGSearch'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface SearchInterfaceProps {
  worldId?: string
}

export function SearchInterface({ worldId }: SearchInterfaceProps) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const searchMutation = useRAGSearch()

  const handleSearch = async () => {
    if (!query.trim()) return

    await searchMutation.mutateAsync({
      query,
      world_id: worldId,
      top_k: topK,
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSearch()
    }
  }

  return (
    <div className="space-y-4">
      {/* 搜索輸入 */}
      <Card>
        <CardHeader>
          <CardTitle>搜索文檔</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="輸入搜索查詢..."
              className="flex-1"
            />
            <Input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
              className="w-20"
              min={1}
              max={20}
              title="返回結果數量"
            />
            <Button
              onClick={handleSearch}
              disabled={!query.trim() || searchMutation.isPending}
            >
              {searchMutation.isPending ? '搜索中...' : '搜索'}
            </Button>
          </div>

          <p className="text-xs text-slate-500">
            提示：按 Enter 搜索，調整數字控制返回結果數量 (1-20)
          </p>
        </CardContent>
      </Card>

      {/* 搜索結果 */}
      {searchMutation.data && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>搜索結果</CardTitle>
              <div className="text-sm text-slate-400">
                找到 {searchMutation.data.total_found} 個結果
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {searchMutation.data.results.length === 0 ? (
              <p className="text-center text-slate-400 py-8">沒有找到相關結果</p>
            ) : (
              <div className="space-y-3">
                {searchMutation.data.results.map((result, idx) => (
                  <div
                    key={`${result.doc_id}-${result.rank}`}
                    className="p-4 bg-slate-800/50 rounded-lg"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-semibold">
                        {result.rank ?? idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs px-2 py-1 bg-blue-900/30 text-blue-300 rounded">
                            分數: {result.score.toFixed(3)}
                          </span>
                          <span className="text-xs text-slate-500">
                            Doc: {result.doc_id.slice(0, 8)}...
                          </span>
                        </div>
                        <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                          {result.content}
                        </p>
                        {result.metadata && Object.keys(result.metadata).length > 0 && (
                          <div className="mt-2 text-xs text-slate-500">
                            {JSON.stringify(result.metadata)}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 錯誤提示 */}
      {searchMutation.error && (
        <Card className="border-red-500/50">
          <CardContent className="py-4">
            <p className="text-red-400 text-sm">
              搜索失敗：{searchMutation.error instanceof Error ? searchMutation.error.message : '未知錯誤'}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
