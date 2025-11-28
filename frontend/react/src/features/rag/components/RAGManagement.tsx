import { useState } from 'react'
import { RAGDashboard } from './RAGDashboard'
import { DocumentUploader } from './DocumentUploader'
import { DocumentList } from './DocumentList'
import { SearchInterface } from './SearchInterface'
import { Button } from '@/components/ui/button'

export function RAGManagement() {
  const [worldId] = useState<string>('default')

  return (
    <div className="container mx-auto p-8 space-y-6">
      {/* 頁面標題 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">RAG 文檔管理</h1>
          <p className="text-slate-400 mt-1">管理知識庫文檔和向量搜索</p>
        </div>
        <Button variant="outline" onClick={() => window.location.href = '/'}>
          返回首頁
        </Button>
      </div>

      {/* 統計儀表板 */}
      <RAGDashboard worldId={worldId} />

      {/* 搜索界面 */}
      <SearchInterface worldId={worldId} />

      {/* 兩欄布局：上傳 + 列表 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <DocumentUploader worldId={worldId} />
        </div>
        <div className="lg:col-span-2">
          <DocumentList worldId={worldId} />
        </div>
      </div>
    </div>
  )
}
