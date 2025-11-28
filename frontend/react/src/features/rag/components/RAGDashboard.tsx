import { useRAGStats } from '../hooks/useRAGStats'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

interface RAGDashboardProps {
  worldId?: string
}

export function RAGDashboard({ worldId }: RAGDashboardProps) {
  const { data: stats, isLoading } = useRAGStats(worldId)

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">加載統計中...</p>
        </CardContent>
      </Card>
    )
  }

  if (!stats) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">無統計數據</p>
        </CardContent>
      </Card>
    )
  }

  const statCards = [
    {
      label: '文檔總數',
      value: stats.total_documents,
      icon: '📄',
      color: 'text-blue-400',
    },
    {
      label: '文本區塊',
      value: stats.total_chunks,
      icon: '📝',
      color: 'text-green-400',
    },
    {
      label: '向量總數',
      value: stats.total_vectors,
      icon: '🔢',
      color: 'text-purple-400',
    },
    {
      label: '索引大小',
      value: `${stats.index_size_mb.toFixed(2)} MB`,
      icon: '💾',
      color: 'text-orange-400',
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {statCards.map((stat) => (
        <Card key={stat.label}>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">{stat.label}</span>
              <span className="text-2xl">{stat.icon}</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${stat.color}`}>
              {stat.value}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
