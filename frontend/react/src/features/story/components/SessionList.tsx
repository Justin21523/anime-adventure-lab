import { useStorySessions } from '../hooks/useStorySessions'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { formatTimestamp } from '@/lib/utils'

interface SessionListProps {
  onSelectSession: (sessionId: string) => void
  onNewSession: () => void
}

export function SessionList({ onSelectSession, onNewSession }: SessionListProps) {
  const { data, isLoading, error } = useStorySessions()

  if (isLoading) {
    return <div className="text-center py-8">加載中...</div>
  }

  if (error) {
    return (
      <div className="text-center py-8 text-red-400">
        加載失敗: {error instanceof Error ? error.message : '未知錯誤'}
      </div>
    )
  }

  const sessions = data?.sessions || []

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">我的冒險</h2>
        <Button onClick={onNewSession}>開始新冒險</Button>
      </div>

      {sessions.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-slate-400 mb-4">還沒有任何冒險記錄</p>
            <Button onClick={onNewSession}>開始第一次冒險</Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {sessions.map((session) => (
            <Card
              key={session.session_id}
              className="hover:border-primary transition-colors cursor-pointer"
              onClick={() => onSelectSession(session.session_id)}
            >
              <CardHeader>
                <CardTitle className="text-lg">{session.player_name}</CardTitle>
                <p className="text-sm text-slate-400">
                  角色扮演: {session.persona_id}
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">回合數</span>
                    <span className="text-slate-200">{session.turn_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">最後遊玩</span>
                    <span className="text-slate-200 text-xs">
                      {formatTimestamp(session.last_played)}
                    </span>
                  </div>
                </div>
                <Button className="w-full mt-4" size="sm">
                  繼續冒險
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
