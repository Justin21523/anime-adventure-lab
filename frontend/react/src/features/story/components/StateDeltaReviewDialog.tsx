import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { ReviewQueueItem } from '../types/reviewQueue.types'

type StateDeltaQueueItem = Extract<ReviewQueueItem, { kind: 'state_delta' }>

interface StateDeltaReviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  item: StateDeltaQueueItem | null
  onMarkReviewed: (itemId: string) => void
  onRemove: (itemId: string) => void
  busy?: boolean
}

export function StateDeltaReviewDialog({
  open,
  onOpenChange,
  item,
  onMarkReviewed,
  onRemove,
  busy,
}: StateDeltaReviewDialogProps) {
  const title = item ? `重大變更審核（Turn ${item.turn + 1}）` : '重大變更審核'
  const diff = (item?.artifacts as any)?.diff || null

  const flagsCount = Array.isArray(diff?.flags) ? diff.flags.length : 0
  const statsCount = Array.isArray(diff?.stats) ? diff.stats.length : 0
  const invCount = (diff?.inventory?.added?.length || 0) + (diff?.inventory?.removed?.length || 0)
  const relCount = Array.isArray(diff?.relationships) ? diff.relationships.length : 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl w-[95vw] h-[90vh] p-0 overflow-hidden">
        <div className="flex flex-col h-full">
          <div className="p-6 border-b border-slate-700">
            <DialogHeader>
              <DialogTitle>{title}</DialogTitle>
              <DialogDescription>
                {item ? (
                  <>
                    {new Date(item.created_at).toLocaleString()} • 回合 artifacts/diff 快照（可用於回寫或除錯）
                  </>
                ) : (
                  '—'
                )}
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {!item ? (
              <Card className="bg-slate-900/40 border-slate-700">
                <CardContent className="py-10 text-center text-slate-400">未選擇項目</CardContent>
              </Card>
            ) : (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={item.status === 'applied' ? 'default' : 'outline'} className="text-xs">
                      {item.status === 'applied' ? '已確認' : '待審核'}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      flags {flagsCount}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      stats {statsCount}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      inv {invCount}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      rel {relCount}
                    </Badge>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onMarkReviewed(item.id)}
                      disabled={busy || item.status === 'applied'}
                    >
                      標記已確認
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onRemove(item.id)}
                      disabled={busy}
                    >
                      移除
                    </Button>
                  </div>
                </div>

                <Card className="bg-slate-900/40 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Artifacts JSON</CardTitle>
                    <p className="text-xs text-slate-500 mt-1">目前先以 JSON 形式呈現；之後可擴增成可套用的 patch。</p>
                  </CardHeader>
                  <CardContent>
                    <Textarea
                      value={JSON.stringify(item.artifacts || {}, null, 2)}
                      readOnly
                      rows={18}
                      className="font-mono text-xs"
                    />
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
