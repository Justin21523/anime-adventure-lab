import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Textarea } from '@/components/ui/textarea'
import type { ReviewQueueItem, WritebackApplySelection } from '../types/reviewQueue.types'

interface WritebackReviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  item: Extract<ReviewQueueItem, { kind: 'world_writeback' }> | null
  onUpdateSelection: (itemId: string, selection: WritebackApplySelection) => void
  onApplyToWorld: (item: Extract<ReviewQueueItem, { kind: 'world_writeback' }>, syncAfter: boolean) => void
  onAddRagNote: (item: Extract<ReviewQueueItem, { kind: 'world_writeback' }>) => void
  onRemove: (itemId: string) => void
  busy?: boolean
}

export function WritebackReviewDialog({
  open,
  onOpenChange,
  item,
  onUpdateSelection,
  onApplyToWorld,
  onAddRagNote,
  onRemove,
  busy,
}: WritebackReviewDialogProps) {
  const title = item ? `世界回寫審核（${item.id.slice(0, 8)}…）` : '世界回寫審核'
  const res = item?.response || null
  const selection = item?.selection || { world_flags: true, characters: true, rag_note: true }

  const flagPatch = (res?.patch as any)?.world_flags as Record<string, any> | undefined
  const flags = flagPatch ? Object.keys(flagPatch).sort() : []
  const charPatch = (res?.patch as any)?.characters as Array<Record<string, any>> | undefined
  const chars = Array.isArray(charPatch) ? charPatch : []

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl w-[95vw] h-[90vh] p-0 overflow-hidden">
        <div className="flex flex-col h-full">
          <div className="p-6 border-b border-slate-700">
            <DialogHeader>
              <DialogTitle>{title}</DialogTitle>
              <DialogDescription>
                {res ? (
                  <>
                    world_id={res.world_id} • session_id={res.session_id} • {new Date(item!.created_at).toLocaleString()}
                  </>
                ) : (
                  '—'
                )}
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {!item || !res ? (
              <Card className="bg-slate-900/40 border-slate-700">
                <CardContent className="py-10 text-center text-slate-400">未選擇項目</CardContent>
              </Card>
            ) : (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={item.status === 'applied' ? 'default' : 'outline'} className="text-xs">
                      {item.status === 'applied' ? '已套用' : '待審核'}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      flags +{flags.length}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      characters +{chars.length}
                    </Badge>
                    {res.rag_note && (
                      <Badge variant="outline" className="text-xs">
                        RAG note
                      </Badge>
                    )}
                    {res.errors?.length ? (
                      <Badge variant="destructive" className="text-xs">
                        errors {res.errors.length}
                      </Badge>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onRemove(item.id)}
                      disabled={busy}
                    >
                      移除
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onApplyToWorld(item, false)}
                      disabled={busy}
                    >
                      保存到世界
                    </Button>
                    <Button
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onApplyToWorld(item, true)}
                      disabled={busy}
                    >
                      保存 + 同步到本故事
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">套用內容</CardTitle>
                        <Badge variant="outline" className="text-xs">
                          patch
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        只會在你確認後，寫回 worldpack（可選擇保存後同步到本故事）
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.world_flags}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, world_flags: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">套用 flags（世界狀態）</span>
                          <Badge variant="outline" className="text-xs">
                            +{flags.length}
                          </Badge>
                        </label>
                        {selection.world_flags && flags.length > 0 && (
                          <div className="text-xs text-slate-300 rounded-md border border-slate-700 bg-slate-950/30 p-3 space-y-1">
                            {flags.slice(0, 40).map((k) => (
                              <div key={k} className="break-words">
                                + {k}
                              </div>
                            ))}
                            {flags.length > 40 && <div className="text-slate-500">顯示前 40 筆（共 {flags.length}）</div>}
                          </div>
                        )}
                      </div>

                      <div className="space-y-2">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.characters}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, characters: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">新增角色模板（worldpack.characters）</span>
                          <Badge variant="outline" className="text-xs">
                            +{chars.length}
                          </Badge>
                        </label>
                        {selection.characters && chars.length > 0 && (
                          <div className="text-xs text-slate-300 rounded-md border border-slate-700 bg-slate-950/30 p-3 space-y-1">
                            {chars.slice(0, 25).map((c, idx) => (
                              <div key={String(c.character_id || c.name || idx)} className="break-words">
                                + {c.name || c.character_id} ({c.character_id}) / role={c.role || 'npc'}
                              </div>
                            ))}
                            {chars.length > 25 && <div className="text-slate-500">顯示前 25 筆（共 {chars.length}）</div>}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">RAG 回寫（可選）</CardTitle>
                        {res.rag_note ? (
                          <Badge variant="outline" className="text-xs">
                            note
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="text-xs">
                            none
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        會以純文字寫入知識庫（world_id 分域）；寫入前仍會再次要求確認
                      </p>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <Checkbox
                          checked={selection.rag_note}
                          onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, rag_note: Boolean(v) })}
                          disabled={!res.rag_note}
                        />
                        <span className="text-sm text-slate-200">允許寫入 RAG note</span>
                      </label>
                      <Textarea
                        value={res.rag_note || '（無 rag_note）'}
                        readOnly
                        rows={10}
                        className="font-mono text-xs"
                      />
                      <div className="flex flex-wrap gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => onAddRagNote(item)}
                          disabled={busy || !res.rag_note || !selection.rag_note}
                        >
                          寫入知識庫（需確認）
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card className="bg-slate-900/40 border-slate-700">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between gap-2">
                      <CardTitle className="text-lg">Raw JSON</CardTitle>
                      <Badge variant="outline" className="text-xs">
                        response
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <Textarea value={JSON.stringify(res, null, 2)} readOnly rows={12} className="font-mono text-xs" />
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
