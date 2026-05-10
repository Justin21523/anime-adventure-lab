import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Textarea } from '@/components/ui/textarea'
import type { WorldPack } from '@/features/worlds/types/world.types'
import { applyWorldPackSelection, computeWorldPackDiff } from '@/features/worlds/utils/worldPackPatch'
import type { ReviewQueueItem } from '../types/reviewQueue.types'

type WorldAiQueueItem = Extract<ReviewQueueItem, { kind: 'world_ai' }>

interface WorldAiReviewDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  item: WorldAiQueueItem | null
  worldpack: WorldPack | null
  onUpdateSelection: (itemId: string, selection: WorldAiQueueItem['selection']) => void
  onApplyToWorld: (item: WorldAiQueueItem, syncAfter: boolean) => void
  onRemove: (itemId: string) => void
  busy?: boolean
}

export function WorldAiReviewDialog({
  open,
  onOpenChange,
  item,
  worldpack,
  onUpdateSelection,
  onApplyToWorld,
  onRemove,
  busy,
}: WorldAiReviewDialogProps) {
  const title = item ? `世界 AI Patch 審核（${item.id.slice(0, 8)}…）` : '世界 AI Patch 審核'
  const res = item?.response || null
  const selection = item?.selection || {
    world: true,
    world_flags: true,
    player_templates: true,
    characters: true,
    visual: true,
  }

  const canDiff = Boolean(worldpack && res?.worldpack)
  const diff = canDiff && worldpack && res?.worldpack ? computeWorldPackDiff(worldpack, res.worldpack) : null

  const previewWorld = canDiff && worldpack && res?.worldpack ? applyWorldPackSelection(worldpack, res.worldpack, selection) : null

  const contributors = (res?.contributors || []).filter(Boolean)
  const errors = (res?.errors || []).filter(Boolean)

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
                    world_id={res.world_id} • {new Date(item!.created_at).toLocaleString()} • 子代理={contributors.length}
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
                    {diff ? (
                      <>
                        <Badge variant="outline" className="text-xs">
                          world {diff.worldFields.length}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          flags {diff.worldFlagsChanged.length}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          chars +{diff.charactersAdded.length}/~{diff.charactersUpdated.length}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          tpl +{diff.playerTemplatesAdded.length}/~{diff.playerTemplatesUpdated.length}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          visual {diff.visualFields.length}
                        </Badge>
                      </>
                    ) : (
                      <Badge variant="outline" className="text-xs">
                        diff unavailable
                      </Badge>
                    )}
                    {errors.length > 0 ? (
                      <Badge variant="destructive" className="text-xs">
                        errors {errors.length}
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
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onApplyToWorld(item, false)}
                      disabled={busy || !worldpack}
                    >
                      保存到世界
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => onApplyToWorld(item, true)}
                      disabled={busy || !worldpack}
                    >
                      保存 + 同步到本故事
                    </Button>
                  </div>
                </div>

                {!worldpack ? (
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardContent className="py-10 text-center text-slate-400">世界資料尚未載入（無法計算 diff / 套用）</CardContent>
                  </Card>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card className="bg-slate-900/40 border-slate-700">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">選擇要套用的分區</CardTitle>
                        <p className="text-xs text-slate-500 mt-1">
                          你可以只套用世界設定/角色/視覺其中一部分；套用前會先用目前 worldpack 計算 diff。
                        </p>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.world}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, world: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">世界基本資訊（name/description/setting/difficulty）</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.world_flags}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, world_flags: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">world_flags</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.characters}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, characters: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">角色模板（characters）</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.player_templates}
                            onCheckedChange={(v) =>
                              onUpdateSelection(item.id, { ...selection, player_templates: Boolean(v) })
                            }
                          />
                          <span className="text-sm text-slate-200">玩家模板（player_templates）</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <Checkbox
                            checked={selection.visual}
                            onCheckedChange={(v) => onUpdateSelection(item.id, { ...selection, visual: Boolean(v) })}
                          />
                          <span className="text-sm text-slate-200">視覺風格（visual / LoRA）</span>
                        </label>

                        {diff && (
                          <div className="pt-3 border-t border-slate-700 space-y-2">
                            <div className="text-xs text-slate-500">Diff 概覽（依目前 worldpack 計算）</div>
                            {diff.worldFields.length > 0 && (
                              <div className="text-xs text-slate-200 space-y-1">
                                {diff.worldFields.slice(0, 10).map((f) => (
                                  <div key={f.field}>
                                    {f.field}: {f.from} → {f.to}
                                  </div>
                                ))}
                              </div>
                            )}
                            {diff.worldFlagsChanged.length > 0 && (
                              <div className="text-xs text-slate-200">
                                flags: {diff.worldFlagsChanged.slice(0, 12).map((f) => f.key).join(', ')}
                                {diff.worldFlagsChanged.length > 12 ? '…' : ''}
                              </div>
                            )}
                            {(diff.charactersAdded.length > 0 || diff.charactersUpdated.length > 0) && (
                              <div className="text-xs text-slate-200">
                                characters +{diff.charactersAdded.length}/~{diff.charactersUpdated.length}
                              </div>
                            )}
                            {(diff.playerTemplatesAdded.length > 0 || diff.playerTemplatesUpdated.length > 0) && (
                              <div className="text-xs text-slate-200">
                                templates +{diff.playerTemplatesAdded.length}/~{diff.playerTemplatesUpdated.length}
                              </div>
                            )}
                            {diff.visualFields.length > 0 && (
                              <div className="text-xs text-slate-200">
                                visual: {diff.visualFields.map((v) => v.field).join(', ')}
                              </div>
                            )}
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    <Card className="bg-slate-900/40 border-slate-700">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">預覽（套用後 worldpack）</CardTitle>
                        <p className="text-xs text-slate-500 mt-1">這是依照勾選分區合併後的 worldpack（尚未保存）。</p>
                      </CardHeader>
                      <CardContent>
                        <Textarea
                          value={previewWorld ? JSON.stringify(previewWorld, null, 2) : '（無法預覽）'}
                          readOnly
                          rows={18}
                          className="font-mono text-xs"
                        />
                      </CardContent>
                    </Card>
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg">子代理貢獻</CardTitle>
                        <Badge variant="outline" className="text-xs">
                          {contributors.length}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {contributors.length === 0 ? (
                        <div className="text-sm text-slate-500">（無）</div>
                      ) : (
                        contributors.slice(0, 12).map((c: any, idx: number) => (
                          <div key={idx} className="p-3 rounded-lg border border-slate-700 bg-slate-950/30">
                            <div className="text-xs text-slate-200 font-semibold">
                              {c.agent || c.name || `agent_${idx + 1}`}
                            </div>
                            {c.reasoning && (
                              <div className="text-xs text-slate-400 mt-1 whitespace-pre-wrap break-words">
                                {String(c.reasoning)}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/40 border-slate-700">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">Raw JSON</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Textarea value={JSON.stringify(res, null, 2)} readOnly rows={18} className="font-mono text-xs" />
                    </CardContent>
                  </Card>
                </div>
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

