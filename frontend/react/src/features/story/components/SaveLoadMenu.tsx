import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { apiGet, apiPost } from '@/api/client'
import { useToast } from '@/hooks/useToast'
import { Save, FolderOpen, Clock, ChevronRight, Loader2, Trash2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface SaveSlot {
  slot_id: string
  name: string
  timestamp: string
  turn_count: number
  scene_image_url: string
  player_name: string
  stats: Record<string, any>
}

interface SaveLoadMenuProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  onLoadSuccess?: () => void
}

export function SaveLoadMenu({ open, onOpenChange, sessionId, onLoadSuccess }: SaveLoadMenuProps) {
  const { toast } = useToast()
  const [mode, setMode] = useState<'save' | 'load'>('save')
  const [slots, setSlots] = useState<SaveSlot[]>([])
  const [loading, setLoading] = useState(false)
  const [newSaveName, setNewSaveName] = useState('')
  const [actionLoading, setActionLoading] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      fetchSlots()
    }
  }, [open, sessionId])

  const fetchSlots = async () => {
    setLoading(true)
    try {
      const data = await apiGet<SaveSlot[]>(`/story/session/${sessionId}/saves`)
      setSlots(data)
    } catch (error) {
      toast({
        title: '獲取存檔失敗',
        description: '無法載入存檔列表。',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!newSaveName.trim()) {
      toast({ title: '請輸入存檔名稱', variant: 'destructive' })
      return
    }
    setActionLoading('saving')
    try {
      await apiPost(`/story/session/${sessionId}/save`, { slot_name: newSaveName })
      toast({ title: '存檔成功' })
      setNewSaveName('')
      fetchSlots()
    } catch (error) {
      toast({ title: '存檔失敗', variant: 'destructive' })
    } finally {
      setActionLoading(null)
    }
  }

  const handleLoad = async (slotId: string) => {
    setActionLoading(slotId)
    try {
      await apiPost(`/story/session/${sessionId}/load/${slotId}`, {})
      toast({ title: '讀檔成功', description: '正在恢復故事狀態...' })
      onOpenChange(false)
      onLoadSuccess?.()
    } catch (error) {
      toast({ title: '讀檔失敗', variant: 'destructive' })
    } finally {
      setActionLoading(null)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col bg-slate-950 border-indigo-900/50 text-slate-100 shadow-2xl overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-2">
          <div className="flex items-center justify-between">
            <DialogTitle className="text-2xl font-bold flex items-center gap-2 text-white">
              {mode === 'save' ? <Save className="text-indigo-400" /> : <FolderOpen className="text-emerald-400" />}
              {mode === 'save' ? '紀錄冒險' : '恢復記憶'}
            </DialogTitle>
            <div className="bg-slate-900/80 p-1 rounded-lg border border-white/5 flex gap-1">
              <Button 
                variant={mode === 'save' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setMode('save')}
                className={mode === 'save' ? 'bg-indigo-600' : 'text-slate-400'}
              >
                存檔
              </Button>
              <Button 
                variant={mode === 'load' ? 'default' : 'ghost'} 
                size="sm" 
                onClick={() => setMode('load')}
                className={mode === 'load' ? 'bg-emerald-600' : 'text-slate-400'}
              >
                讀檔
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {mode === 'save' && (
            <div className="flex gap-3 bg-indigo-950/20 p-4 rounded-xl border border-indigo-500/20 shadow-inner">
              <Input 
                placeholder="輸入存檔名稱... (例如: 幽暗森林入口)" 
                value={newSaveName}
                onChange={(e) => setNewSaveName(e.target.value)}
                className="bg-slate-900 border-indigo-900/50 focus:ring-indigo-500/50 text-white"
                onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              />
              <Button 
                onClick={handleSave} 
                disabled={actionLoading === 'saving'}
                className="bg-indigo-600 hover:bg-indigo-500"
              >
                {actionLoading === 'saving' ? <Loader2 className="animate-spin" /> : '建立存檔'}
              </Button>
            </div>
          )}

          {loading ? (
            <div className="py-20 flex flex-col items-center gap-4">
              <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
              <p className="text-slate-400 animate-pulse">讀取存檔清單中...</p>
            </div>
          ) : slots.length === 0 ? (
            <div className="py-20 text-center border-2 border-dashed border-slate-800 rounded-2xl">
              <p className="text-slate-500 italic">尚未有任何存檔紀錄</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <AnimatePresence>
                {slots.map((slot, idx) => (
                  <motion.div
                    key={slot.slot_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    <Card className="group relative bg-slate-900/50 border-slate-800 hover:border-indigo-500/50 overflow-hidden transition-all cursor-default">
                      <div className="flex h-32">
                        {/* Thumbnail */}
                        <div className="w-1/3 bg-slate-950 flex-shrink-0 relative overflow-hidden">
                          {slot.scene_image_url ? (
                            <img src={slot.scene_image_url} alt="Scene" className="w-full h-full object-cover grayscale-[30%] group-hover:grayscale-0 transition-all duration-500" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-slate-700 font-bold text-xs uppercase">No Image</div>
                          )}
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent to-slate-900/90" />
                        </div>
                        
                        {/* Info */}
                        <div className="flex-1 p-4 flex flex-col justify-between">
                          <div>
                            <h4 className="font-bold text-white truncate text-sm mb-1">{slot.name}</h4>
                            <div className="flex items-center gap-3 text-[10px] text-slate-400">
                              <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {new Date(slot.timestamp).toLocaleString()}</span>
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between mt-2">
                            <Badge variant="outline" className="text-[10px] border-slate-700 text-slate-300">回合 {slot.turn_count}</Badge>
                            <Button 
                              size="sm" 
                              variant={mode === 'save' ? 'ghost' : 'default'}
                              className={mode === 'load' ? "bg-emerald-600 hover:bg-emerald-500 h-7 text-xs px-4" : "h-7 text-xs text-slate-500 hover:text-white"}
                              disabled={actionLoading !== null}
                              onClick={() => mode === 'load' && handleLoad(slot.slot_id)}
                            >
                              {actionLoading === slot.slot_id ? <Loader2 className="w-3 h-3 animate-spin" /> : (mode === 'load' ? '讀取' : '覆蓋')}
                            </Button>
                          </div>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>

        <DialogFooter className="px-6 py-4 border-t border-white/5 bg-slate-950/50">
          <Button variant="ghost" onClick={() => onOpenChange(false)} className="text-slate-400 hover:text-white">
            關閉
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
