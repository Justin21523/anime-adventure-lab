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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { apiGet, apiPut } from '@/api/client'
import { useToast } from '@/hooks/useToast'
import { Loader2, Database, Cpu, Image as ImageIcon, Search } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface LocalModel {
  name: string
  path: string
  type: 'llm' | 'vlm' | 't2i' | 'embedding'
  size_gb: number
}

interface ModelConfig {
  chat_model: string
  vqa_model: string
  caption_model: string
  sd_model: string
  embedding_model: string
}

interface ModelSettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function ModelSettingsDialog({ open, onOpenChange }: ModelSettingsDialogProps) {
  const { toast } = useToast()
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [availableModels, setAvailableModels] = useState<LocalModel[]>([])
  const [config, setConfig] = useState<ModelConfig | null>(null)

  useEffect(() => {
    if (open) {
      fetchData()
    }
  }, [open])

  const fetchData = async () => {
    setLoading(true)
    try {
      const [models, currentConfig] = await Promise.all([
        apiGet<LocalModel[]>('/models/local'),
        apiGet<ModelConfig>('/models/config'),
      ])
      setAvailableModels(models)
      setConfig(currentConfig)
    } catch (error) {
      toast({
        title: '獲取模型數據失敗',
        description: '無法從伺服器讀取本地模型列表或配置。',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!config) return
    setSaving(true)
    try {
      await apiPut('/models/config', config)
      toast({
        title: '配置已更新',
        description: '模型配置已成功套用，將在下次使用時重新載入。',
      })
      onOpenChange(false)
    } catch (error) {
      toast({
        title: '更新失敗',
        description: '無法儲存模型配置。',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }

  const filteredModels = (type: string) => 
    availableModels.filter(m => m.type === type)

  if (!open) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl bg-slate-950 border-indigo-900/50 text-slate-100 shadow-2xl backdrop-blur-xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold flex items-center gap-2">
            <Database className="w-6 h-6 text-indigo-400" />
            AI 模型設定
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            掃描目錄: <code className="bg-slate-900 px-1.5 py-0.5 rounded text-indigo-300">/mnt/c/ai_models/</code>
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="py-20 flex flex-col items-center justify-center gap-4">
            <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
            <p className="text-indigo-300 animate-pulse font-medium">正在掃描本地模型庫...</p>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            {/* LLM Chat Model */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-400" /> 語言模型 (LLM)
                </Label>
                <Badge variant="outline" className="text-[10px] border-indigo-500/30 text-indigo-400">對話與劇情生成</Badge>
              </div>
              <Select 
                value={config?.chat_model} 
                onValueChange={(v) => setConfig(prev => prev ? {...prev, chat_model: v} : null)}
              >
                <SelectTrigger className="bg-slate-900 border-slate-800 focus:ring-indigo-500/50">
                  <SelectValue placeholder="選擇語言模型..." />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-slate-800 text-slate-200">
                  {filteredModels('llm').map(m => (
                    <SelectItem key={m.path} value={m.name} className="focus:bg-indigo-600">
                      <div className="flex justify-between items-center w-full gap-8">
                        <span>{m.name}</span>
                        <span className="text-[10px] text-slate-500">{m.size_gb}GB</span>
                      </div>
                    </SelectItem>
                  ))}
                  <SelectItem value="Qwen/Qwen2.5-7B-Instruct">Qwen2.5-7B-Instruct (預設)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* VLM Model */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Search className="w-4 h-4 text-emerald-400" /> 視覺模型 (VLM)
                </Label>
                <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400">圖片理解與分析</Badge>
              </div>
              <Select 
                value={config?.vqa_model} 
                onValueChange={(v) => setConfig(prev => prev ? {...prev, vqa_model: v, caption_model: v} : null)}
              >
                <SelectTrigger className="bg-slate-900 border-slate-800 focus:ring-emerald-500/50">
                  <SelectValue placeholder="選擇視覺模型..." />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-slate-800 text-slate-200">
                  {filteredModels('vlm').map(m => (
                    <SelectItem key={m.path} value={m.name} className="focus:bg-emerald-600">
                      <div className="flex justify-between items-center w-full gap-8">
                        <span>{m.name}</span>
                        <span className="text-[10px] text-slate-500">{m.size_gb}GB</span>
                      </div>
                    </SelectItem>
                  ))}
                  <SelectItem value="llava-hf/llava-1.5-7b-hf">Llava 1.5 7B (預設)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* T2I Model */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <Label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <ImageIcon className="w-4 h-4 text-purple-400" /> 繪圖模型 (T2I)
                </Label>
                <Badge variant="outline" className="text-[10px] border-purple-500/30 text-purple-400">場景與角色生成</Badge>
              </div>
              <Select 
                value={config?.sd_model} 
                onValueChange={(v) => setConfig(prev => prev ? {...prev, sd_model: v} : null)}
              >
                <SelectTrigger className="bg-slate-900 border-slate-800 focus:ring-purple-500/50">
                  <SelectValue placeholder="選擇繪圖模型..." />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-slate-800 text-slate-200">
                  {filteredModels('t2i').map(m => (
                    <SelectItem key={m.path} value={m.name} className="focus:bg-purple-600">
                      <div className="flex justify-between items-center w-full gap-8">
                        <span>{m.name}</span>
                        <span className="text-[10px] text-slate-500">{m.size_gb}GB</span>
                      </div>
                    </SelectItem>
                  ))}
                  <SelectItem value="runwayml/stable-diffusion-v1-5">SD v1.5 (預設)</SelectItem>
                  <SelectItem value="stabilityai/stable-diffusion-xl-base-1.0">SDXL Base (預設)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        <DialogFooter className="border-t border-slate-800/50 pt-4 mt-2">
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
            取消
          </Button>
          <Button 
            className="bg-indigo-600 hover:bg-indigo-500 text-white gap-2 min-w-[120px]" 
            onClick={handleSave}
            disabled={loading || saving}
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : '儲存配置'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
