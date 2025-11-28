import { useState } from 'react'
import { useT2IGenerate } from '../hooks/useT2IGenerate'
import { useLoRAs } from '../hooks/useLoRAs'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { LoRAConfig } from '../types/t2i.types'

interface T2IGeneratorProps {
  sessionId?: string // Optional: link to story session for scene generation
  onImageGenerated?: (imageUrl: string) => void
}

export function T2IGenerator({ sessionId, onImageGenerated }: T2IGeneratorProps) {
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('low quality, blurry, distorted, ugly')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(768)
  const [steps, setSteps] = useState(30)
  const [cfgScale, setCfgScale] = useState(7.5)
  const [seed, setSeed] = useState<number | undefined>(undefined)
  const [selectedLoras, setSelectedLoras] = useState<LoRAConfig[]>([])

  const generateMutation = useT2IGenerate()
  const { data: lorasData } = useLoRAs()

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    const result = await generateMutation.mutateAsync({
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      num_inference_steps: steps,
      guidance_scale: cfgScale,
      seed,
      loras: selectedLoras,
      session_id: sessionId,
    })

    if (result.images.length > 0 && onImageGenerated) {
      onImageGenerated(result.images[0].image_url)
    }
  }

  const toggleLora = (loraName: string) => {
    const existing = selectedLoras.find((l) => l.name === loraName)
    if (existing) {
      setSelectedLoras(selectedLoras.filter((l) => l.name !== loraName))
    } else {
      setSelectedLoras([...selectedLoras, { name: loraName, weight: 0.8 }])
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>場景生成器</CardTitle>
        <p className="text-sm text-slate-400">
          {sessionId ? '為當前故事場景生成視覺畫面' : '生成圖像'}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Prompt */}
        <div>
          <Label>提示詞 (Prompt)</Label>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="描述你想生成的場景、角色或事件..."
            rows={3}
            className="mt-1"
          />
        </div>

        {/* Negative Prompt */}
        <div>
          <Label>負面提示詞 (Negative Prompt)</Label>
          <Textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="不想要的元素..."
            rows={2}
            className="mt-1"
          />
        </div>

        {/* Dimensions */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>寬度 (Width)</Label>
            <Input
              type="number"
              value={width}
              onChange={(e) => setWidth(parseInt(e.target.value) || 512)}
              min={256}
              max={1024}
              step={64}
              className="mt-1"
            />
          </div>
          <div>
            <Label>高度 (Height)</Label>
            <Input
              type="number"
              value={height}
              onChange={(e) => setHeight(parseInt(e.target.value) || 768)}
              min={256}
              max={1024}
              step={64}
              className="mt-1"
            />
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>推理步數 (Steps)</Label>
            <Input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 30)}
              min={10}
              max={100}
              className="mt-1"
            />
          </div>
          <div>
            <Label>引導強度 (CFG Scale)</Label>
            <Input
              type="number"
              value={cfgScale}
              onChange={(e) => setCfgScale(parseFloat(e.target.value) || 7.5)}
              min={1}
              max={20}
              step={0.5}
              className="mt-1"
            />
          </div>
        </div>

        {/* Seed */}
        <div>
          <Label>隨機種子 (可選)</Label>
          <Input
            type="number"
            value={seed || ''}
            onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value) : undefined)}
            placeholder="留空使用隨機種子"
            className="mt-1"
          />
        </div>

        {/* LoRA Selection */}
        {lorasData && lorasData.loras.length > 0 && (
          <div>
            <Label>LoRA 模型 (風格)</Label>
            <div className="flex flex-wrap gap-2 mt-2">
              {lorasData.loras.map((lora) => {
                const isSelected = selectedLoras.some((l) => l.name === lora.name)
                return (
                  <button
                    key={lora.name}
                    onClick={() => toggleLora(lora.name)}
                    className={`px-3 py-1 rounded text-sm transition-colors ${
                      isSelected
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {lora.name}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Generate Button */}
        <Button
          onClick={handleGenerate}
          disabled={!prompt.trim() || generateMutation.isPending}
          className="w-full"
        >
          {generateMutation.isPending ? '生成中...' : '生成圖像'}
        </Button>

        {/* Generation Info */}
        {generateMutation.data && (
          <div className="p-3 bg-green-900/20 border border-green-500/30 rounded text-sm">
            <p className="text-green-400">
              生成成功！耗時: {generateMutation.data.generation_time.toFixed(2)}s
            </p>
            <p className="text-slate-400 text-xs mt-1">
              模型: {generateMutation.data.model_used}
            </p>
          </div>
        )}

        {/* Error */}
        {generateMutation.error && (
          <div className="p-3 bg-red-900/20 border border-red-500/30 rounded text-sm">
            <p className="text-red-400">
              生成失敗：{generateMutation.error instanceof Error ? generateMutation.error.message : '未知錯誤'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
