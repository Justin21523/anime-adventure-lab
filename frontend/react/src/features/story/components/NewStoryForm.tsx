import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { usePersonas } from '../hooks/usePersonas'
import { useStorySession } from '../hooks/useStorySession'
import { useWorld } from '@/features/worlds/hooks/useWorld'
import { useWorlds } from '@/features/worlds/hooks/useWorlds'
import { useRuntimePresets } from '@/features/runtime/hooks/useRuntimePresets'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'

const newStorySchema = z.object({
  player_name: z.string().min(1, '請輸入角色名稱'),
  persona_id: z.string().min(1, '請選擇一個人格'),
  world_id: z.string().min(1, '請選擇世界'),
  runtime_preset_id: z.string().optional(),
  player_template_id: z.string().optional(),
  initial_prompt: z.string().optional(),
})

type NewStoryFormData = z.infer<typeof newStorySchema>

interface NewStoryFormProps {
  onSuccess: (sessionId: string) => void
  onCancel: () => void
}

export function NewStoryForm({ onSuccess, onCancel }: NewStoryFormProps) {
  const { data: personas, isLoading: personasLoading } = usePersonas()
  const { data: worlds, isLoading: worldsLoading } = useWorlds()
  const { data: runtimeCatalog, isLoading: runtimeLoading } = useRuntimePresets()
  const { createSession } = useStorySession()
  const [isCreating, setIsCreating] = useState(false)

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
    getValues,
  } = useForm<NewStoryFormData>({
    resolver: zodResolver(newStorySchema),
    defaultValues: {
      player_name: '',
      persona_id: '',
      world_id: 'default',
      runtime_preset_id: '',
      player_template_id: undefined,
      initial_prompt: '',
    },
  })

  const selectedPersonaId = watch('persona_id')
  const selectedPersona = personas?.find((p) => p.persona_id === selectedPersonaId)
  const selectedWorldId = watch('world_id')
  const { data: selectedWorld } = useWorld(selectedWorldId || 'default')
  const selectedRuntimeId = watch('runtime_preset_id') || ''
  const runtimePresets = runtimeCatalog?.presets || []
  const runtimeDefaultId = runtimeCatalog?.default_preset_id || ''
  const selectedRuntimePreset = runtimePresets.find((p) => p.preset_id === selectedRuntimeId) || null

  useEffect(() => {
    if (!selectedWorld) return
    const templates = selectedWorld.player_templates || []
    const current = getValues('player_template_id')
    if (templates.length === 0) {
      if (current) setValue('player_template_id', undefined)
      return
    }
    if (!current || !templates.some((t) => t.template_id === current)) {
      setValue('player_template_id', templates[0].template_id)
    }
  }, [selectedWorld?.world_id, selectedWorld?.updated_at, getValues, setValue])

  const onSubmit = async (data: NewStoryFormData) => {
    setIsCreating(true)
    try {
      const result = await createSession.mutateAsync(data)
      onSuccess(result.session_id)
    } catch (error) {
      console.error('Failed to create session:', error)
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>開始新冒險</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* 角色名稱 */}
          <div>
            <Label htmlFor="player_name">角色名稱 *</Label>
            <Input
              id="player_name"
              {...register('player_name')}
              placeholder="例如：艾莉亞"
              className="mt-1"
            />
            {errors.player_name && (
              <p className="text-sm text-red-400 mt-1">{errors.player_name.message}</p>
            )}
          </div>

          {/* 選擇人格 */}
          <div>
            <Label htmlFor="persona_id">故事敘述者 *</Label>
            {personasLoading ? (
              <p className="text-sm text-slate-400 mt-1">加載中...</p>
            ) : (
              <>
                <select
                  id="persona_id"
                  {...register('persona_id')}
                  className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
                >
                  <option value="">-- 選擇敘述風格 --</option>
                  {personas?.map((persona) => (
                    <option key={persona.persona_id} value={persona.persona_id}>
                      {persona.name}
                    </option>
                  ))}
                </select>
                {errors.persona_id && (
                  <p className="text-sm text-red-400 mt-1">{errors.persona_id.message}</p>
                )}
                {selectedPersona && (
                  <div className="mt-3 p-3 bg-slate-900/50 rounded-md border border-slate-700">
                    <p className="text-sm text-slate-300 mb-2">{selectedPersona.description}</p>
                    {selectedPersona.personality_traits.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {selectedPersona.personality_traits.map((trait, idx) => (
                          <span
                            key={idx}
                            className="text-xs px-2 py-1 bg-blue-500/20 text-blue-300 rounded"
                          >
                            {trait}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          {/* 選擇世界 */}
          <div>
            <Label htmlFor="world_id">世界 *</Label>
            {worldsLoading ? (
              <p className="text-sm text-slate-400 mt-1">加載中...</p>
            ) : (
              <>
                <select
                  id="world_id"
                  {...register('world_id')}
                  className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
                >
                  {(worlds && worlds.length > 0) ? (
                    worlds.map((w) => (
                      <option key={w.world_id} value={w.world_id}>
                        {w.name} ({w.world_id})
                      </option>
                    ))
                  ) : (
                    <option value="default">預設世界 (default)</option>
                  )}
                </select>
                {errors.world_id && (
                  <p className="text-sm text-red-400 mt-1">{errors.world_id.message}</p>
                )}
                {selectedWorld && (
                  <div className="mt-3 p-3 bg-slate-900/50 rounded-md border border-slate-700">
                    <p className="text-sm text-slate-300 mb-1">{selectedWorld.description}</p>
                    <p className="text-xs text-slate-500">
                      setting: {selectedWorld.setting} • difficulty: {selectedWorld.difficulty}
                    </p>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Runtime preset（可選） */}
          <div>
            <Label htmlFor="runtime_preset_id">Runtime Preset（LLM + SDXL，可選）</Label>
            {runtimeLoading ? (
              <p className="text-sm text-slate-400 mt-1">加載中...</p>
            ) : (
              <>
                <select
                  id="runtime_preset_id"
                  {...register('runtime_preset_id')}
                  className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
                >
                  <option value="">
                    auto（世界預設：{selectedWorld?.runtime_preset_id || runtimeDefaultId || '—'}）
                  </option>
                  {runtimePresets.map((p) => (
                    <option key={p.preset_id} value={p.preset_id}>
                      {p.name} ({p.preset_id})
                    </option>
                  ))}
                </select>
                <div className="mt-2 text-xs text-slate-500">
                  {selectedRuntimePreset?.description
                    ? selectedRuntimePreset.description
                    : '提示：auto 會跟隨世界預設（若世界未設定，則使用 API 預設）。'}
                </div>
              </>
            )}
          </div>

          {/* 玩家角色風格模板（依世界提供） */}
          {selectedWorld && selectedWorld.player_templates.length > 0 && (
            <div>
              <Label htmlFor="player_template_id">玩家角色風格</Label>
              <select
                id="player_template_id"
                {...register('player_template_id')}
                className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
              >
                {selectedWorld.player_templates.map((t) => (
                  <option key={t.template_id} value={t.template_id}>
                    {t.name} ({t.template_id})
                  </option>
                ))}
              </select>
              {errors.player_template_id && (
                <p className="text-sm text-red-400 mt-1">{errors.player_template_id.message}</p>
              )}
            </div>
          )}

          {/* 初始提示（可選） */}
          <div>
            <Label htmlFor="initial_prompt">初始場景（可選）</Label>
            <Textarea
              id="initial_prompt"
              {...register('initial_prompt')}
              placeholder="描述你想開始的場景，或留空讓系統自動生成..."
              className="mt-1"
              rows={4}
            />
            <p className="text-xs text-slate-500 mt-1">
              例如：「你站在一座古老城堡的大門前...」
            </p>
          </div>

          {/* 操作按鈕 */}
          <div className="flex gap-3">
            <Button type="submit" disabled={isCreating} className="flex-1">
              {isCreating ? '創建中...' : '開始冒險'}
            </Button>
            <Button type="button" variant="outline" onClick={onCancel}>
              取消
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
