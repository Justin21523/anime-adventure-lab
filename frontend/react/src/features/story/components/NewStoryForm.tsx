import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { usePersonas } from '../hooks/usePersonas'
import { useStorySession } from '../hooks/useStorySession'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'

const newStorySchema = z.object({
  player_name: z.string().min(1, '請輸入角色名稱'),
  persona_id: z.string().min(1, '請選擇一個人格'),
  world_id: z.string().optional(),
  initial_prompt: z.string().optional(),
})

type NewStoryFormData = z.infer<typeof newStorySchema>

interface NewStoryFormProps {
  onSuccess: (sessionId: string) => void
  onCancel: () => void
}

export function NewStoryForm({ onSuccess, onCancel }: NewStoryFormProps) {
  const { data: personas, isLoading: personasLoading } = usePersonas()
  const { createSession } = useStorySession()
  const [isCreating, setIsCreating] = useState(false)

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<NewStoryFormData>({
    resolver: zodResolver(newStorySchema),
    defaultValues: {
      player_name: '',
      persona_id: '',
      world_id: 'default',
      initial_prompt: '',
    },
  })

  const selectedPersonaId = watch('persona_id')
  const selectedPersona = personas?.find((p) => p.persona_id === selectedPersonaId)

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
                      {persona.name} - {persona.style}
                    </option>
                  ))}
                </select>
                {errors.persona_id && (
                  <p className="text-sm text-red-400 mt-1">{errors.persona_id.message}</p>
                )}
                {selectedPersona && (
                  <p className="text-sm text-slate-400 mt-2">{selectedPersona.description}</p>
                )}
              </>
            )}
          </div>

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
