import { useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { apiGet, apiPost, apiPut, apiUploadFile } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import { useUiStore } from '@/stores/uiStore'
import { useStoryWorldSync } from '@/features/story/hooks/useStoryWorldSync'
import { useStoryWorldWritebackSuggest } from '@/features/story/hooks/useStoryWorldWritebackSuggest'
import type { StoryWorldWritebackSuggestResponse } from '@/features/story/types/story.types'
import type { RAGAddDocumentResponse } from '@/features/rag/types/rag.types'
import { useAgentCatalog } from '@/features/agent/hooks/useAgentCatalog'
import { useRuntimePresets } from '@/features/runtime/hooks/useRuntimePresets'
import { JobProgressCard } from '@/features/jobs/components/JobProgressCard'
import { useJob } from '@/features/jobs/hooks/useJob'
import type { ReviewQueueItem } from '@/features/story/types/reviewQueue.types'
import { enqueueReviewQueueItem } from '@/features/story/lib/reviewQueueStorage'
import { useLoRAs } from '@/features/t2i/hooks/useLoRAs'
import { RAGDashboard } from '@/features/rag/components/RAGDashboard'
import { RAGMaintenancePanel } from '@/features/rag/components/RAGMaintenancePanel'
import { SearchInterface } from '@/features/rag/components/SearchInterface'
import { DocumentUploader } from '@/features/rag/components/DocumentUploader'
import { DocumentList } from '@/features/rag/components/DocumentList'
import { useWorld } from '../hooks/useWorld'
import { useWorlds } from '../hooks/useWorlds'
import { useWorldAgentSuggest } from '../hooks/useWorldAgents'
import { useCreateWorld, useDeleteWorld, useUpdateWorld } from '../hooks/useWorldMutations'
import type {
  WorldAgentProfile,
  WorldCharacterTemplate,
  WorldAgentSuggestResponse,
  WorldPack,
  WorldPlayerTemplate,
  WorldVisualStyle,
} from '../types/world.types'

type AiApplySelection = {
  world: boolean
  world_flags: boolean
  player_templates: boolean
  characters: boolean
  visual: boolean
}

type AiDiff = {
  worldChanged: boolean
  worldFields: Array<{ field: string; from: string; to: string }>
  worldFlagsChanged: Array<{ key: string; from: boolean | undefined; to: boolean | undefined }>
  visualChanged: boolean
  visualFields: Array<{ field: string; from: string; to: string }>
  charactersAdded: Array<{ id: string; name: string }>
  charactersUpdated: Array<{ id: string; name: string }>
  playerTemplatesAdded: Array<{ id: string; name: string }>
  playerTemplatesUpdated: Array<{ id: string; name: string }>
}

function parseCsv(value: string): string[] {
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean)
}

function joinCsv(values: string[]): string {
  return (values || []).join(', ')
}

function safeJsonParseObject(value: string): { ok: true; obj: Record<string, any> } | { ok: false; error: string } {
  try {
    const parsed = JSON.parse(value || '{}')
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { ok: false, error: '必須是 JSON 物件 (object)' }
    }
    return { ok: true, obj: parsed }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : 'JSON 解析失敗' }
  }
}

function coerceBooleanRecord(obj: Record<string, any>): { ok: true; value: Record<string, boolean> } | { ok: false; error: string } {
  const out: Record<string, boolean> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v !== 'boolean') {
      return { ok: false, error: `world_flags['${k}'] 必須是 boolean` }
    }
    out[k] = v
  }
  return { ok: true, value: out }
}

function computeAiDiff(base: WorldPack, candidate: WorldPack): AiDiff {
  const worldFields: AiDiff['worldFields'] = []
  for (const field of ['name', 'description', 'setting', 'difficulty'] as const) {
    const from = String((base as any)[field] ?? '')
    const to = String((candidate as any)[field] ?? '')
    if (from !== to) worldFields.push({ field, from, to })
  }

  const baseFlags = base.world_flags || {}
  const candFlags = candidate.world_flags || {}
  const keys = new Set([...Object.keys(baseFlags), ...Object.keys(candFlags)])
  const worldFlagsChanged: AiDiff['worldFlagsChanged'] = []
  for (const key of Array.from(keys).sort()) {
    const from = baseFlags[key]
    const to = candFlags[key]
    if (from !== to) worldFlagsChanged.push({ key, from, to })
  }

  const visualFields: AiDiff['visualFields'] = []
  const baseVisual = base.visual
  const candVisual = candidate.visual
  const visualPairs: Array<[string, any, any]> = [
    ['prompt_prefix', baseVisual?.prompt_prefix, candVisual?.prompt_prefix],
    ['negative_prompt', baseVisual?.negative_prompt, candVisual?.negative_prompt],
    ['base_model', baseVisual?.base_model ?? '', candVisual?.base_model ?? ''],
    ['default_loras', JSON.stringify(baseVisual?.default_loras || []), JSON.stringify(candVisual?.default_loras || [])],
  ]
  for (const [field, from, to] of visualPairs) {
    if (String(from ?? '') !== String(to ?? '')) visualFields.push({ field, from: String(from ?? ''), to: String(to ?? '') })
  }

  const baseChars = new Map((base.characters || []).map((c) => [c.character_id, c]))
  const candChars = new Map((candidate.characters || []).map((c) => [c.character_id, c]))
  const charactersAdded: AiDiff['charactersAdded'] = []
  const charactersUpdated: AiDiff['charactersUpdated'] = []
  for (const [id, c] of candChars.entries()) {
    if (!baseChars.has(id)) {
      charactersAdded.push({ id, name: c.name || id })
      continue
    }
    const before = baseChars.get(id)
    if (before && JSON.stringify(before) !== JSON.stringify(c)) {
      charactersUpdated.push({ id, name: c.name || id })
    }
  }

  const baseTpl = new Map((base.player_templates || []).map((t) => [t.template_id, t]))
  const candTpl = new Map((candidate.player_templates || []).map((t) => [t.template_id, t]))
  const playerTemplatesAdded: AiDiff['playerTemplatesAdded'] = []
  const playerTemplatesUpdated: AiDiff['playerTemplatesUpdated'] = []
  for (const [id, t] of candTpl.entries()) {
    if (!baseTpl.has(id)) {
      playerTemplatesAdded.push({ id, name: t.name || id })
      continue
    }
    const before = baseTpl.get(id)
    if (before && JSON.stringify(before) !== JSON.stringify(t)) {
      playerTemplatesUpdated.push({ id, name: t.name || id })
    }
  }

  return {
    worldChanged: worldFields.length > 0,
    worldFields,
    worldFlagsChanged,
    visualChanged: visualFields.length > 0,
    visualFields,
    charactersAdded,
    charactersUpdated,
    playerTemplatesAdded,
    playerTemplatesUpdated,
  }
}

function applyAiSelectionToWorldPack(base: WorldPack, candidate: WorldPack, selection: AiApplySelection): WorldPack {
  const next: WorldPack = JSON.parse(JSON.stringify(base)) as WorldPack

  if (selection.world) {
    next.name = candidate.name
    next.description = candidate.description
    next.setting = candidate.setting
    next.difficulty = candidate.difficulty
  }

  if (selection.world_flags) {
    next.world_flags = JSON.parse(JSON.stringify(candidate.world_flags || {})) as Record<string, boolean>
  }

  if (selection.visual) {
    next.visual = JSON.parse(JSON.stringify(candidate.visual)) as WorldVisualStyle
  }

  if (selection.player_templates) {
    const byId = new Map<string, WorldPlayerTemplate>()
    for (const t of next.player_templates || []) byId.set(t.template_id, t)
    for (const t of candidate.player_templates || []) byId.set(t.template_id, t)
    next.player_templates = Array.from(byId.values())
  }

  if (selection.characters) {
    const byId = new Map<string, WorldCharacterTemplate>()
    for (const c of next.characters || []) byId.set(c.character_id, c)
    for (const c of candidate.characters || []) byId.set(c.character_id, c)
    next.characters = Array.from(byId.values())
  }

  return next
}

interface WorldStudioPanelProps {
  worldId?: string
  sessionId?: string
}

export function WorldStudioPanel({ worldId, sessionId }: WorldStudioPanelProps) {
  const { addNotification } = useUiStore()
  const queryClient = useQueryClient()
  const { data: worlds, isLoading: worldsLoading } = useWorlds()
  const { data: lorasData } = useLoRAs()
  const syncWorldpack = useStoryWorldSync()
  const worldAgentSuggest = useWorldAgentSuggest()
  const worldWritebackSuggest = useStoryWorldWritebackSuggest()
  const { data: agentCatalog } = useAgentCatalog()
  const { data: runtimeCatalog } = useRuntimePresets()

  const storyCatalog = agentCatalog?.story
  const storySubAgents = storyCatalog?.sub_agents || []
  const storyAllowedTools = storyCatalog?.allowed_tools || []
  const defaultAgentProfile: WorldAgentProfile = useMemo(
    () =>
      (storyCatalog?.default_agent_profile as any) || {
        enabled: true,
        enabled_agents: [],
        max_tool_calls: 6,
        max_llm_calls: 1,
        allowed_tools: [],
      },
    [storyCatalog?.default_agent_profile]
  )

  const runtimePresets = runtimeCatalog?.presets || []
  const runtimeDefaultId = runtimeCatalog?.default_preset_id || null

  const [selectedWorldId, setSelectedWorldId] = useState<string>(worldId || 'default')
  const { data: worldpack, isLoading: worldLoading } = useWorld(selectedWorldId)

  const createWorld = useCreateWorld()
  const updateWorld = useUpdateWorld()
  const deleteWorld = useDeleteWorld()

  const [draft, setDraft] = useState<WorldPack | null>(null)
  const [worldFlagsText, setWorldFlagsText] = useState<string>('{}')
  const [syncMode, setSyncMode] = useState<'add_only' | 'merge'>('add_only')
  const [autoSync, setAutoSync] = useState<boolean>(false)
  const selectedRuntimePreset = useMemo(() => {
    const id = String(draft?.runtime_preset_id || '').trim()
    if (!id) return null
    return runtimePresets.find((p) => p.preset_id === id) || null
  }, [draft?.runtime_preset_id, runtimePresets])

  // Training (LoRA / QLoRA)
  const [trainingKind, setTrainingKind] = useState<'lora_sdxl' | 'llm_lora'>('lora_sdxl')
  const [trainingBaseModel, setTrainingBaseModel] = useState<string>('')
  const [trainingDatasetPath, setTrainingDatasetPath] = useState<string>('')
  const [trainingOutputName, setTrainingOutputName] = useState<string>('')
  const [trainingTags, setTrainingTags] = useState<string>('')
  const [trainingSimulate, setTrainingSimulate] = useState<boolean>(true)
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null)
  const [trainingSubmitting, setTrainingSubmitting] = useState<boolean>(false)
  const [sdxlTrainConfig, setSdxlTrainConfig] = useState({
    resolution: 1024,
    batch_size: 1,
    gradient_accumulation_steps: 4,
    max_steps: 1000,
    learning_rate: 1e-4,
    lora_rank: 16,
    seed: 42,
    mixed_precision: 'fp16',
    save_steps: 500,
  })
  const [llmTrainConfig, setLlmTrainConfig] = useState({
    max_length: 2048,
    batch_size: 1,
    gradient_accumulation_steps: 8,
    max_steps: 500,
    warmup_steps: 50,
    lr_scheduler: 'cosine',
    learning_rate: 2e-4,
    lora_rank: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    use_4bit: true,
    target_modules_csv: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
  })

  // Dataset Builder (images/zip -> captions -> metadata.jsonl -> train)
  const [datasetList, setDatasetList] = useState<any[]>([])
  const [datasetSelectedId, setDatasetSelectedId] = useState<string>('')
  const [datasetDetail, setDatasetDetail] = useState<any | null>(null)
  const [datasetUploadName, setDatasetUploadName] = useState<string>('')
  const [datasetUploadFile, setDatasetUploadFile] = useState<File | null>(null)
  const [datasetUploading, setDatasetUploading] = useState<boolean>(false)
  const [datasetBuilding, setDatasetBuilding] = useState<boolean>(false)
  const [datasetCaptionJobId, setDatasetCaptionJobId] = useState<string | null>(null)
  const [captionEdits, setCaptionEdits] = useState<Record<string, string>>({})
  const [captionSaving, setCaptionSaving] = useState<string | null>(null)

  const [aiInstruction, setAiInstruction] = useState<string>('')
  const [aiOptions, setAiOptions] = useState({
    include_visual: true,
    rag_top_k: 6,
    max_new_characters: 3,
    max_new_player_templates: 1,
    apply: false,
  })
  const [aiPreview, setAiPreview] = useState<WorldAgentSuggestResponse | null>(null)
  const [aiSelection, setAiSelection] = useState<AiApplySelection>({
    world: true,
    world_flags: true,
    player_templates: true,
    characters: true,
    visual: true,
  })

  const [writebackPreview, setWritebackPreview] = useState<StoryWorldWritebackSuggestResponse | null>(null)
  const [writebackSelection, setWritebackSelection] = useState<AiApplySelection>({
    world: false,
    world_flags: true,
    player_templates: false,
    characters: true,
    visual: false,
  })

  const aiDiff = useMemo(() => {
    if (!draft || !aiPreview?.worldpack) return null
    return computeAiDiff(draft, aiPreview.worldpack)
  }, [aiPreview?.worldpack, draft])

  const writebackDiff = useMemo(() => {
    if (!draft || !writebackPreview?.worldpack) return null
    return computeAiDiff(draft, writebackPreview.worldpack)
  }, [draft, writebackPreview?.worldpack])

  useEffect(() => {
    if (!draft || !aiPreview?.worldpack) return
    const diff = computeAiDiff(draft, aiPreview.worldpack)
    setAiSelection({
      world: diff.worldChanged,
      world_flags: diff.worldFlagsChanged.length > 0,
      player_templates: diff.playerTemplatesAdded.length + diff.playerTemplatesUpdated.length > 0,
      characters: diff.charactersAdded.length + diff.charactersUpdated.length > 0,
      visual: diff.visualChanged,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiPreview?.worldpack?.updated_at, aiPreview?.worldpack?.world_id])

  useEffect(() => {
    if (!draft || !writebackPreview?.worldpack) return
    const diff = computeAiDiff(draft, writebackPreview.worldpack)
    setWritebackSelection({
      world: diff.worldChanged,
      world_flags: diff.worldFlagsChanged.length > 0,
      player_templates: diff.playerTemplatesAdded.length + diff.playerTemplatesUpdated.length > 0,
      characters: diff.charactersAdded.length + diff.charactersUpdated.length > 0,
      visual: diff.visualChanged,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [writebackPreview?.worldpack, draft?.updated_at])

  // Best-effort prefill: use world visual base_model as SDXL training base model
  useEffect(() => {
    if (trainingKind !== 'lora_sdxl') return
    if (trainingBaseModel.trim()) return
    const candidate = draft?.visual?.base_model
    if (candidate && String(candidate).trim()) setTrainingBaseModel(String(candidate))
  }, [draft?.visual?.base_model, trainingBaseModel, trainingKind])

  const trainingJobState = useJob(trainingJobId, { enabled: Boolean(trainingJobId), refetchIntervalMs: 2000 })
  const lastTrainingJobStatusRef = useRef<string | null>(null)
  useEffect(() => {
    if (!trainingJobId) return
    const status = String(trainingJobState.job?.status || '')
    if (!status || lastTrainingJobStatusRef.current === status) return
    lastTrainingJobStatusRef.current = status
    if (status === 'completed') {
      queryClient.invalidateQueries({ queryKey: [CACHE_KEYS.t2i.loras()] })
    }
  }, [queryClient, trainingJobId, trainingJobState.job?.status])

  const [createOpen, setCreateOpen] = useState(false)
  const [createForm, setCreateForm] = useState({
    world_id: '',
    name: '',
    description: '',
    setting: 'fantasy',
    difficulty: 'medium',
  })

  const availableWorldIds = useMemo(() => new Set((worlds || []).map((w) => w.world_id)), [worlds])

  useEffect(() => {
    if (worldId) setSelectedWorldId(worldId)
  }, [worldId])

  useEffect(() => {
    if (!availableWorldIds.size) return
    if (!availableWorldIds.has(selectedWorldId)) {
      setSelectedWorldId('default')
    }
  }, [availableWorldIds, selectedWorldId])

  useEffect(() => {
    if (!worldpack) return
    setDraft(worldpack)
    setWorldFlagsText(JSON.stringify(worldpack.world_flags || {}, null, 2))
  }, [worldpack?.world_id, worldpack?.updated_at])

  const refreshDatasets = async (targetWorldId?: string) => {
    const wid = (targetWorldId || selectedWorldId || 'default').trim() || 'default'
    try {
      const res = await apiGet<any>(`/datasets/${wid}`, { retry: false })
      setDatasetList(Array.isArray(res?.datasets) ? res.datasets : [])
    } catch {
      setDatasetList([])
    }
  }

  useEffect(() => {
    void refreshDatasets()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedWorldId])

  useEffect(() => {
    if (!datasetSelectedId) {
      setDatasetDetail(null)
      setCaptionEdits({})
      return
    }
    let cancelled = false
    const tick = async () => {
      try {
        const res = await apiGet<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}?limit=80`, { retry: false })
        if (cancelled) return
        setDatasetDetail(res)
        const nextEdits: Record<string, string> = {}
        for (const item of Array.isArray(res?.items) ? res.items : []) {
          if (item?.item_id) nextEdits[String(item.item_id)] = String(item.caption || '')
        }
        setCaptionEdits(nextEdits)
      } catch {
        if (!cancelled) {
          setDatasetDetail(null)
          setCaptionEdits({})
        }
      }
    }
    void tick()
    return () => {
      cancelled = true
    }
  }, [datasetSelectedId, selectedWorldId])

  const datasetCaptionJobState = useJob(datasetCaptionJobId, {
    enabled: Boolean(datasetCaptionJobId),
    refetchIntervalMs: 2000,
  })
  const lastDatasetCaptionStatusRef = useRef<string | null>(null)
  useEffect(() => {
    if (!datasetCaptionJobId) return
    const status = String(datasetCaptionJobState.job?.status || '')
    if (!status || lastDatasetCaptionStatusRef.current === status) return
    lastDatasetCaptionStatusRef.current = status
    if (status === 'completed') {
      void refreshDatasets()
      if (datasetSelectedId) {
        void apiGet<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}?limit=80`, { retry: false })
          .then((res) => setDatasetDetail(res))
          .catch(() => {
            // ignore
          })
      }
    }
  }, [datasetCaptionJobId, datasetCaptionJobState.job?.status, datasetSelectedId, selectedWorldId])

  useEffect(() => {
    if (!sessionId) return
    try {
      const key = `worldstudio_auto_sync_${sessionId}`
      const raw = localStorage.getItem(key)
      if (raw === null) return
      setAutoSync(raw === 'true')
    } catch {
      // ignore
    }
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    try {
      const key = `worldstudio_auto_sync_${sessionId}`
      localStorage.setItem(key, String(autoSync))
    } catch {
      // ignore
    }
  }, [autoSync, sessionId])

  const updateDraft = (patch: Partial<WorldPack>) => {
    setDraft((prev) => (prev ? { ...prev, ...patch } : prev))
  }

  const updateVisual = (patch: Partial<WorldVisualStyle>) => {
    setDraft((prev) => (prev ? { ...prev, visual: { ...prev.visual, ...patch } } : prev))
  }

  const updateAgentProfile = (patch: Partial<WorldAgentProfile>) => {
    setDraft((prev) => {
      if (!prev) return prev
      const current = prev.agent_profile || defaultAgentProfile
      return { ...prev, agent_profile: { ...current, ...patch } }
    })
  }

  const enqueueToStoryReviewQueue = (item: ReviewQueueItem) => {
    if (!sessionId) {
      addNotification({ type: 'error', title: '缺少 session_id', message: '請在 Story 工作台內使用審核佇列功能' })
      return
    }
    enqueueReviewQueueItem(sessionId, item, 20)
    addNotification({ type: 'success', title: '已加入審核佇列', message: `id=${item.id}` })
  }

  const enqueueAiPreview = () => {
    if (!aiPreview) return
    const item: ReviewQueueItem = {
      kind: 'world_ai',
      id: `ai_${Date.now()}`,
      created_at: new Date().toISOString(),
      status: 'pending',
      world_id: selectedWorldId,
      selection: aiSelection as any,
      response: aiPreview as any,
    }
    enqueueToStoryReviewQueue(item)
  }

  const enqueueWritebackPreview = () => {
    if (!writebackPreview) return
    const item: ReviewQueueItem = {
      kind: 'world_writeback',
      id: `wb_${Date.now()}`,
      created_at: new Date().toISOString(),
      status: 'pending',
      world_id: selectedWorldId,
      selection: {
        world_flags: Boolean(writebackSelection.world_flags),
        characters: Boolean(writebackSelection.characters),
        rag_note: Boolean(writebackPreview.rag_note),
      },
      response: writebackPreview as any,
    }
    enqueueToStoryReviewQueue(item)
  }

  const updateRagProfile = (patch: { enable_rerank?: boolean }) => {
    setDraft((prev) => {
      if (!prev) return prev
      const current = prev.rag_profile || { enable_rerank: false }
      return { ...prev, rag_profile: { ...current, ...patch } }
    })
  }

  const toggleAgentEnabled = (agentId: string) => {
    setDraft((prev) => {
      if (!prev) return prev
      const profile = prev.agent_profile || defaultAgentProfile
      const current = new Set(profile.enabled_agents || [])
      if (current.has(agentId)) current.delete(agentId)
      else current.add(agentId)
      return { ...prev, agent_profile: { ...profile, enabled_agents: Array.from(current) } }
    })
  }

  const toggleAllowedTool = (toolId: string) => {
    setDraft((prev) => {
      if (!prev) return prev
      const profile = prev.agent_profile || defaultAgentProfile
      const current = new Set(profile.allowed_tools || [])
      if (current.has(toolId)) current.delete(toolId)
      else current.add(toolId)
      return { ...prev, agent_profile: { ...profile, allowed_tools: Array.from(current) } }
    })
  }

  const handleSubmitTraining = async () => {
    const baseModel = trainingBaseModel.trim()
    const datasetPath = trainingDatasetPath.trim()
    const outputName = trainingOutputName.trim()

    if (!baseModel || !datasetPath || !outputName) {
      addNotification({
        type: 'error',
        title: '欄位不足',
        message: '請填 base_model / dataset_path / output_name',
      })
      return
    }

    setTrainingSubmitting(true)
    try {
      const config =
        trainingKind === 'lora_sdxl'
          ? {
              ...sdxlTrainConfig,
            }
          : {
              ...llmTrainConfig,
              target_modules: parseCsv(llmTrainConfig.target_modules_csv),
            }

      const payload = {
        job_type: trainingKind,
        base_model: baseModel,
        dataset_path: datasetPath,
        output_name: outputName,
        simulate: trainingSimulate,
        tags: parseCsv(trainingTags),
        config,
      }

      const res = await apiPost<any, any>('/finetune/lora', payload)
      setTrainingJobId(res?.job_id || null)
      addNotification({
        type: 'success',
        title: '訓練任務已建立',
        message: res?.job_id ? `job_id: ${res.job_id}` : undefined,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '提交訓練失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    } finally {
      setTrainingSubmitting(false)
    }
  }

  const handleUploadDatasetZip = async () => {
    if (!datasetUploadFile) {
      addNotification({ type: 'error', title: '請選擇 zip 檔', message: '支援：images.zip（內含 png/jpg/webp）' })
      return
    }
    setDatasetUploading(true)
    try {
      const res = await apiUploadFile<any>(
        `/datasets/${selectedWorldId}/upload_zip`,
        datasetUploadFile,
        datasetUploadName.trim() ? { name: datasetUploadName.trim() } : undefined,
        undefined,
        { retry: false }
      )
      const id = String(res?.dataset?.dataset_id || '')
      addNotification({
        type: 'success',
        title: 'Dataset 已建立',
        message: id ? `dataset_id: ${id}` : undefined,
      })
      setDatasetUploadFile(null)
      setDatasetUploadName('')
      await refreshDatasets()
      if (id) setDatasetSelectedId(id)
    } catch (error) {
      addNotification({
        type: 'error',
        title: '上傳失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    } finally {
      setDatasetUploading(false)
    }
  }

  const handleBuildDatasetMetadata = async () => {
    if (!datasetSelectedId) return
    setDatasetBuilding(true)
    try {
      await apiPost<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}/build_metadata`)
      addNotification({ type: 'success', title: 'metadata.jsonl 已產生' })
      const res = await apiGet<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}?limit=80`, { retry: false })
      setDatasetDetail(res)
      await refreshDatasets()
    } catch (error) {
      addNotification({
        type: 'error',
        title: '產生 metadata.jsonl 失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    } finally {
      setDatasetBuilding(false)
    }
  }

  const handleAutoCaptionDataset = async () => {
    if (!datasetSelectedId) return
    try {
      const res = await apiPost<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}/auto_caption`)
      const jobId = String(res?.job_id || '')
      if (jobId) {
        setDatasetCaptionJobId(jobId)
        addNotification({ type: 'success', title: '已送出 auto-caption job', message: `job_id: ${jobId}` })
      } else {
        addNotification({ type: 'error', title: 'auto-caption 失敗', message: '未取得 job_id' })
      }
    } catch (error) {
      addNotification({
        type: 'error',
        title: 'auto-caption 失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    }
  }

  const handleSaveCaption = async (itemId: string) => {
    if (!datasetSelectedId) return
    setCaptionSaving(itemId)
    try {
      const caption = String(captionEdits[itemId] || '')
      await apiPut<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}/items/${itemId}`, { caption })
      addNotification({ type: 'success', title: '已保存 caption', message: itemId })
      const res = await apiGet<any>(`/datasets/${selectedWorldId}/${datasetSelectedId}?limit=80`, { retry: false })
      setDatasetDetail(res)
    } catch (error) {
      addNotification({
        type: 'error',
        title: '保存 caption 失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    } finally {
      setCaptionSaving(null)
    }
  }

  const handleTrainFromDataset = async () => {
    const datasetPath = String(datasetDetail?.dataset_path || '').trim()
    if (!datasetSelectedId || !datasetPath) return

    const baseModel = (trainingBaseModel.trim() || String(draft?.visual?.base_model || '').trim())
    if (!baseModel) {
      addNotification({
        type: 'error',
        title: '缺少 base_model',
        message: '請先在「訓練」或「視覺風格」填入 base_model（SDXL）',
      })
      return
    }

    const outputName = trainingOutputName.trim() || `${selectedWorldId}_${datasetSelectedId}`

    setTrainingKind('lora_sdxl')
    setTrainingBaseModel(baseModel)
    setTrainingDatasetPath(datasetPath)
    setTrainingOutputName(outputName)

    setTrainingSubmitting(true)
    try {
      const payload = {
        job_type: 'lora_sdxl',
        base_model: baseModel,
        dataset_path: datasetPath,
        output_name: outputName,
        simulate: trainingSimulate,
        tags: parseCsv(trainingTags),
        config: { ...sdxlTrainConfig },
      }
      const res = await apiPost<any, any>('/finetune/lora', payload)
      setTrainingJobId(res?.job_id || null)
      addNotification({
        type: 'success',
        title: '訓練任務已建立',
        message: res?.job_id ? `job_id: ${res.job_id}` : undefined,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '提交訓練失敗',
        message: error instanceof Error ? error.message : '未知錯誤',
      })
    } finally {
      setTrainingSubmitting(false)
    }
  }

  const addPlayerTemplate = () => {
    const template: WorldPlayerTemplate = {
      template_id: `template_${Date.now()}`,
      name: '新模板',
      description: '',
      personality_traits: [],
      speaking_style: '',
      background_story: '',
      motivations: [],
      persona_prompt: '',
    }
    setDraft((prev) => (prev ? { ...prev, player_templates: [...prev.player_templates, template] } : prev))
  }

  const updatePlayerTemplate = (index: number, patch: Partial<WorldPlayerTemplate>) => {
    setDraft((prev) => {
      if (!prev) return prev
      const next = prev.player_templates.slice()
      next[index] = { ...next[index], ...patch }
      return { ...prev, player_templates: next }
    })
  }

  const removePlayerTemplate = (index: number) => {
    setDraft((prev) => {
      if (!prev) return prev
      const next = prev.player_templates.slice()
      next.splice(index, 1)
      return { ...prev, player_templates: next }
    })
  }

  const addCharacter = () => {
    const character: WorldCharacterTemplate = {
      character_id: `character_${Date.now()}`,
      name: '新角色',
      role: 'npc',
      personality_traits: [],
      speaking_style: '',
      background_story: '',
      motivations: [],
      relationships: {},
      persona_prompt: '',
      content_restrictions: [],
      start_in_opening: false,
    }
    setDraft((prev) => (prev ? { ...prev, characters: [...prev.characters, character] } : prev))
  }

  const updateCharacter = (index: number, patch: Partial<WorldCharacterTemplate>) => {
    setDraft((prev) => {
      if (!prev) return prev
      const next = prev.characters.slice()
      next[index] = { ...next[index], ...patch }
      return { ...prev, characters: next }
    })
  }

  const removeCharacter = (index: number) => {
    setDraft((prev) => {
      if (!prev) return prev
      const next = prev.characters.slice()
      next.splice(index, 1)
      return { ...prev, characters: next }
    })
  }

  const toggleLora = (loraId: string) => {
    setDraft((prev) => {
      if (!prev) return prev
      const existing = prev.visual.default_loras.find((l) => l.lora_id === loraId)
      const nextLoras = existing
        ? prev.visual.default_loras.filter((l) => l.lora_id !== loraId)
        : [...prev.visual.default_loras, { lora_id: loraId, weight: 0.8 }]
      return { ...prev, visual: { ...prev.visual, default_loras: nextLoras } }
    })
  }

  const updateLoraWeight = (loraId: string, weight: number) => {
    setDraft((prev) => {
      if (!prev) return prev
      const nextLoras = prev.visual.default_loras.map((l) =>
        l.lora_id === loraId ? { ...l, weight } : l
      )
      return { ...prev, visual: { ...prev.visual, default_loras: nextLoras } }
    })
  }

  const handleSave = async (opts?: { skipAutoSync?: boolean; overrideDraft?: WorldPack }): Promise<WorldPack | null> => {
    const source = opts?.overrideDraft || draft
    if (!source) return null

    const parsed = safeJsonParseObject(worldFlagsText)
    if (!parsed.ok) {
      addNotification({ type: 'error', title: 'world_flags 無法解析', message: parsed.error })
      return null
    }
    const coerced = coerceBooleanRecord(parsed.obj)
    if (!coerced.ok) {
      addNotification({ type: 'error', title: 'world_flags 格式錯誤', message: coerced.error })
      return null
    }

    const payload: WorldPack = { ...source, world_flags: coerced.value }

    try {
      const saved = await updateWorld.mutateAsync({ worldId: payload.world_id, world: payload })
      setDraft(saved)
      setWorldFlagsText(JSON.stringify(saved.world_flags || {}, null, 2))
      addNotification({ type: 'success', title: '世界已保存', message: `World: ${saved.world_id}` })
      if (!opts?.skipAutoSync && autoSync && sessionId && worldId && selectedWorldId === worldId) {
        try {
          const syncRes = await syncWorldpack.mutateAsync({ sessionId, request: { mode: syncMode } })
          addNotification({
            type: 'success',
            title: '已自動套用到本故事',
            message: `新增角色 ${syncRes.characters_added?.length ?? 0} / 更新角色 ${syncRes.characters_updated?.length ?? 0}`,
          })
        } catch (err) {
          addNotification({
            type: 'error',
            title: '自動套用失敗',
            message: err instanceof Error ? err.message : '未知錯誤',
          })
        }
      }
      return saved
    } catch (err) {
      addNotification({ type: 'error', title: '保存失敗', message: err instanceof Error ? err.message : '未知錯誤' })
      return null
    }
  }

  const getTrainingOutputName = () => {
    const fromPayload = (trainingJobState.job as any)?.payload?.output_name || (trainingJobState.job as any)?.payload?.output
    const fromRoot = (trainingJobState.job as any)?.output_name || (trainingJobState.job as any)?.output_name
    return String(fromPayload || fromRoot || trainingOutputName || '').trim()
  }

  const handleAttachTrainingLoRA = async (opts?: { syncToStory?: boolean }) => {
    if (!draft) return
    const jobType = String((trainingJobState.job as any)?.job_type || (trainingJobState.job as any)?.payload?.job_type || '').trim()
    if (jobType !== 'lora_sdxl') {
      addNotification({ type: 'error', title: '目前不是 SDXL LoRA 訓練', message: `job_type=${jobType || 'unknown'}` })
      return
    }

    const outputName = getTrainingOutputName()
    if (!outputName) {
      addNotification({ type: 'error', title: '缺少 output_name', message: '請在訓練提交時填入 output_name' })
      return
    }

    const already = draft.visual.default_loras.some((l) => l.lora_id === outputName)
    if (already) {
      addNotification({ type: 'success', title: '此 LoRA 已在世界預設清單中', message: outputName })
      return
    }

    const nextDraft: WorldPack = {
      ...draft,
      visual: {
        ...draft.visual,
        default_loras: [...draft.visual.default_loras, { lora_id: outputName, weight: 0.8 }],
      },
    }

    const saved = await handleSave({ overrideDraft: nextDraft, skipAutoSync: Boolean(opts?.syncToStory) })
    if (!saved) return

    if (opts?.syncToStory) {
      if (!sessionId || !worldId || selectedWorldId !== worldId) {
        addNotification({
          type: 'error',
          title: '無法套用到本故事',
          message: worldId ? `目前故事 world_id=${worldId}，但你正在編輯 ${selectedWorldId}` : '缺少 world_id',
        })
        return
      }
      try {
        const syncRes = await syncWorldpack.mutateAsync({ sessionId, request: { mode: syncMode } })
        addNotification({
          type: 'success',
          title: '已套用到本故事',
          message: `新增角色 ${syncRes.characters_added?.length ?? 0} / 更新角色 ${syncRes.characters_updated?.length ?? 0}`,
        })
      } catch (err) {
        addNotification({
          type: 'error',
          title: '套用到本故事失敗',
          message: err instanceof Error ? err.message : '未知錯誤',
        })
      }
    }
  }

  const handleCreate = async () => {
    if (!createForm.world_id.trim() || !createForm.name.trim()) {
      addNotification({ type: 'error', title: '請填寫 world_id 與名稱' })
      return
    }
    try {
      const created = await createWorld.mutateAsync({
        world_id: createForm.world_id.trim(),
        name: createForm.name.trim(),
        description: createForm.description,
        setting: createForm.setting,
        difficulty: createForm.difficulty,
      })
      addNotification({ type: 'success', title: '世界已建立', message: `World: ${created.world_id}` })
      setCreateOpen(false)
      setSelectedWorldId(created.world_id)
    } catch (err) {
      addNotification({ type: 'error', title: '建立失敗', message: err instanceof Error ? err.message : '未知錯誤' })
    }
  }

  const handleDelete = async () => {
    if (!draft) return
    if (draft.world_id === 'default') {
      addNotification({ type: 'error', title: '不能刪除 default 世界' })
      return
    }
    if (!confirm(`確定要刪除世界 "${draft.name}" (${draft.world_id}) 嗎？此操作不可復原。`)) return

    try {
      await deleteWorld.mutateAsync(draft.world_id)
      addNotification({ type: 'success', title: '世界已刪除', message: draft.world_id })
      setSelectedWorldId('default')
    } catch (err) {
      addNotification({ type: 'error', title: '刪除失敗', message: err instanceof Error ? err.message : '未知錯誤' })
    }
  }

  const handleSaveAndSync = async () => {
    if (!sessionId || !worldId || selectedWorldId !== worldId) {
      addNotification({
        type: 'error',
        title: '無法套用到本故事',
        message: worldId ? `目前故事 world_id=${worldId}，但你正在編輯 ${selectedWorldId}` : '缺少 world_id',
      })
      return
    }
    const saved = await handleSave({ skipAutoSync: true })
    if (!saved) return
    await handleSyncToStory()
  }

  const handleSyncToStory = async () => {
    if (!sessionId) return
    if (!draft) return
    if (worldId && selectedWorldId !== worldId) {
      addNotification({
        type: 'error',
        title: '無法套用到本故事',
        message: `目前故事 world_id=${worldId}，但你正在編輯 ${selectedWorldId}（請新建 Story 才能切換 world）`,
      })
      return
    }

    try {
      const res = await syncWorldpack.mutateAsync({
        sessionId,
        request: { mode: syncMode },
      })
      addNotification({
        type: 'success',
        title: '已套用到本故事',
        message: `新增角色 ${res.characters_added?.length ?? 0} / 更新角色 ${res.characters_updated?.length ?? 0}`,
      })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '套用失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const handleAiSuggest = async (apply: boolean) => {
    if (!aiInstruction.trim()) {
      addNotification({ type: 'error', title: '請先輸入 AI 指令' })
      return
    }
    try {
      const res = await worldAgentSuggest.mutateAsync({
        worldId: selectedWorldId,
        request: {
          instruction: aiInstruction.trim(),
          apply,
          include_visual: aiOptions.include_visual,
          rag_top_k: aiOptions.rag_top_k,
          max_new_characters: aiOptions.max_new_characters,
          max_new_player_templates: aiOptions.max_new_player_templates,
        },
      })
      setAiPreview(res)
      if (!res.success) {
        addNotification({
          type: 'error',
          title: 'AI 建議失敗',
          message: (res.errors || []).slice(0, 2).join(' / ') || '未知錯誤',
        })
        return
      }
      addNotification({
        type: 'success',
        title: apply ? 'AI 已套用並保存' : 'AI 已產生預覽',
        message: `World: ${res.world_id} / 子代理: ${(res.contributors || []).length}`,
      })
      if (apply) {
        setDraft(res.worldpack)
        setWorldFlagsText(JSON.stringify(res.worldpack.world_flags || {}, null, 2))
        if (autoSync && sessionId && worldId && selectedWorldId === worldId) {
          try {
            const syncRes = await syncWorldpack.mutateAsync({ sessionId, request: { mode: syncMode } })
            addNotification({
              type: 'success',
              title: '已自動套用到本故事',
              message: `新增角色 ${syncRes.characters_added?.length ?? 0} / 更新角色 ${syncRes.characters_updated?.length ?? 0}`,
            })
          } catch (err) {
            addNotification({
              type: 'error',
              title: '自動套用失敗',
              message: err instanceof Error ? err.message : '未知錯誤',
            })
          }
        }
      }
    } catch (err) {
      addNotification({
        type: 'error',
        title: 'AI 呼叫失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const getAiSelectedNextDraft = (): WorldPack | null => {
    if (!aiPreview?.worldpack) return null
    if (!draft) return null
    return applyAiSelectionToWorldPack(draft, aiPreview.worldpack, aiSelection)
  }

  const applyAiSelectionToDraft = () => {
    const next = getAiSelectedNextDraft()
    if (!next) return
    setDraft(next)
    setWorldFlagsText(JSON.stringify(next.world_flags || {}, null, 2))
    addNotification({ type: 'success', title: '已套用 AI（選取分區）到草稿' })
  }

  const saveAiSelection = async (syncAfter: boolean) => {
    const next = getAiSelectedNextDraft()
    if (!next) return
    try {
      const saved = await updateWorld.mutateAsync({ worldId: next.world_id, world: next })
      setDraft(saved)
      setWorldFlagsText(JSON.stringify(saved.world_flags || {}, null, 2))
      addNotification({ type: 'success', title: 'AI（選取分區）已保存', message: `World: ${saved.world_id}` })

      if (syncAfter && sessionId) {
        await handleSyncToStory()
      }
    } catch (err) {
      addNotification({ type: 'error', title: '保存失敗', message: err instanceof Error ? err.message : '未知錯誤' })
    }
  }

  const handleWritebackSuggest = async () => {
    if (!sessionId) {
      addNotification({ type: 'error', title: '缺少 session_id', message: '請在 Story 工作台內使用世界回寫功能' })
      return
    }
    if (!worldId || selectedWorldId !== worldId) {
      addNotification({
        type: 'error',
        title: '無法從本故事回寫',
        message: worldId ? `目前故事 world_id=${worldId}，但你正在編輯 ${selectedWorldId}` : '缺少 world_id',
      })
      return
    }

    try {
      const res = await worldWritebackSuggest.mutateAsync({
        sessionId,
        request: {
          include_flags: true,
          include_characters: true,
          include_rag_note: true,
          max_new_characters: 10,
        },
      })
      setWritebackPreview(res)

      if (!res.success) {
        addNotification({
          type: 'error',
          title: '回寫預覽部分失敗',
          message: (res.errors || []).slice(0, 2).join(' / ') || '未知錯誤',
        })
        return
      }

      addNotification({
        type: 'success',
        title: '回寫預覽已產生',
        message: `flags新增 ${res.summary?.flags_added ?? 0} / 角色新增 ${res.summary?.characters_added ?? 0}`,
      })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '回寫匯出失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  const getWritebackSelectedNextDraft = (): WorldPack | null => {
    if (!writebackPreview?.worldpack) return null
    if (!draft) return null
    return applyAiSelectionToWorldPack(draft, writebackPreview.worldpack, writebackSelection)
  }

  const applyWritebackSelectionToDraft = () => {
    const next = getWritebackSelectedNextDraft()
    if (!next) return
    setDraft(next)
    setWorldFlagsText(JSON.stringify(next.world_flags || {}, null, 2))
    addNotification({ type: 'success', title: '已套用回寫（選取分區）到草稿' })
  }

  const saveWritebackSelection = async (syncAfter: boolean) => {
    const next = getWritebackSelectedNextDraft()
    if (!next) return
    try {
      const saved = await updateWorld.mutateAsync({ worldId: next.world_id, world: next })
      setDraft(saved)
      setWorldFlagsText(JSON.stringify(saved.world_flags || {}, null, 2))
      addNotification({ type: 'success', title: '回寫（選取分區）已保存', message: `World: ${saved.world_id}` })

      if (syncAfter && sessionId) {
        await handleSyncToStory()
      }
    } catch (err) {
      addNotification({ type: 'error', title: '保存失敗', message: err instanceof Error ? err.message : '未知錯誤' })
    }
  }

  const writebackAddRagNote = async () => {
    if (!writebackPreview?.rag_note) return
    if (!confirm('確定要把此摘要寫入世界知識庫 (RAG) 嗎？這會影響之後的 RAG 檢索結果。')) return

    const docId = `story_writeback_${selectedWorldId}_${sessionId || 'unknown'}_${Date.now()}`
    try {
      await apiPost<RAGAddDocumentResponse, any>('/rag/add', {
        doc_id: docId,
        content: writebackPreview.rag_note,
        metadata: {
          world_id: selectedWorldId,
          title: `Story writeback ${sessionId || ''}`.trim(),
          tags: ['story_writeback', 'world', selectedWorldId, ...(sessionId ? [sessionId] : [])],
        },
      })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.documents(selectedWorldId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.rag.stats(selectedWorldId) })
      addNotification({ type: 'success', title: '已寫入 RAG 知識庫', message: `doc_id=${docId}` })
    } catch (err) {
      addNotification({
        type: 'error',
        title: '寫入 RAG 失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  return (
    <div className="space-y-6">
      {/* Top bar */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <CardTitle>世界工作室</CardTitle>
              <p className="text-sm text-slate-400 mt-1">
                以 world_id 分域管理：世界設定 / 角色模板 / RAG 知識庫 / LoRA 視覺風格
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => setCreateOpen(true)}>
                新建世界
              </Button>
              {sessionId && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSaveAndSync}
                  disabled={!draft || updateWorld.isPending || syncWorldpack.isPending || !!(worldId && selectedWorldId !== worldId)}
                >
                  {updateWorld.isPending || syncWorldpack.isPending ? '處理中...' : '保存+套用'}
                </Button>
              )}
              <Button size="sm" onClick={() => void handleSave()} disabled={!draft || updateWorld.isPending}>
                {updateWorld.isPending ? '保存中...' : '保存'}
              </Button>
              <Button variant="outline" size="sm" onClick={handleDelete} disabled={!draft || deleteWorld.isPending}>
                刪除
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <Label htmlFor="world_selector">目前編輯世界</Label>
              <select
                id="world_selector"
                value={selectedWorldId}
                onChange={(e) => setSelectedWorldId(e.target.value)}
                className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
              >
                {worldsLoading && <option value="default">載入中...</option>}
                {!worldsLoading && (worlds || []).length === 0 && <option value="default">default</option>}
                {(worlds || []).map((w) => (
                  <option key={w.world_id} value={w.world_id}>
                    {w.name} ({w.world_id})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label>狀態</Label>
              <div className="mt-1 text-sm text-slate-300">
                {worldLoading ? '讀取中…' : draft ? `更新於 ${draft.updated_at}` : '尚未載入'}
              </div>
            </div>
          </div>

          {sessionId && (
            <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40">
              <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
                <div className="space-y-1">
                  <div className="text-sm font-semibold text-slate-200">套用到目前故事</div>
                  <div className="text-xs text-slate-400">提示：先按「保存」再套用（Session: {sessionId.slice(0, 8)}…）</div>
                </div>
                <div className="flex flex-col sm:flex-row gap-2 sm:items-end">
                  <div className="flex items-center gap-2 sm:pb-1">
                    <Checkbox checked={autoSync} onCheckedChange={(checked) => setAutoSync(checked)} />
                    <span className="text-xs text-slate-300">保存後自動套用</span>
                  </div>
                  <div>
                    <Label className="text-xs">同步模式</Label>
                    <select
                      value={syncMode}
                      onChange={(e) => setSyncMode(e.target.value as 'add_only' | 'merge')}
                      className="w-full sm:w-44 mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200 text-sm"
                    >
                      <option value="add_only">只新增缺少項目</option>
                      <option value="merge">合併更新 persona</option>
                    </select>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSyncToStory}
                    disabled={!draft || syncWorldpack.isPending || !!(worldId && selectedWorldId !== worldId)}
                  >
                    {syncWorldpack.isPending ? '套用中...' : '套用到本故事'}
                  </Button>
                </div>
              </div>
              {worldId && selectedWorldId !== worldId && (
                <div className="mt-2 text-xs text-amber-300">
                  目前故事 world_id={worldId}，你正在編輯 {selectedWorldId}；此故事不能直接切換 world（請新建 Story）。
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main tabs */}
      <Tabs defaultValue="world">
        <TabsList className="w-full overflow-x-auto">
          <TabsTrigger value="world">世界設定</TabsTrigger>
          <TabsTrigger value="players">玩家模板</TabsTrigger>
          <TabsTrigger value="characters">角色 / NPC</TabsTrigger>
          <TabsTrigger value="knowledge">知識庫 (RAG)</TabsTrigger>
          <TabsTrigger value="visual">視覺風格 (LoRA)</TabsTrigger>
          <TabsTrigger value="dataset">Dataset Builder</TabsTrigger>
          <TabsTrigger value="training">訓練</TabsTrigger>
          <TabsTrigger value="story_agents">故事代理設定</TabsTrigger>
          <TabsTrigger value="ai">AI 代理</TabsTrigger>
        </TabsList>

        <TabsContent value="world" className="mt-6 space-y-6">
          {!draft ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-slate-400">載入世界資料中...</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>基本資訊</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>world_id</Label>
                    <Input value={draft.world_id} disabled className="mt-1" />
                  </div>
                  <div>
                    <Label>名稱</Label>
                    <Input
                      value={draft.name}
                      onChange={(e) => updateDraft({ name: e.target.value })}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>描述</Label>
                    <Textarea
                      value={draft.description}
                      onChange={(e) => updateDraft({ description: e.target.value })}
                      className="mt-1"
                      rows={4}
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>setting</Label>
                      <Input
                        value={draft.setting}
                        onChange={(e) => updateDraft({ setting: e.target.value })}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>difficulty</Label>
                      <Input
                        value={draft.difficulty}
                        onChange={(e) => updateDraft({ difficulty: e.target.value })}
                        className="mt-1"
                      />
                    </div>
                  </div>

                  <div>
                    <Label>runtime preset（LLM + SDXL）</Label>
                    <select
                      value={draft.runtime_preset_id || ''}
                      onChange={(e) => {
                        const next = String(e.target.value || '').trim()
                        updateDraft({ runtime_preset_id: next ? next : null })
                      }}
                      className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
                    >
                      <option value="">
                        auto（API 預設：{runtimeDefaultId || '—'}）
                      </option>
                      {runtimePresets.map((p) => (
                        <option key={p.preset_id} value={p.preset_id}>
                          {p.name} ({p.preset_id})
                        </option>
                      ))}
                    </select>
                    <div className="mt-2 text-xs text-slate-500 space-y-1">
                      <div>
                        影響：Story 的 scene image 參數、LLM/T2I 的預設策略（不會自動下載模型）。
                      </div>
                      {selectedRuntimePreset?.description && <div>{selectedRuntimePreset.description}</div>}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>world_flags</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <p className="text-xs text-slate-500">
                    這裡用 JSON 編輯布林旗標（例如：{"{ \"magic_enabled\": true }"}）
                  </p>
                  <Textarea
                    value={worldFlagsText}
                    onChange={(e) => setWorldFlagsText(e.target.value)}
                    rows={10}
                    className="font-mono text-xs"
                  />
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="players" className="mt-6 space-y-6">
          {!draft ? null : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-100">玩家角色風格模板</h3>
                <Button size="sm" variant="outline" onClick={addPlayerTemplate}>
                  新增模板
                </Button>
              </div>

              {draft.player_templates.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center">
                    <p className="text-slate-400">尚未建立任何玩家模板</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {draft.player_templates.map((t, idx) => (
                    <Card key={t.template_id}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between gap-3">
                          <CardTitle className="text-base">{t.name || t.template_id}</CardTitle>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                            onClick={() => removePlayerTemplate(idx)}
                          >
                            移除
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <Label>template_id</Label>
                            <Input
                              value={t.template_id}
                              onChange={(e) => updatePlayerTemplate(idx, { template_id: e.target.value })}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label>名稱</Label>
                            <Input
                              value={t.name}
                              onChange={(e) => updatePlayerTemplate(idx, { name: e.target.value })}
                              className="mt-1"
                            />
                          </div>
                        </div>

                        <div>
                          <Label>描述</Label>
                          <Textarea
                            value={t.description}
                            onChange={(e) => updatePlayerTemplate(idx, { description: e.target.value })}
                            className="mt-1"
                            rows={2}
                          />
                        </div>

                        <div>
                          <Label>人格特質（逗號分隔）</Label>
                          <Input
                            value={joinCsv(t.personality_traits)}
                            onChange={(e) => updatePlayerTemplate(idx, { personality_traits: parseCsv(e.target.value) })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>說話/行動風格</Label>
                          <Input
                            value={t.speaking_style}
                            onChange={(e) => updatePlayerTemplate(idx, { speaking_style: e.target.value })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>背景故事</Label>
                          <Textarea
                            value={t.background_story}
                            onChange={(e) => updatePlayerTemplate(idx, { background_story: e.target.value })}
                            className="mt-1"
                            rows={3}
                          />
                        </div>

                        <div>
                          <Label>動機（逗號分隔）</Label>
                          <Input
                            value={joinCsv(t.motivations)}
                            onChange={(e) => updatePlayerTemplate(idx, { motivations: parseCsv(e.target.value) })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>persona_prompt</Label>
                          <Textarea
                            value={t.persona_prompt}
                            onChange={(e) => updatePlayerTemplate(idx, { persona_prompt: e.target.value })}
                            className="mt-1"
                            rows={3}
                          />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="characters" className="mt-6 space-y-6">
          {!draft ? null : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-100">NPC / Characters</h3>
                <Button size="sm" variant="outline" onClick={addCharacter}>
                  新增角色
                </Button>
              </div>

              {draft.characters.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center">
                    <p className="text-slate-400">尚未建立任何角色</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {draft.characters.map((c, idx) => (
                    <Card key={c.character_id}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between gap-3">
                          <CardTitle className="text-base">{c.name || c.character_id}</CardTitle>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                            onClick={() => removeCharacter(idx)}
                          >
                            移除
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <Label>character_id</Label>
                            <Input
                              value={c.character_id}
                              onChange={(e) => updateCharacter(idx, { character_id: e.target.value })}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label>名稱</Label>
                            <Input
                              value={c.name}
                              onChange={(e) => updateCharacter(idx, { name: e.target.value })}
                              className="mt-1"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <Label>role</Label>
                            <select
                              value={c.role}
                              onChange={(e) => updateCharacter(idx, { role: e.target.value as any })}
                              className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-600 rounded-md text-slate-200"
                            >
                              <option value="npc">npc</option>
                              <option value="companion">companion</option>
                              <option value="antagonist">antagonist</option>
                            </select>
                          </div>
                          <div className="flex items-end gap-2">
                            <label className="flex items-center gap-2 text-sm text-slate-200">
                              <input
                                type="checkbox"
                                checked={c.start_in_opening}
                                onChange={(e) => updateCharacter(idx, { start_in_opening: e.target.checked })}
                              />
                              開場出現
                            </label>
                          </div>
                        </div>

                        <div>
                          <Label>人格特質（逗號分隔）</Label>
                          <Input
                            value={joinCsv(c.personality_traits)}
                            onChange={(e) => updateCharacter(idx, { personality_traits: parseCsv(e.target.value) })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>說話風格</Label>
                          <Input
                            value={c.speaking_style}
                            onChange={(e) => updateCharacter(idx, { speaking_style: e.target.value })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>背景故事</Label>
                          <Textarea
                            value={c.background_story}
                            onChange={(e) => updateCharacter(idx, { background_story: e.target.value })}
                            className="mt-1"
                            rows={3}
                          />
                        </div>

                        <div>
                          <Label>動機（逗號分隔）</Label>
                          <Input
                            value={joinCsv(c.motivations)}
                            onChange={(e) => updateCharacter(idx, { motivations: parseCsv(e.target.value) })}
                            className="mt-1"
                          />
                        </div>

                        <div>
                          <Label>relationships（JSON）</Label>
                          <Textarea
                            value={JSON.stringify(c.relationships || {}, null, 2)}
                            onChange={(e) => {
                              const parsed = safeJsonParseObject(e.target.value)
                              if (parsed.ok) updateCharacter(idx, { relationships: parsed.obj })
                            }}
                            className="mt-1 font-mono text-xs"
                            rows={4}
                          />
                        </div>

                        <div>
                          <Label>persona_prompt</Label>
                          <Textarea
                            value={c.persona_prompt}
                            onChange={(e) => updateCharacter(idx, { persona_prompt: e.target.value })}
                            className="mt-1"
                            rows={3}
                          />
                        </div>

                        <div>
                          <Label>content_restrictions（逗號分隔）</Label>
                          <Input
                            value={joinCsv(c.content_restrictions)}
                            onChange={(e) => updateCharacter(idx, { content_restrictions: parseCsv(e.target.value) })}
                            className="mt-1"
                          />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="knowledge" className="mt-6 space-y-6">
          <div className="space-y-6">
            {draft && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle>Reranker（世界層級）</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <div className="text-sm text-slate-200">啟用 rerank（CrossEncoder）</div>
                      <div className="text-xs text-slate-500">
                        建議用 CPU；此開關會影響 rerank_mode=auto 的 Story（下一回合開始生效）。
                      </div>
                    </div>
                    <Switch
                      checked={Boolean(draft.rag_profile?.enable_rerank)}
                      onCheckedChange={(checked) => updateRagProfile({ enable_rerank: Boolean(checked) })}
                      label={draft.rag_profile?.enable_rerank ? 'ON' : 'OFF'}
                    />
                  </div>
                  <div className="text-xs text-slate-500">
                    也可在故事畫面把 rerank_mode 設為 on/off 進行覆蓋。
                  </div>
                </CardContent>
              </Card>
            )}
            <RAGDashboard worldId={selectedWorldId} />
            <RAGMaintenancePanel worldId={selectedWorldId} />
            <SearchInterface worldId={selectedWorldId} />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <DocumentUploader worldId={selectedWorldId} />
              </div>
              <div className="lg:col-span-2">
                <DocumentList worldId={selectedWorldId} />
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="visual" className="mt-6 space-y-6">
          {!draft ? null : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>視覺語言</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>prompt_prefix</Label>
                    <Textarea
                      value={draft.visual.prompt_prefix}
                      onChange={(e) => updateVisual({ prompt_prefix: e.target.value })}
                      className="mt-1"
                      rows={4}
                    />
                  </div>
                  <div>
                    <Label>negative_prompt</Label>
                    <Textarea
                      value={draft.visual.negative_prompt}
                      onChange={(e) => updateVisual({ negative_prompt: e.target.value })}
                      className="mt-1"
                      rows={4}
                    />
                  </div>
                  <div>
                    <Label>base_model（可選）</Label>
                    <Input
                      value={draft.visual.base_model || ''}
                      onChange={(e) => updateVisual({ base_model: e.target.value || null })}
                      className="mt-1"
                      placeholder="例如：runwayml/stable-diffusion-v1-5"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>預設 LoRA</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {draft.visual.default_loras.length === 0 ? (
                    <p className="text-sm text-slate-400">尚未選擇任何 LoRA</p>
                  ) : (
                    <div className="space-y-2">
                      {draft.visual.default_loras.map((l) => (
                        <div
                          key={l.lora_id}
                          className="flex items-center justify-between gap-3 p-3 bg-slate-800/50 rounded-lg"
                        >
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{l.lora_id}</div>
                            <div className="text-xs text-slate-500">weight: {l.weight}</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Input
                              type="number"
                              min={0}
                              max={2}
                              step={0.05}
                              value={l.weight}
                              onChange={(e) => updateLoraWeight(l.lora_id, Number(e.target.value) || 0)}
                              className="w-24"
                            />
                            <Button variant="ghost" size="sm" onClick={() => toggleLora(l.lora_id)}>
                              移除
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="pt-2 border-t border-slate-700">
                    <div className="text-sm font-semibold text-slate-200 mb-2">可用 LoRA</div>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                      {(lorasData as any)?.loras?.map((lora: any) => {
                        const loraId = lora.lora_id || lora.name
                        const isSelected = draft.visual.default_loras.some((x) => x.lora_id === loraId)
                        return (
                          <button
                            key={loraId}
                            type="button"
                            onClick={() => toggleLora(loraId)}
                            className={`w-full text-left p-3 rounded-lg border transition-colors ${
                              isSelected
                                ? 'border-primary bg-primary/10'
                                : 'border-slate-700 bg-slate-900/40 hover:bg-slate-900/70'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="min-w-0">
                                <div className="text-sm font-medium truncate">{lora.name || loraId}</div>
                                <div className="text-xs text-slate-500 truncate">{loraId}</div>
                              </div>
                              <div className="text-xs text-slate-400">{isSelected ? '已選' : '加入'}</div>
                            </div>
                          </button>
                        )
                      })}
                      {!((lorasData as any)?.loras?.length > 0) && (
                        <div className="text-sm text-slate-400">沒有找到 LoRA（可能尚未掛載模型資料夾）</div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="dataset" className="mt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Dataset Builder（圖片資料集）</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-xs text-slate-500">
                  目標：上傳 images.zip →（可選）auto-caption → 編修 caption → 產生 `metadata.jsonl` → 一鍵送訓練。
                </div>

                <div className="space-y-2">
                  <Label>資料集名稱（可選）</Label>
                  <Input value={datasetUploadName} onChange={(e) => setDatasetUploadName(e.target.value)} />
                </div>

                <div className="space-y-2">
                  <Label>上傳 images.zip</Label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setDatasetUploadFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-slate-200 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
                  />
                  <div className="text-xs text-slate-500">
                    world_id：<span className="font-mono">{selectedWorldId}</span>
                  </div>
                </div>

                <Button
                  variant="outline"
                  onClick={() => void handleUploadDatasetZip()}
                  disabled={datasetUploading || !datasetUploadFile}
                >
                  {datasetUploading ? '上傳中...' : '建立 Dataset'}
                </Button>

                <div className="pt-3 border-t border-slate-700 space-y-2">
                  <Label>已有 Datasets</Label>
                  <select
                    value={datasetSelectedId}
                    onChange={(e) => setDatasetSelectedId(e.target.value)}
                    className="w-full mt-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-200"
                  >
                    <option value="">（選擇 dataset）</option>
                    {datasetList.map((d) => (
                      <option key={d.dataset_id} value={d.dataset_id}>
                        {d.name || d.dataset_id} ({d.total_images || 0})
                      </option>
                    ))}
                  </select>
                  <div className="text-xs text-slate-500">
                    提示：目前只顯示前 80 張做編修（避免頁面太重）。
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Dataset 操作</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {!datasetDetail ? (
                  <div className="text-sm text-slate-400">請先選擇一個 dataset</div>
                ) : (
                  <>
                    <div className="space-y-1">
                      <div className="text-xs text-slate-500">dataset_path（訓練用）</div>
                      <div className="text-xs font-mono text-slate-200 break-all">{datasetDetail.dataset_path}</div>
                      <div className="text-xs text-slate-500">
                        images={datasetDetail.total_images} • metadata.jsonl={String(datasetDetail.has_metadata_jsonl)}
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void handleAutoCaptionDataset()}
                        disabled={!datasetSelectedId}
                      >
                        auto-caption（VLM）
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void handleBuildDatasetMetadata()}
                        disabled={datasetBuilding || !datasetSelectedId}
                      >
                        {datasetBuilding ? '產生中...' : '產生 metadata.jsonl'}
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        onClick={() => void handleTrainFromDataset()}
                        disabled={trainingSubmitting || !datasetSelectedId}
                      >
                        一鍵送訓練（SDXL LoRA）
                      </Button>
                    </div>

                    {datasetCaptionJobId && (
                      <JobProgressCard
                        title="auto-caption job"
                        jobId={datasetCaptionJobId}
                        job={datasetCaptionJobState.job}
                        isLoading={datasetCaptionJobState.isLoading}
                        error={datasetCaptionJobState.error}
                        cancelling={datasetCaptionJobState.cancelJob.isPending}
                        onCancel={() => void datasetCaptionJobState.cancelJob.mutate()}
                      />
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {datasetDetail?.items?.length ? (
            <Card>
              <CardHeader>
                <CardTitle>Caption 編修（前 80 張）</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {datasetDetail.items.map((item: any) => {
                  const itemId = String(item.item_id || item.filename || '')
                  const caption = captionEdits[itemId] ?? String(item.caption || '')
                  return (
                    <div key={itemId} className="rounded-lg border border-slate-700 bg-slate-900/40 overflow-hidden">
                      <div className="aspect-square bg-slate-800">
                        {item.image_url ? (
                          <img src={item.image_url} className="w-full h-full object-cover" alt={item.filename} />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs text-slate-500">
                            無預覽
                          </div>
                        )}
                      </div>
                      <div className="p-3 space-y-2">
                        <div className="text-xs text-slate-400 font-mono truncate">{item.filename}</div>
                        <Textarea
                          value={caption}
                          onChange={(e) => setCaptionEdits((prev) => ({ ...prev, [itemId]: e.target.value }))}
                          rows={3}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => void handleSaveCaption(itemId)}
                          disabled={captionSaving === itemId}
                        >
                          {captionSaving === itemId ? '保存中...' : '保存'}
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </CardContent>
            </Card>
          ) : null}
        </TabsContent>

        <TabsContent value="training" className="mt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>LoRA / QLoRA 訓練</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-xs text-slate-500">
                  建議：資料集放 `/mnt/c/ai_datasets/...`、模型放 `/mnt/c/ai_models/...`。
                </div>

                <div>
                  <Label>訓練類型</Label>
                  <select
                    className="mt-1 w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-sm"
                    value={trainingKind}
                    onChange={(e) => setTrainingKind(e.target.value as any)}
                  >
                    <option value="lora_sdxl">SDXL LoRA（場景/風格）</option>
                    <option value="llm_lora">LLM LoRA（Qwen / 對話/規則）</option>
                  </select>
                </div>

                <div>
                  <Label>base_model</Label>
                  <Input
                    value={trainingBaseModel}
                    onChange={(e) => setTrainingBaseModel(e.target.value)}
                    className="mt-1"
                    placeholder={
                      trainingKind === 'lora_sdxl'
                        ? 'stabilityai/stable-diffusion-xl-base-1.0 或本地資料夾'
                        : 'Qwen/Qwen2.5-7B-Instruct 或本地資料夾'
                    }
                  />
                </div>

                <div>
                  <Label>dataset_path</Label>
                  <Input
                    value={trainingDatasetPath}
                    onChange={(e) => setTrainingDatasetPath(e.target.value)}
                    className="mt-1"
                    placeholder={
                      trainingKind === 'lora_sdxl'
                        ? '/mnt/c/ai_datasets/...（資料夾或 metadata.jsonl）'
                        : '/mnt/c/ai_datasets/.../train.jsonl'
                    }
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>output_name</Label>
                    <Input
                      value={trainingOutputName}
                      onChange={(e) => setTrainingOutputName(e.target.value)}
                      className="mt-1"
                      placeholder="例如：npc_miko_style_v1"
                    />
                  </div>
                  <div>
                    <Label>tags（逗號分隔）</Label>
                    <Input
                      value={trainingTags}
                      onChange={(e) => setTrainingTags(e.target.value)}
                      className="mt-1"
                      placeholder="anime, sdxl, world_id:xxx"
                    />
                  </div>
                </div>

                <div className="flex items-start justify-between gap-3 p-3 rounded-lg border border-slate-700 bg-slate-900/40">
                  <div>
                    <div className="text-sm font-semibold text-slate-200">simulate（僅模擬，不跑真訓練）</div>
                    <div className="text-xs text-slate-500 mt-1">
                      開發/無 GPU 時用；真實環境請關閉 simulate。
                    </div>
                  </div>
                  <div className="pt-1">
                    <Checkbox checked={trainingSimulate} onCheckedChange={(v) => setTrainingSimulate(Boolean(v))} />
                  </div>
                </div>

                {trainingKind === 'lora_sdxl' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>resolution</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.resolution}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, resolution: Number(e.target.value) || 1024 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>batch_size</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.batch_size}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, batch_size: Number(e.target.value) || 1 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>gradient_accumulation_steps</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.gradient_accumulation_steps}
                        onChange={(e) =>
                          setSdxlTrainConfig((p) => ({ ...p, gradient_accumulation_steps: Number(e.target.value) || 4 }))
                        }
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>max_steps</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.max_steps}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, max_steps: Number(e.target.value) || 1000 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>learning_rate</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.learning_rate}
                        onChange={(e) =>
                          setSdxlTrainConfig((p) => ({ ...p, learning_rate: Number(e.target.value) || 1e-4 }))
                        }
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>lora_rank</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.lora_rank}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, lora_rank: Number(e.target.value) || 16 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>seed</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.seed}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, seed: Number(e.target.value) || 42 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>save_steps</Label>
                      <Input
                        type="number"
                        value={sdxlTrainConfig.save_steps}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, save_steps: Number(e.target.value) || 500 }))}
                        className="mt-1"
                      />
                    </div>
                    <div className="md:col-span-2">
                      <Label>mixed_precision</Label>
                      <Input
                        value={sdxlTrainConfig.mixed_precision}
                        onChange={(e) => setSdxlTrainConfig((p) => ({ ...p, mixed_precision: e.target.value }))}
                        className="mt-1"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>max_length</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.max_length}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, max_length: Number(e.target.value) || 2048 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>batch_size</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.batch_size}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, batch_size: Number(e.target.value) || 1 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>gradient_accumulation_steps</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.gradient_accumulation_steps}
                        onChange={(e) =>
                          setLlmTrainConfig((p) => ({ ...p, gradient_accumulation_steps: Number(e.target.value) || 8 }))
                        }
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>max_steps</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.max_steps}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, max_steps: Number(e.target.value) || 500 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>warmup_steps</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.warmup_steps}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, warmup_steps: Number(e.target.value) || 50 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>lr_scheduler</Label>
                      <Input
                        value={llmTrainConfig.lr_scheduler}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, lr_scheduler: e.target.value }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>learning_rate</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.learning_rate}
                        onChange={(e) =>
                          setLlmTrainConfig((p) => ({ ...p, learning_rate: Number(e.target.value) || 2e-4 }))
                        }
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>lora_rank</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.lora_rank}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, lora_rank: Number(e.target.value) || 16 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>lora_alpha</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.lora_alpha}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, lora_alpha: Number(e.target.value) || 32 }))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>lora_dropout</Label>
                      <Input
                        type="number"
                        value={llmTrainConfig.lora_dropout}
                        onChange={(e) =>
                          setLlmTrainConfig((p) => ({ ...p, lora_dropout: Number(e.target.value) || 0.05 }))
                        }
                        className="mt-1"
                      />
                    </div>
                    <div className="md:col-span-2 flex items-start justify-between gap-3 p-3 rounded-lg border border-slate-700 bg-slate-900/40">
                      <div>
                        <div className="text-sm font-semibold text-slate-200">use_4bit（QLoRA）</div>
                        <div className="text-xs text-slate-500 mt-1">16GB 單卡建議開啟</div>
                      </div>
                      <div className="pt-1">
                        <Checkbox checked={llmTrainConfig.use_4bit} onCheckedChange={(v) => setLlmTrainConfig((p) => ({ ...p, use_4bit: Boolean(v) }))} />
                      </div>
                    </div>
                    <div className="md:col-span-2">
                      <Label>target_modules（逗號分隔）</Label>
                      <Input
                        value={llmTrainConfig.target_modules_csv}
                        onChange={(e) => setLlmTrainConfig((p) => ({ ...p, target_modules_csv: e.target.value }))}
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}

                <Button onClick={handleSubmitTraining} disabled={trainingSubmitting} className="w-full">
                  {trainingSubmitting ? '提交中...' : '建立訓練任務'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>訓練狀態</CardTitle>
              </CardHeader>
	              <CardContent className="space-y-3">
	                {!trainingJobId ? (
	                  <div className="text-sm text-slate-400">尚未提交任何訓練任務</div>
	                ) : (
	                  <>
	                    <JobProgressCard
	                      title="訓練任務"
	                      jobId={trainingJobId}
	                      job={trainingJobState.job}
	                      isLoading={trainingJobState.isLoading}
	                      error={trainingJobState.error}
	                      cancelling={trainingJobState.cancelJob.isPending}
	                      onCancel={() => void trainingJobState.cancelJob.mutate()}
	                    />

	                    {((trainingJobState.job as any)?.current_loss ?? null) !== null && (
	                      <div className="flex items-center justify-between gap-2">
	                        <div className="text-xs text-slate-500">loss</div>
	                        <div className="text-xs text-slate-300">{Number((trainingJobState.job as any)?.current_loss).toFixed(4)}</div>
	                      </div>
	                    )}

	                    {trainingJobState.job?.result_path && (
	                      <div className="text-xs text-slate-500 break-all">
	                        result: <span className="text-slate-300">{String(trainingJobState.job.result_path)}</span>
	                      </div>
	                    )}

	                    {draft &&
	                      String((trainingJobState.job as any)?.status || '') === 'completed' &&
	                      String((trainingJobState.job as any)?.job_type || (trainingJobState.job as any)?.payload?.job_type || '') === 'lora_sdxl' && (
	                        <div className="pt-3 border-t border-slate-700 space-y-2">
	                          <div className="text-xs text-slate-500">訓練產物 → 世界預設 LoRA</div>
	                          <div className="text-xs text-slate-300">
	                            output_name: <span className="font-mono">{getTrainingOutputName() || '—'}</span>
	                          </div>
	                          {Boolean((trainingJobState.job as any)?.payload?.simulate) && (
	                            <div className="text-xs text-amber-300">
	                              注意：此 job 為 simulate 模式，可能沒有真實 LoRA 產物（請在 GPU 環境關掉 simulate）。
	                            </div>
	                          )}
                          <div className="flex flex-wrap gap-2">
                            <Button
                              size="sm"
                              variant="default"
                              onClick={() => void handleAttachTrainingLoRA()}
                              disabled={!getTrainingOutputName()}
                            >
                              加入此世界 default_loras
                            </Button>
                            {sessionId && worldId && selectedWorldId === worldId && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => void handleAttachTrainingLoRA({ syncToStory: true })}
                                disabled={!getTrainingOutputName()}
                              >
                                保存 + 套用到本故事
                              </Button>
                            )}
                          </div>
                        </div>
                      )}

	                    <div className="flex gap-2">
	                      <Button
	                        size="sm"
	                        variant="outline"
	                        onClick={() => {
	                          setTrainingJobId(null)
	                        }}
	                      >
	                        清除
	                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => queryClient.invalidateQueries({ queryKey: [CACHE_KEYS.t2i.loras()] })}
                      >
                        刷新 LoRA 清單
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="story_agents" className="mt-6 space-y-6">
          {!draft ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-slate-400">載入世界資料中...</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Story Orchestrator（agent_profile）</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-200">啟用多代理 Story Director</div>
                      <div className="text-xs text-slate-500 mt-1">
                        影響每回合的「任務/關係/道具/節奏/連貫性/RAG」決策；保存後可「套用到本故事」立即生效。
                      </div>
                    </div>
                    <div className="pt-1">
                      <Checkbox
                        checked={draft.agent_profile?.enabled ?? true}
                        onCheckedChange={(checked) => updateAgentProfile({ enabled: checked })}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-sm">max_tool_calls（每回合工具呼叫上限）</Label>
                      <Input
                        type="number"
                        min={0}
                        max={20}
                        value={draft.agent_profile?.max_tool_calls ?? 6}
                        onChange={(e) =>
                          updateAgentProfile({
                            max_tool_calls: Math.max(0, Math.min(20, Number(e.target.value) || 0)),
                          })
                        }
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-sm">max_llm_calls（LLM Director 上限）</Label>
                      <Input
                        type="number"
                        min={0}
                        max={10}
                        value={draft.agent_profile?.max_llm_calls ?? 1}
                        onChange={(e) =>
                          updateAgentProfile({
                            max_llm_calls: Math.max(0, Math.min(10, Number(e.target.value) || 0)),
                          })
                        }
                        className="mt-1"
                      />
                    </div>
                  </div>

                  <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40 space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <div className="text-sm font-semibold text-slate-200">子代理控制</div>
                        <div className="text-xs text-slate-500">提示：空清單 = 全部子代理（預設）</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Checkbox
                          checked={(draft.agent_profile?.enabled_agents?.length ?? 0) > 0}
                          onCheckedChange={(checked) => {
                            if (!checked) {
                              updateAgentProfile({ enabled_agents: [] })
                              return
                            }
                            updateAgentProfile({ enabled_agents: storySubAgents.map((a) => a.id) })
                          }}
                        />
                        <span className="text-xs text-slate-300">限制為勾選清單</span>
                      </div>
                    </div>

                    {(draft.agent_profile?.enabled_agents?.length ?? 0) > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {storySubAgents.map((a) => {
                          const checked = (draft.agent_profile?.enabled_agents || []).includes(a.id)
                          return (
                            <div key={a.id} className="p-2 rounded border border-slate-700 bg-slate-800/40">
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                  <div className="text-sm font-medium text-slate-200 truncate">{a.name}</div>
                                  <div className="text-xs text-slate-500 mt-1">{a.description}</div>
                                  <div className="text-[11px] text-slate-600 mt-1">{a.id}</div>
                                </div>
                                <Checkbox checked={checked} onCheckedChange={() => toggleAgentEnabled(a.id)} />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <div className="text-xs text-slate-400">目前：全部子代理（預設）</div>
                    )}
                  </div>

                  <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40 space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <div className="text-sm font-semibold text-slate-200">工具白名單</div>
                        <div className="text-xs text-slate-500">提示：空清單 = 使用預設工具集合</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Checkbox
                          checked={(draft.agent_profile?.allowed_tools?.length ?? 0) > 0}
                          onCheckedChange={(checked) => {
                            if (!checked) {
                              updateAgentProfile({ allowed_tools: [] })
                              return
                            }
                            updateAgentProfile({ allowed_tools: storyAllowedTools.map((t) => t.id) })
                          }}
                        />
                        <span className="text-xs text-slate-300">限制工具</span>
                      </div>
                    </div>

                    {(draft.agent_profile?.allowed_tools?.length ?? 0) > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {storyAllowedTools.map((t) => {
                          const checked = (draft.agent_profile?.allowed_tools || []).includes(t.id)
                          return (
                            <div key={t.id} className="p-2 rounded border border-slate-700 bg-slate-800/40">
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                  <div className="text-sm font-medium text-slate-200 truncate">{t.id}</div>
                                  <div className="text-xs text-slate-500 mt-1">{t.description}</div>
                                </div>
                                <Checkbox checked={checked} onCheckedChange={() => toggleAllowedTool(t.id)} />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <div className="text-xs text-slate-400">目前：使用預設工具集合</div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>目前設定快照</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-xs text-slate-500">（保存後會寫入 worldpack；若在故事中，記得「套用到本故事」）</div>
                  <Textarea
                    value={JSON.stringify(draft.agent_profile || defaultAgentProfile, null, 2)}
                    readOnly
                    rows={18}
                    className="font-mono text-xs"
                  />
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader className="pb-3">
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                    <CardTitle>世界回寫（從本故事）</CardTitle>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void handleWritebackSuggest()}
                        disabled={!sessionId || !worldId || selectedWorldId !== worldId || worldWritebackSuggest.isPending}
                      >
                        {worldWritebackSuggest.isPending ? '匯出中...' : '匯出預覽'}
                      </Button>
                      {writebackPreview && (
                        <>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={applyWritebackSelectionToDraft}
                            disabled={!writebackPreview || !writebackDiff}
                          >
                            套用選取到草稿
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => void saveWritebackSelection(false)}
                            disabled={!writebackPreview || !writebackDiff || updateWorld.isPending}
                          >
                            {updateWorld.isPending ? '保存中...' : '套用並保存'}
                          </Button>
	                          {sessionId && (
	                            <Button
	                              variant="outline"
	                              size="sm"
	                              onClick={() => void saveWritebackSelection(true)}
	                              disabled={!writebackPreview || !writebackDiff || updateWorld.isPending || syncWorldpack.isPending}
	                            >
	                              {updateWorld.isPending || syncWorldpack.isPending ? '處理中...' : '保存+套用'}
	                            </Button>
	                          )}
	                          {sessionId && (
	                            <Button variant="outline" size="sm" onClick={enqueueWritebackPreview} disabled={!writebackPreview}>
	                              加入審核佇列
	                            </Button>
	                          )}
	                          <Button variant="ghost" size="sm" onClick={() => setWritebackPreview(null)}>
	                            清除
	                          </Button>
	                        </>
	                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {!sessionId ? (
                    <div className="text-sm text-slate-400">需要在「Story 工作台」內使用（才有 session_id）</div>
                  ) : worldId && selectedWorldId !== worldId ? (
                    <div className="text-sm text-amber-300">
                      目前故事 world_id={worldId}，你正在編輯 {selectedWorldId}；此故事不能直接切換 world（請新建 Story）。
                    </div>
                  ) : (
                    <>
                      <div className="text-xs text-slate-500">
                        從本故事 session 狀態匯出：新出現角色、任務/地點/NPC/物品/事件/成就 flags。會先產生 patch 預覽，需你確認後再保存/套用。
                      </div>

                      {!writebackPreview ? (
                        <div className="text-sm text-slate-400">尚未匯出任何回寫預覽</div>
                      ) : (
                        <>
                          {writebackDiff && (
                            <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40 space-y-3">
                              <div className="flex items-center justify-between">
                                <div className="text-sm font-semibold text-slate-200">分區差異（可勾選）</div>
                                <div className="text-xs text-slate-500">以「目前草稿」對比匯出候選 worldpack</div>
                              </div>

                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                <div
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setWritebackSelection((p) => ({ ...p, world_flags: !p.world_flags }))}
                                  onKeyDown={(e) =>
                                    e.key === 'Enter' && setWritebackSelection((p) => ({ ...p, world_flags: !p.world_flags }))
                                  }
                                  className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                                >
                                  <div className="flex items-center gap-2">
                                    <Checkbox
                                      checked={writebackSelection.world_flags}
                                      onCheckedChange={(checked) => setWritebackSelection((p) => ({ ...p, world_flags: checked }))}
                                      onClick={(e) => e.stopPropagation()}
                                    />
                                    <span className="text-sm text-slate-200">world_flags</span>
                                  </div>
                                  <span className="text-xs text-slate-400">{writebackDiff.worldFlagsChanged.length} 項</span>
                                </div>

                                <div
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => setWritebackSelection((p) => ({ ...p, characters: !p.characters }))}
                                  onKeyDown={(e) =>
                                    e.key === 'Enter' && setWritebackSelection((p) => ({ ...p, characters: !p.characters }))
                                  }
                                  className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                                >
                                  <div className="flex items-center gap-2">
                                    <Checkbox
                                      checked={writebackSelection.characters}
                                      onCheckedChange={(checked) => setWritebackSelection((p) => ({ ...p, characters: checked }))}
                                      onClick={(e) => e.stopPropagation()}
                                    />
                                    <span className="text-sm text-slate-200">角色 / NPC</span>
                                  </div>
                                  <span className="text-xs text-slate-400">
                                    {writebackDiff.charactersAdded.length + writebackDiff.charactersUpdated.length} 項
                                  </span>
                                </div>
                              </div>

                              <div className="text-xs text-slate-500 space-y-1">
                                {writebackDiff.worldFlagsChanged.length > 0 && (
                                  <div>
                                    <span className="text-slate-500">flags：</span>
                                    {writebackDiff.worldFlagsChanged
                                      .slice(0, 6)
                                      .map((x) => x.key)
                                      .join('、')}
                                    {writebackDiff.worldFlagsChanged.length > 6 ? '…' : ''}
                                  </div>
                                )}
                                {writebackDiff.charactersAdded.length + writebackDiff.charactersUpdated.length > 0 && (
                                  <div>
                                    <span className="text-slate-500">角色：</span>
                                    {[
                                      ...writebackDiff.charactersAdded.slice(0, 3).map((x) => `+${x.name}`),
                                      ...writebackDiff.charactersUpdated.slice(0, 3).map((x) => `~${x.name}`),
                                    ].join('、')}
                                    {writebackDiff.charactersAdded.length + writebackDiff.charactersUpdated.length > 6 ? '…' : ''}
                                  </div>
                                )}
                              </div>

                              <div className="text-xs text-slate-500">
                                提示：先「套用選取到草稿」，再用上方「保存」或「保存+套用」串進 Story 主流程。
                              </div>
                            </div>
                          )}

                          <div
                            className={`p-3 rounded-lg border ${
                              writebackPreview.success
                                ? 'border-green-500/30 bg-green-500/5'
                                : 'border-amber-500/30 bg-amber-500/5'
                            }`}
                          >
                            <div className="text-sm text-slate-200">{writebackPreview.success ? '預覽產生成功' : '預覽部分失敗'}</div>
                            <div className="text-xs text-slate-400 mt-1">
                              flags新增 {writebackPreview.summary?.flags_added ?? 0} / 角色新增 {writebackPreview.summary?.characters_added ?? 0}
                            </div>
                          </div>

                          {writebackPreview.errors?.length > 0 && (
                            <div className="p-3 rounded-lg border border-amber-500/30 bg-amber-900/10">
                              <div className="text-xs font-semibold text-amber-300 mb-1">Errors</div>
                              <div className="text-xs text-amber-200 whitespace-pre-wrap">
                                {(writebackPreview.errors || []).slice(0, 5).join('\n')}
                              </div>
                            </div>
                          )}

                          {writebackPreview.rag_note && (
                            <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40 space-y-2">
                              <div className="flex items-center justify-between gap-2">
                                <div className="text-sm font-semibold text-slate-200">可寫回 RAG 的摘要（需你確認）</div>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => void writebackAddRagNote()}
                                  disabled={!writebackPreview.rag_note}
                                >
                                  寫入知識庫
                                </Button>
                              </div>
                              <Textarea value={writebackPreview.rag_note} readOnly rows={8} className="font-mono text-xs" />
                            </div>
                          )}
                        </>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="ai" className="mt-6 space-y-6">
          {!draft ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-slate-400">載入世界資料中...</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>World Studio AI（多代理）</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>AI 指令</Label>
                    <Textarea
                      value={aiInstruction}
                      onChange={(e) => setAiInstruction(e.target.value)}
                      placeholder="例如：根據世界知識庫，建立 3 個重要 NPC（含動機、說話風格、persona_prompt）與 1 個玩家模板；並給出視覺風格 LoRA 建議。"
                      rows={5}
                      className="mt-1"
                    />
                    <p className="text-xs text-slate-500 mt-1">
                      會參考 world_id={selectedWorldId} 的 RAG snippets、現有 worldpack 與可用 LoRA 清單。
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center gap-2 pt-2">
                      <Checkbox
                        checked={aiOptions.include_visual}
                        onCheckedChange={(checked) => setAiOptions((p) => ({ ...p, include_visual: checked }))}
                      />
                      <Label className="text-sm">同時建議視覺風格 / LoRA</Label>
                    </div>

                    <div>
                      <Label className="text-sm">RAG snippets</Label>
                      <Input
                        type="number"
                        min={0}
                        max={20}
                        value={aiOptions.rag_top_k}
                        onChange={(e) => setAiOptions((p) => ({ ...p, rag_top_k: Number(e.target.value) || 0 }))}
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-sm">新增角色上限</Label>
                      <Input
                        type="number"
                        min={0}
                        max={12}
                        value={aiOptions.max_new_characters}
                        onChange={(e) =>
                          setAiOptions((p) => ({ ...p, max_new_characters: Number(e.target.value) || 0 }))
                        }
                        className="mt-1"
                      />
                    </div>

                    <div>
                      <Label className="text-sm">新增玩家模板上限</Label>
                      <Input
                        type="number"
                        min={0}
                        max={6}
                        value={aiOptions.max_new_player_templates}
                        onChange={(e) =>
                          setAiOptions((p) => ({ ...p, max_new_player_templates: Number(e.target.value) || 0 }))
                        }
                        className="mt-1"
                      />
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-2 pt-2">
                    <Button
                      variant="outline"
                      onClick={() => handleAiSuggest(false)}
                      disabled={worldAgentSuggest.isPending}
                      className="flex-1"
                    >
                      {worldAgentSuggest.isPending ? '產生中...' : '產生預覽（不保存）'}
                    </Button>
                    <Button onClick={() => handleAiSuggest(true)} disabled={worldAgentSuggest.isPending} className="flex-1">
                      {worldAgentSuggest.isPending ? '套用中...' : '產生並保存'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle>結果 / 子代理</CardTitle>
                    <div className="flex items-center gap-2">
                      {aiPreview && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={applyAiSelectionToDraft}
                          disabled={!aiPreview || !aiDiff}
                        >
                          套用選取到草稿
                        </Button>
                      )}
                      {aiPreview && (
                        <Button
                          size="sm"
                          onClick={() => void saveAiSelection(false)}
                          disabled={!aiPreview || !aiDiff || updateWorld.isPending}
                        >
                          {updateWorld.isPending ? '保存中...' : '套用並保存'}
                        </Button>
                      )}
	                      {aiPreview && sessionId && (
	                        <Button
	                          variant="outline"
	                          size="sm"
	                          onClick={() => void saveAiSelection(true)}
	                          disabled={!aiPreview || !aiDiff || updateWorld.isPending || syncWorldpack.isPending}
	                        >
	                          {updateWorld.isPending || syncWorldpack.isPending ? '處理中...' : '保存+套用'}
	                        </Button>
	                      )}
	                      {aiPreview && sessionId && (
	                        <Button variant="outline" size="sm" onClick={enqueueAiPreview} disabled={!aiPreview}>
	                          加入審核佇列
	                        </Button>
	                      )}
	                      {aiPreview && (
	                        <Button variant="ghost" size="sm" onClick={() => setAiPreview(null)}>
	                          清除
	                        </Button>
	                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {!aiPreview ? (
                    <p className="text-sm text-slate-400">尚未產生任何 AI 建議</p>
                  ) : (
                    <>
                      {aiDiff && (
                        <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/40 space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-semibold text-slate-200">分區差異（可勾選）</div>
                            <div className="text-xs text-slate-500">
                              以「目前草稿」對比 AI 候選 worldpack
                            </div>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() => setAiSelection((p) => ({ ...p, world: !p.world }))}
                              onKeyDown={(e) => e.key === 'Enter' && setAiSelection((p) => ({ ...p, world: !p.world }))}
                              className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                            >
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  checked={aiSelection.world}
                                  onCheckedChange={(checked) => setAiSelection((p) => ({ ...p, world: checked }))}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="text-sm text-slate-200">世界設定</span>
                              </div>
                              <span className="text-xs text-slate-400">{aiDiff.worldFields.length} 項</span>
                            </div>

                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() => setAiSelection((p) => ({ ...p, world_flags: !p.world_flags }))}
                              onKeyDown={(e) =>
                                e.key === 'Enter' && setAiSelection((p) => ({ ...p, world_flags: !p.world_flags }))
                              }
                              className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                            >
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  checked={aiSelection.world_flags}
                                  onCheckedChange={(checked) => setAiSelection((p) => ({ ...p, world_flags: checked }))}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="text-sm text-slate-200">world_flags</span>
                              </div>
                              <span className="text-xs text-slate-400">{aiDiff.worldFlagsChanged.length} 項</span>
                            </div>

                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() => setAiSelection((p) => ({ ...p, player_templates: !p.player_templates }))}
                              onKeyDown={(e) =>
                                e.key === 'Enter' && setAiSelection((p) => ({ ...p, player_templates: !p.player_templates }))
                              }
                              className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                            >
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  checked={aiSelection.player_templates}
                                  onCheckedChange={(checked) => setAiSelection((p) => ({ ...p, player_templates: checked }))}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="text-sm text-slate-200">玩家模板</span>
                              </div>
                              <span className="text-xs text-slate-400">
                                +{aiDiff.playerTemplatesAdded.length} / ~{aiDiff.playerTemplatesUpdated.length}
                              </span>
                            </div>

                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() => setAiSelection((p) => ({ ...p, characters: !p.characters }))}
                              onKeyDown={(e) =>
                                e.key === 'Enter' && setAiSelection((p) => ({ ...p, characters: !p.characters }))
                              }
                              className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 cursor-pointer"
                            >
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  checked={aiSelection.characters}
                                  onCheckedChange={(checked) => setAiSelection((p) => ({ ...p, characters: checked }))}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="text-sm text-slate-200">角色 / NPC</span>
                              </div>
                              <span className="text-xs text-slate-400">
                                +{aiDiff.charactersAdded.length} / ~{aiDiff.charactersUpdated.length}
                              </span>
                            </div>

                            <div
                              role="button"
                              tabIndex={0}
                              onClick={() => setAiSelection((p) => ({ ...p, visual: !p.visual }))}
                              onKeyDown={(e) => e.key === 'Enter' && setAiSelection((p) => ({ ...p, visual: !p.visual }))}
                              className="flex items-center justify-between gap-2 p-2 rounded border border-slate-700 bg-slate-800/40 hover:bg-slate-800/60 sm:col-span-2 cursor-pointer"
                            >
                              <div className="flex items-center gap-2">
                                <Checkbox
                                  checked={aiSelection.visual}
                                  onCheckedChange={(checked) => setAiSelection((p) => ({ ...p, visual: checked }))}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="text-sm text-slate-200">視覺風格（LoRA / prompt）</span>
                              </div>
                              <span className="text-xs text-slate-400">{aiDiff.visualFields.length} 項</span>
                            </div>
                          </div>

                          <div className="text-xs text-slate-400 space-y-1">
                            {aiDiff.worldFields.length > 0 && (
                              <div>
                                <span className="text-slate-500">世界：</span>
                                {aiDiff.worldFields
                                  .slice(0, 3)
                                  .map((x) => x.field)
                                  .join('、')}
                                {aiDiff.worldFields.length > 3 ? '…' : ''}
                              </div>
                            )}
                            {aiDiff.worldFlagsChanged.length > 0 && (
                              <div>
                                <span className="text-slate-500">flags：</span>
                                {aiDiff.worldFlagsChanged
                                  .slice(0, 3)
                                  .map((x) => x.key)
                                  .join('、')}
                                {aiDiff.worldFlagsChanged.length > 3 ? '…' : ''}
                              </div>
                            )}
                            {(aiDiff.charactersAdded.length > 0 || aiDiff.charactersUpdated.length > 0) && (
                              <div>
                                <span className="text-slate-500">角色：</span>
                                {[
                                  ...aiDiff.charactersAdded.slice(0, 2).map((x) => `+${x.name}`),
                                  ...aiDiff.charactersUpdated.slice(0, 2).map((x) => `~${x.name}`),
                                ].join('、')}
                                {aiDiff.charactersAdded.length + aiDiff.charactersUpdated.length > 4 ? '…' : ''}
                              </div>
                            )}
                            {(aiDiff.playerTemplatesAdded.length > 0 || aiDiff.playerTemplatesUpdated.length > 0) && (
                              <div>
                                <span className="text-slate-500">玩家模板：</span>
                                {[
                                  ...aiDiff.playerTemplatesAdded.slice(0, 2).map((x) => `+${x.name}`),
                                  ...aiDiff.playerTemplatesUpdated.slice(0, 2).map((x) => `~${x.name}`),
                                ].join('、')}
                                {aiDiff.playerTemplatesAdded.length + aiDiff.playerTemplatesUpdated.length > 4 ? '…' : ''}
                              </div>
                            )}
                            {aiDiff.visualFields.length > 0 && (
                              <div>
                                <span className="text-slate-500">視覺：</span>
                                {aiDiff.visualFields
                                  .slice(0, 3)
                                  .map((x) => x.field)
                                  .join('、')}
                                {aiDiff.visualFields.length > 3 ? '…' : ''}
                              </div>
                            )}
                          </div>

                          <div className="text-xs text-slate-500">
                            提示：先「套用選取到草稿」，再用上方「保存」或「保存+套用」串進 Story 主流程。
                          </div>
                        </div>
                      )}

                      <div
                        className={`p-3 rounded-lg border ${
                          aiPreview.success
                            ? 'border-green-500/30 bg-green-500/5'
                            : 'border-red-500/30 bg-red-500/5'
                        }`}
                      >
                        <div className="text-sm text-slate-200">
                          {aiPreview.success ? '成功' : '失敗'} / {aiPreview.applied ? '已保存' : '未保存'}
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          world_id={aiPreview.world_id} / 子代理={(aiPreview.contributors || []).length}
                        </div>
                      </div>

                      {aiPreview.errors?.length > 0 && (
                        <div className="p-3 rounded-lg border border-red-500/30 bg-red-900/10">
                          <div className="text-xs font-semibold text-red-300 mb-1">Errors</div>
                          <div className="text-xs text-red-200 whitespace-pre-wrap">
                            {(aiPreview.errors || []).slice(0, 5).join('\n')}
                          </div>
                        </div>
                      )}

                      <div className="max-h-[420px] overflow-y-auto space-y-2">
                        {(aiPreview.contributors || []).map((c, idx) => (
                          <div key={idx} className="p-3 rounded-lg border border-slate-700 bg-slate-800/40">
                            <div className="text-sm font-semibold text-slate-200 truncate">{c.agent || 'agent'}</div>
                            {c.reasoning && (
                              <div className="text-xs text-slate-400 mt-1 whitespace-pre-wrap break-words">
                                {c.reasoning}
                              </div>
                            )}
                          </div>
                        ))}
                        {(aiPreview.contributors || []).length === 0 && (
                          <div className="text-sm text-slate-400">沒有子代理輸出（可能使用了 fallback）</div>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Create world dialog */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>新建世界</DialogTitle>
            <DialogDescription>建立一個新的 worldpack（world_id 需唯一）</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label>world_id</Label>
              <Input
                value={createForm.world_id}
                onChange={(e) => setCreateForm((p) => ({ ...p, world_id: e.target.value }))}
                placeholder="例如：my_world_01"
                className="mt-1"
              />
            </div>
            <div>
              <Label>名稱</Label>
              <Input
                value={createForm.name}
                onChange={(e) => setCreateForm((p) => ({ ...p, name: e.target.value }))}
                placeholder="例如：蒸氣朋克新都"
                className="mt-1"
              />
            </div>
            <div>
              <Label>描述</Label>
              <Textarea
                value={createForm.description}
                onChange={(e) => setCreateForm((p) => ({ ...p, description: e.target.value }))}
                className="mt-1"
                rows={3}
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label>setting</Label>
                <Input
                  value={createForm.setting}
                  onChange={(e) => setCreateForm((p) => ({ ...p, setting: e.target.value }))}
                  className="mt-1"
                />
              </div>
              <div>
                <Label>difficulty</Label>
                <Input
                  value={createForm.difficulty}
                  onChange={(e) => setCreateForm((p) => ({ ...p, difficulty: e.target.value }))}
                  className="mt-1"
                />
              </div>
            </div>

            <div className="flex gap-2 pt-2">
              <Button onClick={handleCreate} disabled={createWorld.isPending} className="flex-1">
                {createWorld.isPending ? '建立中...' : '建立'}
              </Button>
              <Button variant="outline" onClick={() => setCreateOpen(false)}>
                取消
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
