import { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useUiStore } from '@/stores/uiStore'
import { useAgentCatalog } from '@/features/agent/hooks/useAgentCatalog'
import { useStoryAgentProfile } from '../hooks/useStoryAgentProfile'
import type { AgentActions } from '../types/story.types'
import type { WorldAgentProfile } from '@/features/worlds/types/world.types'

function normalizeProfile(profile: WorldAgentProfile | null | undefined, defaults: WorldAgentProfile): WorldAgentProfile {
  if (!profile) return { ...defaults }
  return {
    enabled: typeof profile.enabled === 'boolean' ? profile.enabled : true,
    enabled_agents: Array.isArray(profile.enabled_agents) ? profile.enabled_agents : [],
    max_tool_calls: Number.isFinite(profile.max_tool_calls) ? profile.max_tool_calls : defaults.max_tool_calls,
    max_llm_calls: Number.isFinite(profile.max_llm_calls) ? profile.max_llm_calls : defaults.max_llm_calls,
    allowed_tools: Array.isArray(profile.allowed_tools) ? profile.allowed_tools : [],
  }
}

interface AgentProfilePanelProps {
  sessionId: string
  lastAgentActions?: AgentActions | null
}

export function AgentProfilePanel({ sessionId, lastAgentActions }: AgentProfilePanelProps) {
  const { addNotification } = useUiStore()
  const { data, isLoading, error, refetch, patch, set } = useStoryAgentProfile(sessionId)
  const { data: agentCatalog } = useAgentCatalog()

  const storyCatalog = agentCatalog?.story
  const storySubAgents = storyCatalog?.sub_agents || []
  const storyAllowedTools = storyCatalog?.allowed_tools || []
  const defaultProfile: WorldAgentProfile = useMemo(
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

  const serverProfile = useMemo(() => normalizeProfile(data?.agent_profile as any, defaultProfile), [data?.agent_profile, defaultProfile])
  const [draft, setDraft] = useState<WorldAgentProfile>(serverProfile)

  useEffect(() => {
    setDraft(serverProfile)
  }, [serverProfile.enabled, serverProfile.max_tool_calls, serverProfile.max_llm_calls, serverProfile.enabled_agents, serverProfile.allowed_tools])

  const allAgentIds = useMemo(() => storySubAgents.map((a) => a.id), [storySubAgents])
  const allToolIds = useMemo(() => storyAllowedTools.map((t) => t.id), [storyAllowedTools])

  const usesAllAgents = (draft.enabled_agents || []).length === 0
  const usesDefaultTools = (draft.allowed_tools || []).length === 0

  const enabledAgentsSet = useMemo(() => new Set((draft.enabled_agents || []).map((x) => String(x).trim()).filter(Boolean)), [draft.enabled_agents])
  const enabledToolsSet = useMemo(() => new Set((draft.allowed_tools || []).map((x) => String(x).trim()).filter(Boolean)), [draft.allowed_tools])

  const lastTurnAgents = useMemo(() => {
    const contributors = lastAgentActions?.contributors || []
    const out = new Set<string>()
    for (const c of contributors) {
      const id = String(c.agent || '').trim()
      if (!id) continue
      if (id === 'orchestrator') continue
      out.add(id)
    }
    return Array.from(out)
  }, [lastAgentActions?.contributors])

  const applyDraft = async () => {
    try {
      await set.mutateAsync({ agent_profile: draft })
      addNotification({ type: 'success', title: 'Agent Profile 已套用到本故事' })
    } catch (err) {
      addNotification({
        type: 'error',
        title: 'Agent Profile 套用失敗',
        message: err instanceof Error ? err.message : '未知錯誤',
      })
    }
  }

  if (isLoading) {
    return (
      <Card className="bg-slate-800/80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Agent Profile</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">載入中...</CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="bg-slate-800/80">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-lg">Agent Profile</CardTitle>
            <Button variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={() => void refetch()}>
              重試
            </Button>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">
          目前無法取得 agent_profile（API 可能尚未啟用）
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-slate-800/80">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-lg">Agent Profile（本故事）</CardTitle>
          <Badge variant={draft.enabled ? 'secondary' : 'outline'} className="text-xs">
            {draft.enabled ? 'enabled' : 'disabled'}
          </Badge>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="text-xs">
            tool_budget: {draft.max_tool_calls}
          </Badge>
          <Badge variant="outline" className="text-xs">
            llm_budget: {draft.max_llm_calls}
          </Badge>
          {lastTurnAgents.length > 0 && (
            <Badge variant="outline" className="text-xs">
              上回合介入：{lastTurnAgents.slice(0, 3).join(', ')}
              {lastTurnAgents.length > 3 ? '…' : ''}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <Switch checked={draft.enabled} onCheckedChange={(v) => setDraft((p) => ({ ...p, enabled: v }))} label="啟用多代理導演" />
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => setDraft(serverProfile)}
              disabled={patch.isPending || set.isPending}
            >
              還原
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => void applyDraft()}
              disabled={patch.isPending || set.isPending}
            >
              {set.isPending ? '套用中...' : '套用'}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label className="text-xs text-slate-400">Max tool calls</Label>
            <Input
              type="number"
              min={0}
              max={20}
              value={draft.max_tool_calls}
              onChange={(e) => setDraft((p) => ({ ...p, max_tool_calls: Number(e.target.value || 0) }))}
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs text-slate-400">Max LLM calls</Label>
            <Input
              type="number"
              min={0}
              max={10}
              value={draft.max_llm_calls}
              onChange={(e) => setDraft((p) => ({ ...p, max_llm_calls: Number(e.target.value || 0) }))}
            />
          </div>
        </div>

        <Tabs defaultValue="agents">
          <TabsList className="w-full">
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="tools">Tools</TabsTrigger>
          </TabsList>

          <TabsContent value="agents" className="mt-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs text-slate-500">
                enabled_agents 空 = 全部預設 agents
              </div>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setDraft((p) => ({ ...p, enabled_agents: [] }))}
              >
                使用全部
              </Button>
            </div>
            <div className="space-y-2">
              {storySubAgents.map((a) => {
                const checked = usesAllAgents || enabledAgentsSet.has(a.id)
                return (
                  <label key={a.id} className="flex items-start gap-2 rounded-md border border-slate-700 bg-slate-900/30 p-2 cursor-pointer">
                    <Checkbox
                      checked={checked}
                      onCheckedChange={(v) => {
                        const nextChecked = Boolean(v)
                        setDraft((prev) => {
                          const current = normalizeProfile(prev, defaultProfile)
                          const set = new Set<string>(
                            (current.enabled_agents || []).length === 0 ? allAgentIds : current.enabled_agents
                          )
                          if (nextChecked) set.add(a.id)
                          else set.delete(a.id)
                          return { ...current, enabled_agents: Array.from(set) }
                        })
                      }}
                    />
                    <div className="min-w-0">
                      <div className="text-sm text-slate-100 font-medium">{a.name}</div>
                      <div className="text-xs text-slate-500 break-words">{a.description}</div>
                    </div>
                  </label>
                )
              })}
            </div>
          </TabsContent>

          <TabsContent value="tools" className="mt-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs text-slate-500">
                allowed_tools 空 = 使用預設工具白名單
              </div>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setDraft((p) => ({ ...p, allowed_tools: [] }))}
              >
                使用預設
              </Button>
            </div>
            <div className="space-y-2">
              {storyAllowedTools.map((t) => {
                const checked = usesDefaultTools || enabledToolsSet.has(t.id)
                return (
                  <label key={t.id} className="flex items-start gap-2 rounded-md border border-slate-700 bg-slate-900/30 p-2 cursor-pointer">
                    <Checkbox
                      checked={checked}
                      onCheckedChange={(v) => {
                        const nextChecked = Boolean(v)
                        setDraft((prev) => {
                          const current = normalizeProfile(prev, defaultProfile)
                          const set = new Set<string>(
                            (current.allowed_tools || []).length === 0 ? allToolIds : current.allowed_tools
                          )
                          if (nextChecked) set.add(t.id)
                          else set.delete(t.id)
                          return { ...current, allowed_tools: Array.from(set) }
                        })
                      }}
                    />
                    <div className="min-w-0">
                      <div className="text-sm text-slate-100 font-medium">{t.id}</div>
                      <div className="text-xs text-slate-500 break-words">{t.description}</div>
                    </div>
                  </label>
                )
              })}
            </div>
          </TabsContent>
        </Tabs>

        <div className="text-xs text-slate-500">
          提示：套用後，下一回合開始會依此 profile 決定哪些 agents 介入與能用哪些 tools。
        </div>
      </CardContent>
    </Card>
  )
}
