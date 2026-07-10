import { type FormEvent, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { v2Get, v2Post, v2Put, type V2Job, type V2Session, type V2Turn, type V2World } from '@/api/v2'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { CitationList, ErrorNotice, JobInspector, KnowledgePanel, ReviewPanel, SystemPanel } from './StoryTechnicalPanels'
import { EngineeringEvidence } from '@/features/engineering-evidence/EngineeringEvidence'

export function StoryWorkbenchV2() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [view, setView] = useState<'story' | 'evidence'>('story')
  return (
    <div className="min-h-screen bg-background text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3 sm:px-6">
          <div className="h-9 w-1 rounded-full bg-gradient-to-b from-cyan-400 to-violet-500" />
          <div><h1 className="font-semibold tracking-tight">SagaForge Story Workbench</h1><p className="text-xs text-slate-500">RAG-grounded narrative · durable jobs · human review</p></div>
          <nav className="ml-auto flex gap-1" aria-label="主要檢視"><Button variant={view === 'story' ? 'secondary' : 'ghost'} size="sm" onClick={() => setView('story')}>Story</Button><Button variant={view === 'evidence' ? 'secondary' : 'ghost'} size="sm" onClick={() => setView('evidence')}>Engineering Evidence</Button></nav>
          {view === 'story' && selectedSessionId && <Button variant="ghost" size="sm" onClick={() => setSelectedSessionId(null)}>返回冒險列表</Button>}
        </div>
      </header>
      <main className="mx-auto max-w-7xl p-4 sm:p-6 lg:p-8">
        {view === 'evidence' ? <EngineeringEvidence /> : selectedSessionId ? <StorySession sessionId={selectedSessionId} /> : <StoryDashboard onSelectSession={setSelectedSessionId} />}
      </main>
    </div>
  )
}

function StoryDashboard({ onSelectSession }: { onSelectSession: (id: string) => void }) {
  const client = useQueryClient()
  const [playerName, setPlayerName] = useState('凜')
  const [worldId, setWorldId] = useState('')
  const [worldName, setWorldName] = useState('月蝕檔案館')
  const [newWorldId, setNewWorldId] = useState('moon-archive')
  const [search, setSearch] = useState('')
  const worlds = useQuery({ queryKey: ['v2', 'worlds'], queryFn: () => v2Get<V2World[]>('/worlds') })
  const sessions = useQuery({ queryKey: ['v2', 'sessions'], queryFn: () => v2Get<V2Session[]>('/story-sessions') })
  const selectedWorldId = worldId || worlds.data?.[0]?.world_id || ''
  const visibleWorlds = (worlds.data ?? []).filter((world) => `${world.name} ${world.world_id}`.toLowerCase().includes(search.toLowerCase()))
  const createWorld = useMutation({
    mutationFn: () => v2Post<V2World, { world_id: string; name: string; pack: Record<string, unknown> }>('/worlds', { world_id: newWorldId, name: worldName, pack: { setting: 'mystery fantasy', description: '一座收藏失落記憶的月下檔案館。' } }),
    onSuccess: (world) => { setWorldId(world.world_id); client.invalidateQueries({ queryKey: ['v2', 'worlds'] }) },
  })
  const createSession = useMutation({
    mutationFn: () => v2Post<V2Session, Record<string, string | null>>('/story-sessions', { world_id: selectedWorldId, player_name: playerName, persona_id: 'wise_sage', runtime_preset_id: null }),
    onSuccess: (session) => { client.invalidateQueries({ queryKey: ['v2', 'sessions'] }); onSelectSession(session.session_id) },
  })
  if (worlds.isLoading || sessions.isLoading) return <StateCard message="正在載入世界與冒險…" />
  if (worlds.isError || sessions.isError) return <StateCard message="載入失敗，請檢查 API 與資料庫。" error />
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <section className="space-y-4">
        <div><p className="text-sm font-medium text-cyan-400">Your adventures</p><h2 className="text-2xl font-semibold">故事工作區</h2><p className="mt-1 text-sm text-slate-400">選擇持久化故事，查看最後進度，或開始新的冒險。</p></div>
        {(sessions.data?.length ?? 0) === 0 ? <StateCard message="目前沒有故事。右側表單會建立第一個可持久化 session。" /> : (
          <div className="grid gap-4 sm:grid-cols-2">
            {sessions.data?.map((session) => (
              <button key={session.session_id} type="button" onClick={() => onSelectSession(session.session_id)} className="rounded-xl border border-slate-800 bg-slate-950/70 p-5 text-left transition hover:-translate-y-0.5 hover:border-cyan-500/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">
                <p className="font-medium">{session.player_name}</p><p className="mt-1 text-xs text-slate-500">world / {session.world_id}</p>
                {Boolean(session.state.last_narrative) && <p className="mt-3 line-clamp-2 text-xs leading-5 text-slate-400">{String(session.state.last_narrative)}</p>}
                <div className="mt-5 flex items-center justify-between text-sm"><span className="text-slate-400">{Number(session.state.turn_count ?? 0)} 回合 · {String(session.state.last_turn_status ?? 'ready')}</span><span className="text-cyan-400">繼續故事 →</span></div>
              </button>
            ))}
          </div>
        )}
      </section>
      <aside className="space-y-4">
        {(worlds.data?.length ?? 0) === 0 ? (
          <Card className="border-amber-500/30"><CardHeader><CardTitle className="text-lg">先建立世界</CardTitle></CardHeader><CardContent className="space-y-3"><Label htmlFor="world-id">World ID</Label><Input id="world-id" value={newWorldId} onChange={(event) => setNewWorldId(event.target.value)} /><Label htmlFor="world-name">世界名稱</Label><Input id="world-name" value={worldName} onChange={(event) => setWorldName(event.target.value)} /><Button className="w-full" onClick={() => createWorld.mutate()} disabled={createWorld.isPending}>{createWorld.isPending ? '建立中…' : '建立範例世界'}</Button>{createWorld.isError && <ErrorNotice error={createWorld.error} />}</CardContent></Card>
        ) : (
          <Card><CardHeader><CardTitle className="text-lg">World registry</CardTitle></CardHeader><CardContent className="space-y-3"><Input aria-label="搜尋世界" placeholder="搜尋世界…" value={search} onChange={(event) => setSearch(event.target.value)} />{visibleWorlds.map((world) => <WorldCard key={world.world_id} world={world} selected={selectedWorldId === world.world_id} onSelect={setWorldId} />)}</CardContent></Card>
        )}
        <Card><CardHeader><CardTitle className="text-lg">開始新冒險</CardTitle></CardHeader><CardContent><form className="space-y-4" onSubmit={(event: FormEvent) => { event.preventDefault(); createSession.mutate() }}><div className="space-y-2"><Label htmlFor="player-name">主角名稱</Label><Input id="player-name" value={playerName} onChange={(event) => setPlayerName(event.target.value)} /></div><div className="space-y-2"><Label htmlFor="world-select">世界</Label><select id="world-select" value={selectedWorldId} onChange={(event) => setWorldId(event.target.value)} className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm"><option value="">選擇世界</option>{worlds.data?.map((world) => <option key={world.world_id} value={world.world_id}>{world.name}</option>)}</select></div>{createSession.isError && <ErrorNotice error={createSession.error} />}<Button className="w-full" disabled={!selectedWorldId || !playerName.trim() || createSession.isPending}>{createSession.isPending ? '建立中…' : '進入故事'}</Button></form></CardContent></Card>
      </aside>
    </div>
  )
}

function StorySession({ sessionId }: { sessionId: string }) {
  const client = useQueryClient()
  const [input, setInput] = useState('調查月光下發亮的檔案盒')
  const [jobId, setJobId] = useState<string | null>(null)
  const [ragMode, setRagMode] = useState<'auto' | 'on' | 'off'>('auto')
  const sessions = useQuery({ queryKey: ['v2', 'sessions'], queryFn: () => v2Get<V2Session[]>('/story-sessions') })
  const session = useMemo(() => sessions.data?.find((item) => item.session_id === sessionId), [sessions.data, sessionId])
  const turns = useQuery({ queryKey: ['v2', 'turns', sessionId], queryFn: () => v2Get<V2Turn[]>(`/story-sessions/${encodeURIComponent(sessionId)}/turns`) })
  const job = useQuery({ queryKey: ['v2', 'job', jobId], queryFn: () => v2Get<V2Job>(`/jobs/${jobId}`), enabled: Boolean(jobId), refetchInterval: (query) => { const value = (query.state.data as V2Job | undefined)?.status; return value && ['completed', 'failed', 'cancelled'].includes(value) ? false : 1000 } })
  useEffect(() => { if (!job.data || !['completed', 'failed', 'cancelled'].includes(job.data.status)) return; client.invalidateQueries({ queryKey: ['v2', 'turns', sessionId] }); client.invalidateQueries({ queryKey: ['v2', 'sessions'] }); client.invalidateQueries({ queryKey: ['v2', 'proposals'] }) }, [job.data, client, sessionId])
  const active = Boolean(jobId && (!job.data || !['completed', 'failed', 'cancelled'].includes(job.data.status)))
  const submit = useMutation({
    mutationFn: () => v2Post<V2Job, Record<string, unknown>>(`/story-sessions/${encodeURIComponent(sessionId)}/turns`, { player_input: input, rag_mode: ragMode, include_image: false, use_agent: false }, { 'Idempotency-Key': crypto.randomUUID() }),
    onSuccess: (created) => { setInput(''); setJobId(created.job_id); client.invalidateQueries({ queryKey: ['v2', 'turns', sessionId] }) },
  })
  if (turns.isLoading || sessions.isLoading) return <StateCard message="正在讀取故事時間線…" />
  if (turns.isError || !session) return <StateCard message="無法讀取這個故事。" error />
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
      <section className="space-y-4">
        <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-950 to-slate-900 p-5 sm:p-7"><p className="text-xs uppercase tracking-[0.2em] text-violet-400">{session.world_id}</p><h2 className="mt-2 text-2xl font-semibold">{session.player_name} 的冒險</h2><p className="mt-2 text-sm text-slate-400">每一回合以 durable job 執行，並保存 RAG evidence 與 review proposal。</p></div>
        {(turns.data?.length ?? 0) === 0 && <StateCard message="故事尚未開始。輸入第一個行動來建立時間線。" />}
        <div className="space-y-3" aria-live="polite">{turns.data?.map((turn) => <article key={turn.turn_id} className="rounded-xl border border-slate-800 bg-slate-950/70 p-5"><div className="flex justify-between text-xs text-slate-500"><span>TURN {turn.turn_number}</span><span className={turn.status === 'failed' ? 'text-red-400' : 'text-cyan-400'}>{turn.status}</span></div><p className="mt-3 text-sm text-slate-400">你：{turn.player_input}</p>{turn.narrative && <p className="mt-3 leading-7">{turn.narrative}</p>}<CitationList citations={turn.citations as Array<Record<string, unknown>>} />{turn.choices.length > 0 && <div className="mt-4 flex flex-wrap gap-2">{turn.choices.map((choice, index) => <button type="button" key={String(choice.id ?? index)} onClick={() => setInput(String(choice.text ?? '繼續'))} className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-cyan-500">{String(choice.text ?? choice.id ?? '繼續')}</button>)}</div>}</article>)}</div>
        <form className="sticky bottom-3 rounded-xl border border-slate-700 bg-slate-950/95 p-3 shadow-2xl backdrop-blur" onSubmit={(event) => { event.preventDefault(); submit.mutate() }}><Label htmlFor="player-action" className="sr-only">你的行動</Label><Textarea id="player-action" rows={3} value={input} onChange={(event) => setInput(event.target.value)} placeholder="描述你的行動…" /><div className="mt-3 flex flex-wrap items-center justify-between gap-3"><span className="text-xs text-slate-500">{active ? `工作執行中 ${job.data?.progress ?? 0}%` : `RAG ${ragMode} · Idempotent turn`}</span><div className="flex gap-2"><label className="sr-only" htmlFor="rag-mode">RAG mode</label><select id="rag-mode" value={ragMode} onChange={(event) => setRagMode(event.target.value as typeof ragMode)} className="rounded border border-slate-700 bg-slate-900 px-2 text-xs"><option value="auto">RAG auto</option><option value="on">RAG on</option><option value="off">RAG off</option></select><Button disabled={!input.trim() || submit.isPending || active}>{active ? '生成中…' : '送出行動'}</Button></div></div>{submit.isError && <ErrorNotice error={submit.error} />}</form>
        <JobInspector job={job.data} />
      </section>
      <aside className="space-y-4"><Card><CardHeader><CardTitle className="text-base">Session state</CardTitle></CardHeader><CardContent className="space-y-2 text-sm"><StateRow label="版本" value={String(session.version)} /><StateRow label="回合" value={String(session.state.turn_count ?? 0)} /><StateRow label="Persona" value={session.persona_id ?? 'auto'} /></CardContent></Card><KnowledgePanel worldId={session.world_id} /><ReviewPanel worldId={session.world_id} sessionId={sessionId} /><SystemPanel /></aside>
    </div>
  )
}

function WorldCard({ world, selected, onSelect }: { world: V2World; selected: boolean; onSelect: (id: string) => void }) {
  const client = useQueryClient(); const [editing, setEditing] = useState(false); const [name, setName] = useState(world.name)
  const update = useMutation({ mutationFn: () => v2Put<V2World, { name: string; pack: Record<string, unknown> }>(`/worlds/${world.world_id}`, { name, pack: world.pack }, { 'If-Match': `"${world.version}"` }), onSuccess: () => { setEditing(false); client.invalidateQueries({ queryKey: ['v2', 'worlds'] }) } })
  return <div className={`rounded-lg border p-3 text-xs ${selected ? 'border-cyan-500/60' : 'border-slate-800'}`}>{editing ? <div className="space-y-2"><Input value={name} onChange={(event) => setName(event.target.value)} /><div className="flex gap-2"><Button size="sm" onClick={() => update.mutate()}>儲存</Button><Button size="sm" variant="ghost" onClick={() => setEditing(false)}>取消</Button></div>{update.isError && <ErrorNotice error={update.error} />}</div> : <><button className="w-full text-left" onClick={() => onSelect(world.world_id)}><span className="font-medium">{world.name}</span><span className="ml-2 text-slate-600">v{world.version}</span><span className="block font-mono text-[10px] text-slate-500">{world.world_id}</span></button><Button className="mt-2" size="sm" variant="ghost" onClick={() => setEditing(true)}>編輯名稱</Button></>}</div>
}

function StateRow({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-4"><span className="text-slate-500">{label}</span><span>{value}</span></div> }
function StateCard({ message, error = false }: { message: string; error?: boolean }) { return <div className={`rounded-xl border border-dashed p-8 text-center text-sm ${error ? 'border-red-500/40 text-red-400' : 'border-slate-700 text-slate-400'}`} role={error ? 'alert' : 'status'}>{message}</div> }
