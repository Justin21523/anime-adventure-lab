import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { v2Get, type V2Job, type V2JobEvent, type V2SystemStatus } from '@/api/v2'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'

const terminalStatuses = new Set(['completed', 'failed', 'cancelled'])

export function EngineeringEvidence() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [status, setStatus] = useState('all')
  const [kind, setKind] = useState('all')
  const [session, setSession] = useState('')
  const jobs = useQuery({
    queryKey: ['v2', 'evidence', 'jobs'],
    queryFn: () => v2Get<V2Job[]>('/jobs?limit=50'),
    refetchInterval: (query) => (query.state.data as V2Job[] | undefined)?.some((job) => !terminalStatuses.has(job.status)) ? 1500 : false,
  })
  const system = useQuery({ queryKey: ['v2', 'system'], queryFn: () => v2Get<V2SystemStatus>('/system/status') })
  const selectedJob = jobs.data?.find((job) => job.job_id === selectedJobId) ?? null
  const events = useQuery({
    queryKey: ['v2', 'evidence', 'events', selectedJobId],
    queryFn: () => v2Get<V2JobEvent[]>(`/jobs/${encodeURIComponent(selectedJobId!)}/events`),
    enabled: Boolean(selectedJobId),
    refetchInterval: selectedJob && !terminalStatuses.has(selectedJob.status) ? 1000 : false,
  })
  const kinds = useMemo(() => [...new Set((jobs.data ?? []).map((job) => job.kind))].sort(), [jobs.data])
  const visible = (jobs.data ?? []).filter((job) =>
    (status === 'all' || job.status === status) &&
    (kind === 'all' || job.kind === kind) &&
    (!session || (job.session_id ?? '').toLowerCase().includes(session.toLowerCase()))
  )

  return <div className="space-y-6">
    <section className="rounded-2xl border border-cyan-500/20 bg-gradient-to-br from-slate-950 to-slate-900 p-5 sm:p-7">
      <p className="text-xs uppercase tracking-[0.2em] text-cyan-400">Engineering evidence</p>
      <h2 className="mt-2 text-2xl font-semibold">Durable AI workflow observability</h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">此畫面直接讀取持久化資料，不使用前端假資料。選擇工作即可檢查 API、worker、scheduler 與 admin 共同寫入的交易事件鏈。</p>
    </section>
    <div className="grid gap-4 md:grid-cols-3">
      <EvidenceFact title="Atomic lifecycle" body="Job 狀態與 audit event 在同一個資料庫 transaction 寫入。" source="core/application/job_events.py" />
      <EvidenceFact title="Delivery safety" body="Idempotency key、lease recovery 與 execution fencing 防止重複副作用。" source="core/application/story_service.py" />
      <EvidenceFact title="Review boundary" body="模型只能提出 world patch；版本檢查與人工核准後才套用。" source="docs/adr/0003-human-review-boundary.md" />
    </div>
    <SystemSummary data={system.data} loading={system.isLoading} failed={system.isError} />
    <div className="grid gap-6 xl:grid-cols-[minmax(22rem,0.8fr)_minmax(0,1.2fr)]">
      <Card><CardHeader><CardTitle>Persistent jobs</CardTitle></CardHeader><CardContent className="space-y-4">
        <div className="grid gap-2 sm:grid-cols-3">
          <select aria-label="工作狀態" value={status} onChange={(event) => setStatus(event.target.value)} className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm"><option value="all">All statuses</option>{['queued', 'running', 'completed', 'failed', 'cancelled'].map((value) => <option key={value}>{value}</option>)}</select>
          <select aria-label="工作類型" value={kind} onChange={(event) => setKind(event.target.value)} className="rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm"><option value="all">All kinds</option>{kinds.map((value) => <option key={value}>{value}</option>)}</select>
          <Input aria-label="Session filter" placeholder="Session ID" value={session} onChange={(event) => setSession(event.target.value)} />
        </div>
        {jobs.isLoading && <State message="Loading durable jobs…" />}
        {jobs.isError && <State message="Jobs API unavailable. Verify API and persistence services." error />}
        {!jobs.isLoading && !jobs.isError && visible.length === 0 && <State message="No jobs match these filters. Run the demo scenario to create evidence." />}
        <div className="space-y-2">{visible.map((job) => <Button key={job.job_id} variant="ghost" onClick={() => setSelectedJobId(job.job_id)} className={`h-auto w-full justify-start border p-3 text-left ${selectedJobId === job.job_id ? 'border-cyan-500/60 bg-cyan-500/5' : 'border-slate-800'}`}>
          <span className="min-w-0 flex-1"><span className="block font-medium">{job.kind} · {job.status}</span><span className="block truncate font-mono text-[10px] text-slate-500">{job.job_id}</span></span><span className="text-xs text-slate-400">{job.progress}% · attempt {job.attempt_count}</span>
        </Button>)}</div>
      </CardContent></Card>
      <Card><CardHeader><CardTitle>Lifecycle event timeline</CardTitle></CardHeader><CardContent>
        {!selectedJobId && <State message="Select a job to inspect its persisted event chain." />}
        {events.isLoading && <State message="Loading event chain…" />}
        {events.isError && <State message="Event API unavailable. The selected job may have been removed." error />}
        {selectedJobId && !events.isLoading && !events.isError && (events.data?.length ?? 0) === 0 && <State message="This job has no recorded events." />}
        <ol className="relative ml-2 border-l border-slate-700 pl-5">{events.data?.map((event) => <li key={event.event_id} className="relative pb-5 last:pb-0"><span className="absolute -left-[1.55rem] top-1 h-3 w-3 rounded-full border-2 border-slate-950 bg-cyan-400" /><div className="flex flex-wrap items-center gap-2"><strong className="text-sm">{event.event_type}</strong><span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] uppercase text-violet-300">{event.actor}</span><span className="text-xs text-slate-500">{event.from_status ?? '∅'} → {event.to_status ?? '∅'}</span></div><p className="mt-1 text-xs text-slate-400">progress {event.progress ?? 0}% · attempt {event.attempt_count} · {new Date(event.occurred_at).toLocaleString()}</p>{Object.keys(event.details).length > 0 && <pre className="mt-2 overflow-auto rounded bg-slate-950 p-2 text-[10px] text-slate-400">{JSON.stringify(event.details, null, 2)}</pre>}</li>)}</ol>
      </CardContent></Card>
    </div>
  </div>
}

function EvidenceFact({ title, body, source }: { title: string; body: string; source: string }) { return <Card><CardHeader><CardTitle className="text-base">{title}</CardTitle></CardHeader><CardContent><p className="text-sm leading-6 text-slate-400">{body}</p><code className="mt-3 block break-all text-[10px] text-cyan-500">{source}</code></CardContent></Card> }
function SystemSummary({ data, loading, failed }: { data?: V2SystemStatus; loading: boolean; failed: boolean }) { return <Card><CardHeader><CardTitle>Runtime status</CardTitle></CardHeader><CardContent>{loading ? <State message="Checking runtime services…" /> : failed || !data ? <State message="System status API unavailable." error /> : <div className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-5"><Metric label="Overall" value={data.status} /><Metric label="Migration" value={data.migration_revision ?? 'unknown'} /><Metric label="Story" value={data.story_runtime} /><Metric label="RAG" value={data.rag_runtime} /><Metric label="Worker" value={data.worker_profile} /></div>}</CardContent></Card> }
function Metric({ label, value }: { label: string; value: string }) { return <div className="rounded-lg border border-slate-800 p-3"><span className="block text-xs text-slate-500">{label}</span><span className="mt-1 block break-all font-mono text-xs">{value}</span></div> }
function State({ message, error = false }: { message: string; error?: boolean }) { return <div role={error ? 'alert' : 'status'} className={`rounded-lg border border-dashed p-5 text-center text-sm ${error ? 'border-red-500/40 text-red-400' : 'border-slate-700 text-slate-500'}`}>{message}</div> }
