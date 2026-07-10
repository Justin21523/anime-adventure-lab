import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  uploadV2Document,
  v2Get,
  v2Post,
  type V2Document,
  type V2Job,
  type V2ReviewApproval,
  type V2ReviewProposal,
  type V2SystemStatus,
  type V2World,
} from '@/api/v2'
import { AppError } from '@/lib/api-error'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'

export function CitationList({ citations }: { citations: Array<Record<string, unknown>> }) {
  if (!citations.length) return null
  return (
    <details className="mt-4 rounded-lg border border-cyan-500/20 bg-cyan-950/10 p-3">
      <summary className="cursor-pointer text-sm font-medium text-cyan-300">
        RAG 引用證據（{citations.length}）
      </summary>
      <div className="mt-3 space-y-3">
        {citations.map((citation) => (
          <article key={String(citation.chunk_id)} className="rounded-md border border-slate-800 p-3 text-xs">
            <div className="flex justify-between gap-3 text-slate-400">
              <span>{String(citation.filename)}</span>
              <span>score {Number(citation.score).toFixed(3)}</span>
            </div>
            <p className="mt-2 leading-5 text-slate-300">{String(citation.excerpt)}</p>
            <p className="mt-2 font-mono text-[10px] text-slate-600">chunk / {String(citation.chunk_id)}</p>
          </article>
        ))}
      </div>
    </details>
  )
}

export function JobInspector({ job }: { job: V2Job | undefined }) {
  const client = useQueryClient()
  const retry = useMutation({
    mutationFn: () => v2Post<V2Job>(`/jobs/${job?.job_id}/retry`),
    onSuccess: (updated) => client.setQueryData(['v2', 'job', updated.job_id], updated),
  })
  if (!job) return null
  const rows = [
    ['Job ID', job.job_id],
    ['Status', job.status],
    ['Dispatch', job.dispatch_status],
    ['Attempt', String(job.attempt_count)],
    ['Progress', `${job.progress}%`],
    ['Duration', job.duration_ms == null ? '—' : `${job.duration_ms} ms`],
    ['Request ID', job.request_id ?? '—'],
    ['Execution ID', job.execution_id ?? '—'],
    ['Lease', job.lease_expires_at ? new Date(job.lease_expires_at).toLocaleTimeString() : '—'],
  ]
  return (
    <Card className="border-violet-500/30">
      <CardHeader><CardTitle className="text-base">Durable Job Inspector</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-xs">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
            <span className="text-slate-500">{label}</span>
            <span className="break-all font-mono text-slate-300">{value}</span>
          </div>
        ))}
        {job.error && <ErrorNotice error={new Error(`${job.error.code}: ${job.error.message}`)} />}
        {job.status === 'failed' && (
          <Button size="sm" variant="outline" onClick={() => retry.mutate()} disabled={retry.isPending}>
            {retry.isPending ? '重新派送中…' : '安全重試工作'}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

export function KnowledgePanel({ worldId }: { worldId: string }) {
  const client = useQueryClient()
  const documents = useQuery({
    queryKey: ['v2', 'documents', worldId],
    queryFn: () => v2Get<V2Document[]>(`/worlds/${encodeURIComponent(worldId)}/documents`),
    refetchInterval: (query) =>
      (query.state.data as V2Document[] | undefined)?.some((item) => ['queued', 'indexing'].includes(item.status))
        ? 1000
        : false,
  })
  const upload = useMutation({
    mutationFn: (file: File) => uploadV2Document(worldId, file),
    onSuccess: () => client.invalidateQueries({ queryKey: ['v2', 'documents', worldId] }),
  })
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">世界知識 · MinIO → pgvector</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <Input
          aria-label="上傳世界觀文件"
          type="file"
          accept=".txt,.md,.json,text/plain,text/markdown,application/json"
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) upload.mutate(file)
          }}
        />
        {upload.isPending && <p role="status" className="text-xs text-cyan-400">上傳並建立 durable index job…</p>}
        {upload.isError && <ErrorNotice error={upload.error} />}
        {documents.isLoading && <p className="text-xs text-slate-500">讀取文件狀態…</p>}
        {documents.data?.length === 0 && <p className="text-xs text-slate-500">尚未上傳 lore。</p>}
        {documents.data?.map((document) => (
          <div key={document.document_id} className="rounded-md border border-slate-800 p-3 text-xs">
            <div className="flex justify-between gap-2"><span>{document.filename}</span><Status value={document.status} /></div>
            <p className="mt-1 text-slate-500">{Number(document.metadata.chunk_count ?? 0)} chunks · {String(document.checksum).slice(0, 10)}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

export function ReviewPanel({ worldId, sessionId }: { worldId: string; sessionId: string }) {
  const client = useQueryClient()
  const proposals = useQuery({
    queryKey: ['v2', 'proposals', worldId, sessionId],
    queryFn: () => v2Get<V2ReviewProposal[]>(`/review-proposals?world_id=${encodeURIComponent(worldId)}&session_id=${encodeURIComponent(sessionId)}`),
  })
  const world = useQuery({ queryKey: ['v2', 'world', worldId], queryFn: () => v2Get<V2World>(`/worlds/${worldId}`) })
  const approve = useMutation({
    mutationFn: (id: string) => v2Post<V2ReviewApproval>(`/review-proposals/${id}/approve`, undefined, { 'If-Match': `"${world.data?.version}"` }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ['v2', 'proposals'] })
      client.invalidateQueries({ queryKey: ['v2', 'world'] })
      client.invalidateQueries({ queryKey: ['v2', 'worlds'] })
    },
  })
  const reject = useMutation({
    mutationFn: (id: string) => v2Post<V2ReviewProposal>(`/review-proposals/${id}/reject`),
    onSuccess: () => client.invalidateQueries({ queryKey: ['v2', 'proposals'] }),
  })
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Human-in-the-loop Review</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        {proposals.data?.length === 0 && <p className="text-xs text-slate-500">完成故事回合後，AI world patch 會在此等待審核。</p>}
        {proposals.data?.map((proposal) => (
          <article key={proposal.proposal_id} className="rounded-md border border-slate-800 p-3 text-xs">
            <div className="flex justify-between"><Status value={proposal.status} /><span className="text-slate-600">world v{world.data?.version ?? '—'}</span></div>
            <p className="mt-2 text-slate-400">{proposal.reasoning || 'AI 建議更新世界設定'}</p>
            <pre className="mt-2 max-h-40 overflow-auto rounded bg-slate-950 p-2 text-[10px] text-cyan-200">{JSON.stringify(proposal.patch, null, 2)}</pre>
            {proposal.status === 'pending' && (
              <div className="mt-3 flex gap-2">
                <Button size="sm" onClick={() => approve.mutate(proposal.proposal_id)} disabled={!world.data || approve.isPending}>Approve</Button>
                <Button size="sm" variant="outline" onClick={() => reject.mutate(proposal.proposal_id)} disabled={reject.isPending}>Reject</Button>
              </div>
            )}
          </article>
        ))}
        {(approve.isError || reject.isError) && <ErrorNotice error={approve.error || reject.error} />}
      </CardContent>
    </Card>
  )
}

export function SystemPanel() {
  const status = useQuery({
    queryKey: ['v2', 'system-status'],
    queryFn: () => v2Get<V2SystemStatus>('/system/status'),
    refetchInterval: 10000,
  })
  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Runtime & Services</CardTitle></CardHeader>
      <CardContent className="space-y-2 text-xs">
        {status.isLoading && <p>檢查服務中…</p>}
        {status.isError && <ErrorNotice error={status.error} />}
        {status.data && (
          <>
            <div className="flex justify-between"><span>Overall</span><Status value={status.data.status} /></div>
            {Object.entries(status.data.services).map(([name, service]) => (
              <div key={name} className="flex justify-between gap-3"><span className="text-slate-400">{name}</span><span><Status value={service.status} /> {service.detail}</span></div>
            ))}
            <p className="border-t border-slate-800 pt-2 text-slate-500">Story {status.data.story_runtime} · RAG {status.data.rag_runtime} · Worker {status.data.worker_profile}</p>
            <p className="font-mono text-[10px] text-slate-600">migration {status.data.migration_revision ?? 'unknown'}</p>
          </>
        )}
      </CardContent>
    </Card>
  )
}

function Status({ value }: { value: string }) {
  const tone = ['healthy', 'ready', 'completed', 'approved'].includes(value)
    ? 'text-emerald-400'
    : ['failed', 'unavailable', 'rejected'].includes(value)
      ? 'text-red-400'
      : 'text-amber-400'
  return <span className={tone}>{value}</span>
}

export function ErrorNotice({ error }: { error: unknown }) {
  const appError = error instanceof AppError ? error : null
  return (
    <p role="alert" className="rounded border border-red-500/30 bg-red-950/20 p-2 text-xs text-red-300">
      {appError?.getUserMessage() || (error instanceof Error ? error.message : '操作失敗')}
      {appError?.requestId && <span className="mt-1 block font-mono text-[10px]">request {appError.requestId}</span>}
    </p>
  )
}
