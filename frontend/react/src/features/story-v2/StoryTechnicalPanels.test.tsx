import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CitationList, JobInspector, KnowledgePanel, SystemPanel } from './StoryTechnicalPanels'

const { v2GetMock } = vi.hoisted(() => ({ v2GetMock: vi.fn() }))
vi.mock('@/api/v2', () => ({
  v2Get: v2GetMock,
  v2Post: vi.fn(),
  uploadV2Document: vi.fn(),
}))

function renderWithQuery(node: React.ReactNode) {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {node}
    </QueryClientProvider>
  )
}

describe('Story technical evidence', () => {
  afterEach(() => v2GetMock.mockReset())

  it('renders source, score, and excerpt for RAG evidence', () => {
    renderWithQuery(
      <CitationList citations={[{ document_id: 'd1', filename: 'lore.md', chunk_id: 'c1', position: 0, excerpt: '月蝕鐘第三次響起', score: 0.9123 }]} />
    )
    expect(screen.getByText('RAG 引用證據（1）')).toBeInTheDocument()
    expect(screen.getByText('lore.md')).toBeInTheDocument()
    expect(screen.getByText('score 0.912')).toBeInTheDocument()
  })

  it('shows observable job identifiers and safe retry for failures', () => {
    renderWithQuery(
      <JobInspector job={{ job_id: 'job-1', kind: 'story_turn', status: 'failed', progress: 45, attempt_count: 2, session_id: 's1', turn_id: 't1', execution_id: 'exec-1', request_id: 'request-1', dispatch_status: 'dispatched', lease_expires_at: null, started_at: null, finished_at: null, duration_ms: null, replayed: false, result: null, error: { code: 'MODEL_DOWN', message: 'offline' }, created_at: '2026-07-10T00:00:00Z', updated_at: '2026-07-10T00:00:01Z' }} />
    )
    expect(screen.getByText('request-1')).toBeInTheDocument()
    expect(screen.getByText('exec-1')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '安全重試工作' })).toBeInTheDocument()
  })

  it('shows indexed documents and degraded service status', async () => {
    v2GetMock.mockImplementation((path: string) => {
      if (path.includes('/documents')) return Promise.resolve([{ document_id: 'd1', world_id: 'moon', filename: 'lore.md', content_type: 'text/markdown', checksum: 'abcdef012345', status: 'ready', metadata: { chunk_count: 3 }, created_at: '2026-07-10T00:00:00Z', updated_at: '2026-07-10T00:00:00Z' }])
      return Promise.resolve({ status: 'degraded', api_version: 'v2', migration_revision: '20260710_02', story_runtime: 'deterministic', rag_runtime: 'deterministic', worker_profile: 'core', checked_at: '2026-07-10T00:00:00Z', services: { worker: { status: 'unavailable', detail: 'no heartbeat' } } })
    })
    renderWithQuery(<><KnowledgePanel worldId="moon" /><SystemPanel /></>)
    expect(await screen.findByText('3 chunks · abcdef0123')).toBeInTheDocument()
    expect(await screen.findByText(/no heartbeat/)).toBeInTheDocument()
  })
})
