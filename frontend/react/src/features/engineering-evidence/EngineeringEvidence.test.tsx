import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { EngineeringEvidence } from './EngineeringEvidence'

const { v2GetMock } = vi.hoisted(() => ({ v2GetMock: vi.fn() }))
vi.mock('@/api/v2', () => ({ v2Get: v2GetMock }))

const job = {
  job_id: 'job-123', kind: 'story_turn', status: 'completed', progress: 100,
  attempt_count: 1, session_id: 'session-moon', turn_id: 'turn-1', execution_id: 'worker-1',
  request_id: 'request-1', dispatch_status: 'dispatched', lease_expires_at: null,
  started_at: '2026-07-10T00:00:01Z', finished_at: '2026-07-10T00:00:02Z', duration_ms: 1000,
  replayed: false, result: {}, error: null, created_at: '2026-07-10T00:00:00Z', updated_at: '2026-07-10T00:00:02Z',
}

function renderEvidence() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><EngineeringEvidence /></QueryClientProvider>)
}

describe('EngineeringEvidence', () => {
  afterEach(() => {
    cleanup()
    v2GetMock.mockReset()
  })

  it('renders persisted events in chronological API order', async () => {
    v2GetMock.mockImplementation((path: string) => {
      if (path.startsWith('/jobs?')) return Promise.resolve([job])
      if (path.endsWith('/events')) return Promise.resolve([
        { event_id: 'e1', job_id: 'job-123', event_type: 'queued', from_status: null, to_status: 'queued', progress: 0, attempt_count: 0, execution_id: null, request_id: 'request-1', actor: 'api', details: { kind: 'story_turn' }, occurred_at: '2026-07-10T00:00:00Z' },
        { event_id: 'e2', job_id: 'job-123', event_type: 'completed', from_status: 'running', to_status: 'completed', progress: 100, attempt_count: 1, execution_id: 'worker-1', request_id: 'request-1', actor: 'worker', details: {}, occurred_at: '2026-07-10T00:00:02Z' },
      ])
      return Promise.resolve({ status: 'healthy', migration_revision: '20260710_03', story_runtime: 'deterministic', rag_runtime: 'deterministic', worker_profile: 'core', api_version: 'v2', services: {}, checked_at: '2026-07-10T00:00:00Z' })
    })
    renderEvidence()
    fireEvent.click(await screen.findByRole('button', { name: /story_turn · completed/ }))
    const queued = await screen.findByText('queued')
    const completed = await screen.findByText('completed')
    expect(queued.compareDocumentPosition(completed) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText('20260710_03')).toBeInTheDocument()
  })

  it('filters jobs and exposes an honest empty state', async () => {
    v2GetMock.mockImplementation((path: string) => path.startsWith('/jobs?') ? Promise.resolve([job]) : Promise.resolve({ status: 'healthy', migration_revision: null, story_runtime: 'deterministic', rag_runtime: 'deterministic', worker_profile: 'core', api_version: 'v2', services: {}, checked_at: '2026-07-10T00:00:00Z' }))
    renderEvidence()
    await screen.findByText(/job-123/)
    fireEvent.change(screen.getByLabelText('Session filter'), { target: { value: 'missing' } })
    expect(screen.getByText('No jobs match these filters. Run the demo scenario to create evidence.')).toBeInTheDocument()
  })

  it('shows API failure states instead of fabricated evidence', async () => {
    v2GetMock.mockRejectedValue(new Error('offline'))
    renderEvidence()
    expect((await screen.findAllByRole('alert')).map((node) => node.textContent).join(' ')).toMatch(/unavailable/i)
  })
})
