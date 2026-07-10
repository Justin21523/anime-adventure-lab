import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AuthGate } from './AuthGate'

const { v2GetMock } = vi.hoisted(() => ({ v2GetMock: vi.fn() }))

vi.mock('@/api/v2', () => ({
  v2Get: v2GetMock,
  v2Post: vi.fn(),
}))

function renderGate() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AuthGate><div>private workbench</div></AuthGate>
    </QueryClientProvider>
  )
}

describe('AuthGate', () => {
  afterEach(() => v2GetMock.mockReset())

  it('renders the workbench directly when auth is disabled', async () => {
    v2GetMock.mockResolvedValue({ auth_required: false })
    renderGate()
    expect(await screen.findByText('private workbench')).toBeInTheDocument()
  })

  it('shows the login form when private auth is required', async () => {
    v2GetMock.mockImplementation((path: string) => {
      if (path === '/system/capabilities') return Promise.resolve({ auth_required: true })
      return Promise.reject(new Error('unauthorized'))
    })
    renderGate()
    expect(await screen.findByRole('heading', { name: '登入 SagaForge' })).toBeInTheDocument()
    expect(screen.getByLabelText('密碼')).toHaveAttribute('type', 'password')
  })
})
