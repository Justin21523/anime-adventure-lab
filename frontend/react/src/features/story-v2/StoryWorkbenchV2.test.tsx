import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StoryWorkbenchV2 } from './StoryWorkbenchV2'

const { v2GetMock } = vi.hoisted(() => ({ v2GetMock: vi.fn() }))

vi.mock('@/api/v2', () => ({
  v2Get: v2GetMock,
  v2Post: vi.fn(),
  uploadV2Document: vi.fn(),
}))

function renderWorkbench() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <StoryWorkbenchV2 />
    </QueryClientProvider>
  )
}

describe('StoryWorkbenchV2', () => {
  afterEach(() => v2GetMock.mockReset())

  it('explains the empty state and asks for a world first', async () => {
    v2GetMock.mockResolvedValue([])
    renderWorkbench()
    expect(await screen.findByText('目前沒有故事。右側表單會建立第一個可持久化 session。')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '先建立世界' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '進入故事' })).toBeDisabled()
  })
})
