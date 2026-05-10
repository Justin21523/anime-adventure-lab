import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { StoryContextSnapshot } from '../types/story.types'

export function useStoryContext(sessionId?: string) {
  return useQuery({
    queryKey: sessionId ? CACHE_KEYS.story.context(sessionId) : ['story', 'context', 'none'],
    queryFn: async () => {
      if (!sessionId) throw new Error('sessionId is required')
      return apiGet<StoryContextSnapshot>(`/story/session/${sessionId}/context`, {
        retry: { maxRetries: 0 },
      })
    },
    enabled: Boolean(sessionId),
    staleTime: 10_000,
  })
}

