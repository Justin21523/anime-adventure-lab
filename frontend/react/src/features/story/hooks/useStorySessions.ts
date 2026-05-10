import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { StorySessionInfo } from '../types/story.types'

/**
 * Hook for fetching all story sessions
 */
export function useStorySessions() {
  return useQuery({
    queryKey: CACHE_KEYS.story.sessions(),
    queryFn: async () => {
      const response = await apiGet<StorySessionInfo[]>('/story/sessions')
      return response
    },
    staleTime: 60_000, // 1 minute
  })
}
