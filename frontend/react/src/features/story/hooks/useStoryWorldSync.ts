import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { StoryWorldSyncRequest, StoryWorldSyncResponse } from '../types/story.types'

export function useStoryWorldSync() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ sessionId, request }: { sessionId: string; request: StoryWorldSyncRequest }) => {
      return apiPost<StoryWorldSyncResponse, StoryWorldSyncRequest>(`/story/session/${sessionId}/sync_worldpack`, request)
    },
    onSuccess: (_res, vars) => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.session(vars.sessionId) })
    },
  })
}

