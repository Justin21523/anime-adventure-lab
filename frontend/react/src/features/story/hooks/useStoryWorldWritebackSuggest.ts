import { useMutation } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import type { StoryWorldWritebackSuggestRequest, StoryWorldWritebackSuggestResponse } from '../types/story.types'

export function useStoryWorldWritebackSuggest() {
  return useMutation({
    mutationFn: async ({ sessionId, request }: { sessionId: string; request: StoryWorldWritebackSuggestRequest }) => {
      return apiPost<StoryWorldWritebackSuggestResponse, StoryWorldWritebackSuggestRequest>(
        `/story/session/${sessionId}/world/writeback/suggest`,
        request
      )
    },
  })
}

