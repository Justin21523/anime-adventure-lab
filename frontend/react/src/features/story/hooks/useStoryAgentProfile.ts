import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPatch, apiPut } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type {
  StoryAgentProfilePatchRequest,
  StoryAgentProfileResponse,
  StoryAgentProfileUpdateRequest,
} from '../types/story.types'

export function useStoryAgentProfile(sessionId?: string) {
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: sessionId ? CACHE_KEYS.story.agentProfile(sessionId) : ['story', 'agent_profile', 'none'],
    queryFn: async () => {
      if (!sessionId) throw new Error('sessionId is required')
      return apiGet<StoryAgentProfileResponse>(`/story/session/${sessionId}/agent_profile`)
    },
    enabled: !!sessionId,
    staleTime: 30_000,
  })

  const patch = useMutation({
    mutationFn: async (patchRequest: StoryAgentProfilePatchRequest) => {
      if (!sessionId) throw new Error('sessionId is required')
      return apiPatch<StoryAgentProfileResponse, StoryAgentProfilePatchRequest>(
        `/story/session/${sessionId}/agent_profile`,
        patchRequest
      )
    },
    onSuccess: () => {
      if (!sessionId) return
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.agentProfile(sessionId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.session(sessionId) })
    },
  })

  const set = useMutation({
    mutationFn: async (request: StoryAgentProfileUpdateRequest) => {
      if (!sessionId) throw new Error('sessionId is required')
      return apiPut<StoryAgentProfileResponse, StoryAgentProfileUpdateRequest>(
        `/story/session/${sessionId}/agent_profile`,
        request
      )
    },
    onSuccess: () => {
      if (!sessionId) return
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.agentProfile(sessionId) })
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.session(sessionId) })
    },
  })

  return { ...query, patch, set }
}

