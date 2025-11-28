import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import { useSessionStore } from '@/stores/sessionStore'
import type {
  StorySession,
  StorySessionCreateRequest,
  StoryTurnRequest,
  StoryTurnResponse,
} from '../types/story.types'

/**
 * Hook for managing a single story session
 */
export function useStorySession(sessionId?: string) {
  const queryClient = useQueryClient()
  const { setCurrentSessionId, addRecentSession } = useSessionStore()

  // Fetch session details
  const {
    data: session,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: CACHE_KEYS.story.session(sessionId!),
    queryFn: async () => {
      const response = await apiGet<StorySession>(`/story/session/${sessionId}`)
      return response
    },
    enabled: !!sessionId,
    staleTime: 30_000, // 30 seconds
  })

  // Create new session
  const createSession = useMutation({
    mutationFn: async (request: StorySessionCreateRequest) => {
      const response = await apiPost<StorySession>('/story/session', request)
      return response
    },
    onSuccess: (data) => {
      // Update cache
      queryClient.setQueryData(CACHE_KEYS.story.session(data.session_id), data)
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.sessions() })

      // Update global state
      setCurrentSessionId(data.session_id)
      addRecentSession(data.session_id)
    },
  })

  // Execute story turn
  const executeTurn = useMutation({
    mutationFn: async (request: StoryTurnRequest) => {
      const response = await apiPost<StoryTurnResponse>('/story/turn', request)
      return response
    },
    onSuccess: (data) => {
      // Invalidate session query to refetch with new data
      if (sessionId) {
        queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.session(sessionId) })
      }
    },
  })

  // Delete session
  const deleteSession = useMutation({
    mutationFn: async (sessionId: string) => {
      await apiPost(`/story/session/${sessionId}/delete`)
    },
    onSuccess: () => {
      if (sessionId) {
        queryClient.removeQueries({ queryKey: CACHE_KEYS.story.session(sessionId) })
        queryClient.invalidateQueries({ queryKey: CACHE_KEYS.story.sessions() })
      }
    },
  })

  return {
    session,
    isLoading,
    error,
    refetch,
    createSession,
    executeTurn,
    deleteSession,
  }
}
