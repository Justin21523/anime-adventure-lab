import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import { useSessionStore } from '@/stores/sessionStore'
import type {
  StorySessionDetail,
  StorySessionCreateRequest,
  StoryTurnRequest,
  StoryTurnJobResponse,
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
      const response = await apiGet<StorySessionDetail>(`/story/session/${sessionId}`)
      return response
    },
    enabled: !!sessionId,
    staleTime: 15_000,  // 15s — short enough for responsive turns
    gcTime: 5 * 60_000, // keep in cache for 5min while playing
    refetchOnWindowFocus: false, // avoid surprise refetch during active play
  })

  // Create new session
  const createSession = useMutation({
    mutationFn: async (request: StorySessionCreateRequest) => {
      const response = await apiPost<StoryTurnResponse>('/story/session', request)
      return response
    },
    onSuccess: (data) => {
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

  // Execute story turn as background job (GPU worker recommended)
  const executeTurnJob = useMutation({
    mutationFn: async (request: StoryTurnRequest) => {
      const response = await apiPost<StoryTurnJobResponse>('/story/turn_job', request)
      return response
    },
  })

  return {
    session,
    isLoading,
    error,
    refetch,
    createSession,
    executeTurn,
    executeTurnJob,
  }
}
