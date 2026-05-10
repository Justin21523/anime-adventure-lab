import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { WorldAgentSuggestRequest, WorldAgentSuggestResponse } from '../types/world.types'

export function useWorldAgentSuggest() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ worldId, request }: { worldId: string; request: WorldAgentSuggestRequest }) => {
      return apiPost<WorldAgentSuggestResponse, WorldAgentSuggestRequest>(`/worlds/${worldId}/agents/suggest`, request)
    },
    onSuccess: (res) => {
      if (res.applied && res.worldpack) {
        queryClient.invalidateQueries({ queryKey: CACHE_KEYS.worlds.list() })
        queryClient.setQueryData(CACHE_KEYS.worlds.detail(res.worldpack.world_id), res.worldpack)
      }
    },
  })
}

