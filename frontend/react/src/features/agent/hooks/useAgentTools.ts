import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { AgentToolsResponse } from '../types/agent.types'

/**
 * Hook for fetching available agent tools
 */
export function useAgentTools() {
  return useQuery({
    queryKey: CACHE_KEYS.agent.tools(),
    queryFn: async () => {
      const response = await apiGet<AgentToolsResponse>('/agent/tools')
      return response
    },
    staleTime: 5 * 60_000, // 5 minutes
  })
}
