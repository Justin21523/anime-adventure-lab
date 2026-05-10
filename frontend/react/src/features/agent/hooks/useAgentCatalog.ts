import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { AgentCatalogResponse } from '../types/agent.types'

export function useAgentCatalog() {
  return useQuery({
    queryKey: CACHE_KEYS.agent.catalog(),
    queryFn: async () => {
      return apiGet<AgentCatalogResponse>('/agent/catalog')
    },
    staleTime: 10 * 60_000,
  })
}

