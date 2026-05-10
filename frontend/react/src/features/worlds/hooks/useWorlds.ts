import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { WorldSummary } from '../types/world.types'

export function useWorlds() {
  return useQuery({
    queryKey: CACHE_KEYS.worlds.list(),
    queryFn: async () => {
      return apiGet<WorldSummary[]>('/worlds')
    },
    staleTime: 60_000,
  })
}

