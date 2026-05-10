import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { WorldPack } from '../types/world.types'

export function useWorld(worldId: string) {
  return useQuery({
    queryKey: CACHE_KEYS.worlds.detail(worldId),
    queryFn: async () => {
      return apiGet<WorldPack>(`/worlds/${worldId}`)
    },
    enabled: Boolean(worldId),
    staleTime: 30_000,
  })
}

