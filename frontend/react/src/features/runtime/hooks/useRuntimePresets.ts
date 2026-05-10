import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { RuntimePresetCatalogResponse } from '../types/runtime.types'

export function useRuntimePresets() {
  return useQuery({
    queryKey: CACHE_KEYS.runtime.presets(),
    queryFn: async () => {
      return apiGet<RuntimePresetCatalogResponse>('/runtime/presets')
    },
    staleTime: 10 * 60_000,
  })
}

