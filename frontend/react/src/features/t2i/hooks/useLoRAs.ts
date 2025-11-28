import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { LoRAListResponse } from '../types/t2i.types'

/**
 * Hook for fetching available LoRA models
 * LoRAs can be used for specific art styles (anime, manga, character-specific)
 */
export function useLoRAs() {
  return useQuery({
    queryKey: CACHE_KEYS.t2i.loras(),
    queryFn: async () => {
      const response = await apiGet<LoRAListResponse>('/t2i/loras')
      return response
    },
    staleTime: 10 * 60_000, // 10 minutes - LoRA list doesn't change frequently
  })
}
