import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { StoryPersona } from '../types/story.types'

/**
 * Hook for fetching available story personas
 */
export function usePersonas() {
  return useQuery({
    queryKey: CACHE_KEYS.story.personas(),
    queryFn: async () => {
      const response = await apiGet<StoryPersona[]>('/story/personas')
      return response
    },
    staleTime: 5 * 60_000, // 5 minutes (personas rarely change)
  })
}

/**
 * Hook for fetching a single persona
 */
export function usePersona(personaId?: string) {
  return useQuery({
    queryKey: CACHE_KEYS.story.persona(personaId!),
    queryFn: async () => {
      const response = await apiGet<StoryPersona[]>('/story/personas')
      const found = response.find((p) => p.persona_id === personaId)
      if (!found) {
        throw new Error(`Persona not found: ${personaId}`)
      }
      return found
    },
    enabled: !!personaId,
    staleTime: 5 * 60_000,
  })
}
