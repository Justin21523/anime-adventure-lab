import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiPost, apiPut } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { WorldCreateRequest, WorldPack, WorldUpdateRequest } from '../types/world.types'

export function useCreateWorld() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: WorldCreateRequest) => {
      return apiPost<WorldPack, WorldCreateRequest>('/worlds', request)
    },
    onSuccess: (world) => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.worlds.list() })
      queryClient.setQueryData(CACHE_KEYS.worlds.detail(world.world_id), world)
    },
  })
}

export function useUpdateWorld() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ worldId, world }: { worldId: string; world: WorldPack }) => {
      const payload: WorldUpdateRequest = { world }
      return apiPut<WorldPack, WorldUpdateRequest>(`/worlds/${worldId}`, payload)
    },
    onSuccess: (world) => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.worlds.list() })
      queryClient.setQueryData(CACHE_KEYS.worlds.detail(world.world_id), world)
    },
  })
}

export function useDeleteWorld() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (worldId: string) => {
      return apiDelete<{ success: boolean }>(`/worlds/${worldId}`)
    },
    onSuccess: (_res, worldId) => {
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.worlds.list() })
      queryClient.removeQueries({ queryKey: CACHE_KEYS.worlds.detail(worldId) })
    },
  })
}

