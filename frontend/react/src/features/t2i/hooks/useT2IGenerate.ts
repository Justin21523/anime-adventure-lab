import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import { CACHE_KEYS } from '@/config/query.config'
import type { T2IGenerateRequest, T2IGenerateResponse } from '../types/t2i.types'

/**
 * Hook for T2I image generation
 * Used by story system to generate scene visuals, character portraits, event illustrations
 */
export function useT2IGenerate() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: T2IGenerateRequest) => {
      const response = await apiPost<T2IGenerateResponse>('/t2i/generate', {
        prompt: request.prompt,
        negative_prompt: request.negative_prompt || 'low quality, blurry, distorted',
        width: request.width || 512,
        height: request.height || 768,
        num_inference_steps: request.num_inference_steps || 30,
        guidance_scale: request.guidance_scale || 7.5,
        num_images: request.num_images || 1,
        seed: request.seed,
        loras: request.loras || [],
        controlnet_image: request.controlnet_image,
        controlnet_type: request.controlnet_type,
        session_id: request.session_id, // Link to story session
      })
      return response
    },
    onSuccess: () => {
      // Invalidate history when new images are generated
      queryClient.invalidateQueries({ queryKey: CACHE_KEYS.t2i.history() })
    },
  })
}
