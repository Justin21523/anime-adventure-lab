/**
 * T2I (Text-to-Image) Feature Types
 */

export interface LoRAConfig {
  name: string
  weight: number
  path?: string
}

export interface T2IGenerateRequest {
  prompt: string
  negative_prompt?: string
  width?: number
  height?: number
  num_inference_steps?: number
  guidance_scale?: number
  num_images?: number
  seed?: number
  loras?: LoRAConfig[]
  controlnet_image?: string
  controlnet_type?: string
  session_id?: string
}

export interface GeneratedImage {
  image_url: string
  seed: number
  prompt: string
  metadata?: Record<string, any>
}

export interface T2IGenerateResponse {
  images: GeneratedImage[]
  generation_time: number
  model_used: string
}

export interface LoRAInfo {
  name: string
  path: string
  description?: string
  trigger_words?: string[]
  base_model: string
  tags?: string[]
}

export interface LoRAListResponse {
  loras: LoRAInfo[]
  total: number
}

export interface T2IHistoryItem {
  id: string
  timestamp: string
  prompt: string
  negative_prompt?: string
  images: GeneratedImage[]
  settings: {
    width: number
    height: number
    steps: number
    cfg_scale: number
    seed?: number
    loras?: LoRAConfig[]
  }
}

export interface T2IHistoryResponse {
  history: T2IHistoryItem[]
  total: number
}
