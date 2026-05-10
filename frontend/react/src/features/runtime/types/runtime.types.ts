export interface RuntimePresetLLM {
  model_name: string
  torch_dtype?: string
  device_map?: string
  use_quantization?: boolean
  quantization_bits?: number
}

export interface RuntimePresetT2I {
  model_id?: string | null
  torch_dtype?: string
  enable_attention_slicing?: boolean
  enable_vae_slicing?: boolean
  enable_vae_tiling?: boolean
  enable_cpu_offload?: boolean
  enable_sequential_cpu_offload?: boolean

  default_width?: number
  default_height?: number
  default_steps?: number
  default_guidance_scale?: number

  max_width?: number
  max_height?: number
  max_steps?: number
}

export interface RuntimePreset {
  preset_id: string
  name: string
  description?: string
  llm: RuntimePresetLLM
  t2i: RuntimePresetT2I
}

export interface RuntimePresetCatalogResponse {
  success: boolean
  default_preset_id: string
  presets: RuntimePreset[]
}

