/**
 * World Studio (WorldPacks) types (API-aligned)
 *
 * Backend references:
 * - schemas/world.py
 * - api/routers/worlds.py
 */

export type WorldCharacterRole = 'npc' | 'companion' | 'antagonist'

export interface WorldLoRAConfig {
  lora_id: string
  weight: number
}

export interface WorldVisualStyle {
  prompt_prefix: string
  negative_prompt: string
  base_model?: string | null
  default_loras: WorldLoRAConfig[]
}

export interface WorldAgentProfile {
  enabled: boolean
  enabled_agents: string[]
  max_tool_calls: number
  max_llm_calls: number
  allowed_tools: string[]
}

export interface WorldRagProfile {
  enable_rerank: boolean
}

export interface WorldPlayerTemplate {
  template_id: string
  name: string
  description: string
  personality_traits: string[]
  speaking_style: string
  background_story: string
  motivations: string[]
  persona_prompt: string
}

export interface WorldCharacterTemplate {
  character_id: string
  name: string
  role: WorldCharacterRole
  image_url?: string
  personality_traits: string[]
  speaking_style: string
  background_story: string
  motivations: string[]
  relationships: Record<string, string>
  persona_prompt: string
  content_restrictions: string[]
  start_in_opening: boolean
}

export interface WorldPack {
  world_id: string
  name: string
  description: string
  setting: string
  difficulty: string
  runtime_preset_id?: string | null

  visual: WorldVisualStyle
  player_templates: WorldPlayerTemplate[]
  characters: WorldCharacterTemplate[]
  world_flags: Record<string, boolean>
  agent_profile: WorldAgentProfile
  rag_profile: WorldRagProfile

  created_at: string
  updated_at: string
}

export interface WorldSummary {
  world_id: string
  name: string
  description: string
  setting: string
  difficulty: string
  updated_at: string
  player_templates_count: number
  characters_count: number
  default_loras_count: number
}

export interface WorldCreateRequest {
  world_id: string
  name: string
  description?: string
  setting?: string
  difficulty?: string
}

export interface WorldUpdateRequest {
  world: WorldPack
}

export interface WorldAgentSuggestRequest {
  instruction: string
  apply?: boolean
  rag_top_k?: number
  max_new_characters?: number
  max_new_player_templates?: number
  include_visual?: boolean
}

export interface WorldAgentSuggestContributor {
  agent?: string
  reasoning?: string
  patch?: Record<string, any>
}

export interface WorldAgentSuggestResponse {
  success: boolean
  applied: boolean
  world_id: string
  patch: Record<string, any>
  worldpack: WorldPack
  contributors: WorldAgentSuggestContributor[]
  errors: string[]
}
