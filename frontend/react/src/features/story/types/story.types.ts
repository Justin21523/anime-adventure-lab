/**
 * Story feature types
 * Updated to match backend SceneImage integration
 */

export interface SceneImage {
  image_url: string
  prompt: string
  negative_prompt: string
  generation_time: number
  seed?: number
  width: number
  height: number
}

export interface StorySession {
  session_id: string
  player_name: string
  persona_id: string
  world_id: string
  turn_count: number
  narrative: string
  dialogue?: string
  choices?: StoryChoice[]
  character: CharacterState
  inventory: InventoryItem[]
  flags: Record<string, any>
  created_at: string
  updated_at: string
  scene_image?: SceneImage | null
}

export interface StoryChoice {
  id: string
  text: string
  description?: string
}

export interface CharacterState {
  name: string
  level: number
  hp: number
  max_hp: number
  mp?: number
  max_mp?: number
  stats?: Record<string, number>
  status_effects?: string[]
}

export interface InventoryItem {
  id: string
  name: string
  description?: string
  quantity: number
  type?: string
}

export interface StoryPersona {
  persona_id: string
  name: string
  description: string
  style: string
  example_dialogue?: string
}

export interface StorySessionCreateRequest {
  player_name: string
  persona_id: string
  world_id?: string
  initial_prompt?: string
}

export interface StoryTurnRequest {
  session_id: string
  player_input: string
  include_image?: boolean
}

export interface StoryTurnResponse {
  session_id: string
  turn_count: number
  narrative: string
  choices: Array<Record<string, any>>
  stats: Record<string, any>
  inventory: string[]
  scene_id?: string | null
  flags: Record<string, any>
  agent_used: boolean
  agent_overlay?: Record<string, any> | null
  knowledge_used?: Array<Record<string, any>> | null
  context?: Record<string, any> | null
  scene_image?: SceneImage | null
}

export interface StorySessionListResponse {
  sessions: Array<{
    session_id: string
    player_name: string
    persona_id: string
    turn_count: number
    last_played: string
  }>
  total: number
}
