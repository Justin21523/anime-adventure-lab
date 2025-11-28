/**
 * Story feature types
 * These will eventually be replaced by auto-generated types from OpenAPI
 */

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
  session: StorySession
  turn_result: {
    narrative: string
    dialogue?: string
    choices?: StoryChoice[]
    image_path?: string
    success: boolean
  }
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
