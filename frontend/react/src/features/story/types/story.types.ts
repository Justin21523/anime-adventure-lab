/**
 * Story feature types (API-aligned)
 *
 * Backend references:
 * - schemas/story.py
 * - api/routers/story.py
 */

import type { WorldAgentProfile, WorldPack } from '@/features/worlds/types/world.types'

export type RagMode = 'auto' | 'on' | 'off'

export interface SceneImage {
  image_url: string
  prompt: string
  negative_prompt: string
  generation_time: number
  seed?: number
  width: number
  height: number
}

export interface StoryTurnHistoryEntry {
  turn: number
  timestamp?: string | null
  player_input: string
  ai_response: string
  choice_id?: string | null
  scene_id?: string | null
  agent_used?: boolean | null
  enriched_player_input?: string | null
  rag_mode?: RagMode | null
  rag_query?: string | null
  rerank_mode?: RagMode | null
  knowledge_used?: Array<{
    content: string
    score?: number
    metadata?: Record<string, any>
    [key: string]: any
  }> | null
  agent_overlay?: Record<string, any> | null
  agent_actions?: AgentActions | null
  state_delta?: {
    flags?: Array<{ key: string; old: any; new: any }>
    stats?: Array<{ key: string; old: any; new: any; change?: number | null }>
    inventory?: {
      added?: Array<{ item: string; count: number }>
      removed?: Array<{ item: string; count: number }>
    }
    relationships?: Array<{ character_id: string; old: number; new: number; change: number }>
  } | null
  scene_image_job_id?: string | null
  scene_image?: SceneImage | null
  artifacts?: TurnArtifacts | null
}

export interface TurnArtifacts {
  rag?: {
    mode?: RagMode | null
    query?: string | null
    rerank_mode?: RagMode | null
    enable_rerank?: boolean | null
    hits?: StoryTurnHistoryEntry['knowledge_used'] | null
    [key: string]: any
  } | null
  agents?: {
    used?: boolean | null
    overlay?: Record<string, any> | null
    actions?: AgentActions | null
    [key: string]: any
  } | null
  diff?: StoryTurnHistoryEntry['state_delta'] | null
  t2i?: {
    scene_image_job_id?: string | null
    scene_image?: SceneImage | null
    [key: string]: any
  } | null
  world?: {
    world_id?: string | null
    applied_worldpack_updated_at?: string | null
    worldpack_updated_at_current?: string | null
    synced?: boolean | null
    [key: string]: any
  } | null
  [key: string]: any
}

export interface StoryMemoryStats {
  short_term_count: number
  summaries_count: number
  total_turns_covered: number
  turns_since_last_summary: number
  rag_available: boolean
}

export interface StoryShortTermMemory {
  turn: number
  action: string
  result: string
  scene?: string
}

export interface StoryMemorySummary {
  turn_range: string
  summary: string
  key_events: string[]
}

export interface StoryMemoryContext {
  short_term: StoryShortTermMemory[]
  summaries: StoryMemorySummary[]
  rag_results: Array<Record<string, any>>
}

export interface StoryContextSnapshot {
  session_id: string
  player_name: string
  current_scene?: Record<string, any> | null
  present_characters: Array<{
    character_id: string
    name: string
    role: string
    current_state?: string
    relationship_score?: number
    [key: string]: any
  }>
  world_flags: Record<string, any>
  main_plot_points: string[]
  recent_decisions: any[]
  total_scenes: number
  total_characters: number
}

export interface AgentToolResult {
  tool: string
  agent?: string
  success: boolean
  result?: any
  error?: string
  rollback_performed?: boolean
}

export interface AgentActions {
  decision_type: string
  reasoning: string
  contributors?: Array<{
    agent?: string
    reasoning?: string
    tool_calls?: Array<Record<string, any>>
  }>
  tool_results: AgentToolResult[]
  overall_success: boolean
  errors: string[]
}

export interface StoryChoice {
  choice_id: string
  text: string
  type?: string
  difficulty?: string
  can_choose?: boolean
  description?: string
  [key: string]: any
}

export interface StorySessionInfo {
  session_id: string
  player_name: string
  persona_id?: string | null
  world_id: string
  turn_count: number
  is_active: boolean
  updated_at: string
  enhanced_mode?: boolean
  current_scene?: string | null
}

export interface StorySessionDetail {
  session_id: string
  player_name: string
  world_id: string
  runtime_preset_id?: string | null
  turn_count: number
  is_active: boolean
  created_at: string
  updated_at: string
  current_scene?: string | null

  persona_id?: string | null
  player_template_id?: string | null
  worldpack_updated_at?: string | null
  narrative?: string | null
  choices: StoryChoice[]

  stats: Record<string, any>
  inventory: string[]
  flags: Record<string, any>
  turn_job_id?: string | null
  scene_image_job_id?: string | null
  scene_image?: SceneImage | null
  memory_stats?: StoryMemoryStats | null
  memory_context?: StoryMemoryContext | null
  turn_history?: StoryTurnHistoryEntry[] | null
  agent_actions?: AgentActions | null
  agent_used?: boolean

  rag_auto?: boolean | null
  rag_mode?: RagMode | null
  rag_available?: boolean | null
  enrich_with_rag?: boolean | null
  rag_next_turn?: boolean | null
  rag_query?: string | null

  rerank_mode?: RagMode | null
  rerank_next_turn?: boolean | null
}

export interface StoryPersona {
  persona_id: string
  name: string
  description: string
  personality_traits: string[]
  special_abilities: string[]
}

export interface StorySessionCreateRequest {
  player_name: string
  persona_id: string
  world_id?: string
  runtime_preset_id?: string | null
  setting?: string
  difficulty?: string
  enhanced_mode?: boolean
  use_agent?: boolean
  enrich_with_rag?: boolean
  rag_mode?: RagMode
  rerank_mode?: RagMode
  rag_query?: string
  initial_prompt?: string
  player_template_id?: string | null
  include_image?: boolean
}

export interface StoryTurnRequest {
  session_id: string
  player_input: string
  choice_id?: string | null
  use_agent?: boolean
  scenario_type?: string | null
  scenario_data?: Record<string, any> | null
  enrich_with_rag?: boolean
  rag_mode?: RagMode
  rerank_mode?: RagMode
  rag_query?: string | null
  top_k?: number
  include_image?: boolean
}

export interface StoryTurnJobResponse {
  success: boolean
  job_id: string
  status: string
}

export type StoryWorldSyncMode = 'add_only' | 'merge'

export interface StoryWorldSyncRequest {
  mode?: StoryWorldSyncMode
}

export interface StoryWorldSyncResponse {
  session_id: string
  world_id: string
  mode: string
  flags_added: string[]
  flags_updated: string[]
  characters_added: string[]
  characters_updated: string[]
  worldpack_updated_at?: string | null
}

export interface StoryWorldWritebackSuggestRequest {
  include_flags?: boolean
  include_characters?: boolean
  include_rag_note?: boolean
  max_new_characters?: number
}

export interface StoryWorldWritebackSuggestResponse {
  success: boolean
  session_id: string
  world_id: string
  patch: Record<string, any>
  worldpack: WorldPack
  rag_note?: string | null
  summary?: Record<string, any>
  errors: string[]
}

export interface StoryAgentProfileResponse {
  session_id: string
  world_id: string
  agent_profile: WorldAgentProfile
}

export interface StoryAgentProfileUpdateRequest {
  agent_profile: WorldAgentProfile
}

export interface StoryAgentProfilePatchRequest {
  enabled?: boolean
  enabled_agents?: string[]
  max_tool_calls?: number
  max_llm_calls?: number
  allowed_tools?: string[]
}

export interface StoryTurnResponse {
  session_id: string
  world_id: string
  turn_count: number
  narrative: string
  choices: StoryChoice[]
  stats: Record<string, any>
  inventory: string[]
  scene_id?: string | null
  flags: Record<string, any>
  agent_used: boolean
  agent_overlay?: Record<string, any> | null
  agent_actions?: Record<string, any> | null
  knowledge_used?: Array<Record<string, any>> | null
  context?: Record<string, any> | null
  scene_image_job_id?: string | null
  scene_image?: SceneImage | null
}
