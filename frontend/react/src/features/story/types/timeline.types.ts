import type { AgentActions, RagMode, SceneImage, StoryTurnHistoryEntry, TurnArtifacts } from './story.types'

export interface TimelineTurn {
  turn_index: number
  turn: number
  timestamp?: string | null
  scene?: string | null
  choice_id?: string | null

  action: string
  result: string

  enriched_player_input?: string | null
  rag_mode?: RagMode | null
  rerank_mode?: RagMode | null
  rag_query?: string | null
  knowledge_used?: StoryTurnHistoryEntry['knowledge_used'] | null

  agent_overlay?: Record<string, any> | null
  agent_actions?: AgentActions | null
  state_delta?: StoryTurnHistoryEntry['state_delta'] | null
  scene_image?: SceneImage | null
  artifacts?: TurnArtifacts | null

  raw?: StoryTurnHistoryEntry | null
}
