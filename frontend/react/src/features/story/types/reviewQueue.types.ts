import type { WorldAgentSuggestResponse } from '@/features/worlds/types/world.types'
import type { WorldPackApplySelection } from '@/features/worlds/utils/worldPackPatch'
import type { StoryWorldWritebackSuggestResponse } from './story.types'

export type ReviewQueueStatus = 'pending' | 'applied'

export type WritebackApplySelection = {
  world_flags: boolean
  characters: boolean
  rag_note: boolean
}

export type ReviewQueueItem =
  | {
      kind: 'world_writeback'
      id: string
      created_at: string
      status: ReviewQueueStatus
      applied_at?: string | null
      world_id?: string | null
      turn?: number | null
      selection: WritebackApplySelection
      response: StoryWorldWritebackSuggestResponse
    }
  | {
      kind: 'world_ai'
      id: string
      created_at: string
      status: ReviewQueueStatus
      applied_at?: string | null
      world_id?: string | null
      selection: WorldPackApplySelection
      response: WorldAgentSuggestResponse
    }
  | {
      kind: 'state_delta'
      id: string
      created_at: string
      status: ReviewQueueStatus
      applied_at?: string | null
      world_id?: string | null
      turn: number
      artifacts: Record<string, any>
    }
