import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * Story session state management
 * Persisted to localStorage for session continuity
 */
interface SessionState {
  // Current active session
  currentSessionId: string | null
  setCurrentSessionId: (id: string | null) => void

  // Recent sessions (for quick access)
  recentSessions: string[]
  addRecentSession: (id: string) => void
  removeRecentSession: (id: string) => void
  clearRecentSessions: () => void

  // Session preferences
  autoSave: boolean
  setAutoSave: (enabled: boolean) => void
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      // State
      currentSessionId: null,
      recentSessions: [],
      autoSave: true,

      // Actions
      setCurrentSessionId: (id) => {
        set({ currentSessionId: id })
        if (id) {
          get().addRecentSession(id)
        }
      },

      addRecentSession: (id) => {
        set((state) => ({
          recentSessions: [
            id,
            ...state.recentSessions.filter((s) => s !== id),
          ].slice(0, 10), // Keep only last 10
        }))
      },

      removeRecentSession: (id) => {
        set((state) => ({
          recentSessions: state.recentSessions.filter((s) => s !== id),
        }))
      },

      clearRecentSessions: () => {
        set({ recentSessions: [] })
      },

      setAutoSave: (enabled) => {
        set({ autoSave: enabled })
      },
    }),
    {
      name: 'story-sessions', // LocalStorage key
      partialize: (state) => ({
        currentSessionId: state.currentSessionId,
        recentSessions: state.recentSessions,
        autoSave: state.autoSave,
      }),
    }
  )
)
