import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * UI state management
 * Theme, sidebar, notifications, etc.
 */
interface UiState {
  // Theme
  theme: 'light' | 'dark' | 'system'
  setTheme: (theme: 'light' | 'dark' | 'system') => void

  // Sidebar
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  toggleSidebar: () => void

  // Notifications
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void

  // Loading states (for global operations)
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  loadingMessage?: string
  setLoadingMessage: (message?: string) => void
}

interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message?: string
  timestamp: number
}

export const useUiStore = create<UiState>()(
  persist(
    (set, get) => ({
      // Theme
      theme: 'dark',
      setTheme: (theme) => {
        set({ theme })

        // Apply theme to document
        const root = window.document.documentElement
        root.classList.remove('light', 'dark')

        if (theme === 'system') {
          const systemTheme = window.matchMedia('(prefers-color-scheme: dark)')
            .matches
            ? 'dark'
            : 'light'
          root.classList.add(systemTheme)
        } else {
          root.classList.add(theme)
        }
      },

      // Sidebar
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      // Notifications
      notifications: [],
      addNotification: (notification) => {
        const id = Math.random().toString(36).substr(2, 9)
        const newNotification: Notification = {
          ...notification,
          id,
          timestamp: Date.now(),
        }

        set((state) => ({
          notifications: [newNotification, ...state.notifications].slice(0, 5), // Keep max 5
        }))

        // Auto-remove after 5 seconds for success/info
        if (notification.type === 'success' || notification.type === 'info') {
          setTimeout(() => {
            get().removeNotification(id)
          }, 5000)
        }
      },

      removeNotification: (id) => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        }))
      },

      clearNotifications: () => {
        set({ notifications: [] })
      },

      // Loading
      isLoading: false,
      setIsLoading: (loading) => set({ isLoading: loading }),
      loadingMessage: undefined,
      setLoadingMessage: (message) => set({ loadingMessage: message }),
    }),
    {
      name: 'ui-state',
      partialize: (state) => ({
        theme: state.theme,
        sidebarOpen: state.sidebarOpen,
      }),
    }
  )
)

// Initialize theme on load
if (typeof window !== 'undefined') {
  const store = useUiStore.getState()
  store.setTheme(store.theme)
}
