import { useState, lazy, Suspense } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient } from './config/query.config'
import { useSessionStore } from './stores/sessionStore'
import { Button } from './components/ui/button'

// Lazy load feature components for better performance
const SessionList = lazy(() => import('./features/story/components/SessionList').then(m => ({ default: m.SessionList })))
const NewStoryForm = lazy(() => import('./features/story/components/NewStoryForm').then(m => ({ default: m.NewStoryForm })))
const StoryGameScreen = lazy(() => import('./features/story/components/StoryGameScreen').then(m => ({ default: m.StoryGameScreen })))

type Route = 'home' | 'new-story' | 'game'

function App() {
  const [route, setRoute] = useState<Route>('home')
  const { currentSessionId, setCurrentSessionId } = useSessionStore()

  const handleSelectSession = (sessionId: string) => {
    setCurrentSessionId(sessionId)
    setRoute('game')
  }

  const handleNewStory = () => {
    setRoute('new-story')
  }

  const handleStoryCreated = (sessionId: string) => {
    setCurrentSessionId(sessionId)
    setRoute('game')
  }

  const handleCancelNewStory = () => {
    setRoute('home')
  }

  // Loading fallback component
  const LoadingFallback = () => (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
        <p className="text-slate-400">載入中...</p>
      </div>
    </div>
  )

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background">
        {/* 簡單導航欄（Story-only） */}
        {route !== 'game' && (
          <nav className="border-b border-slate-800">
            <div className="container mx-auto px-8 py-4">
              <div className="flex items-center gap-4">
                <h1 className="text-xl font-bold">Anime Adventure Lab</h1>
                <div className="flex gap-2 ml-auto">
                  <Button
                    variant={route === 'home' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setRoute('home')}
                  >
                    故事
                  </Button>
                </div>
              </div>
            </div>
          </nav>
        )}

        {/* 路由內容 - Wrapped with Suspense for code splitting */}
        <Suspense fallback={<LoadingFallback />}>
          {route === 'home' && (
            <div className="container mx-auto p-8">
              <SessionList
                onSelectSession={handleSelectSession}
                onNewSession={handleNewStory}
              />
            </div>
          )}

          {route === 'new-story' && (
            <div className="container mx-auto p-8">
              <NewStoryForm
                onSuccess={handleStoryCreated}
                onCancel={handleCancelNewStory}
              />
            </div>
          )}

          {route === 'game' && currentSessionId && (
            <StoryGameScreen sessionId={currentSessionId} />
          )}
        </Suspense>
      </div>

      {/* React Query DevTools (only in development) */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

export default App
