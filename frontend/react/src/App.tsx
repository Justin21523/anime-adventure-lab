import { useState } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient } from './config/query.config'
import { useSessionStore } from './stores/sessionStore'
import { SessionList } from './features/story/components/SessionList'
import { NewStoryForm } from './features/story/components/NewStoryForm'
import { StoryGameScreen } from './features/story/components/StoryGameScreen'

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

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background">
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
      </div>

      {/* React Query DevTools (only in development) */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

export default App
