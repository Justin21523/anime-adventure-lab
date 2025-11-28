import { useState } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient } from './config/query.config'
import { useSessionStore } from './stores/sessionStore'
import { Button } from './components/ui/button'
import { SessionList } from './features/story/components/SessionList'
import { NewStoryForm } from './features/story/components/NewStoryForm'
import { StoryGameScreen } from './features/story/components/StoryGameScreen'
import { RAGManagement } from './features/rag/components/RAGManagement'
import { BatchMonitor } from './features/batch/components/BatchMonitor'

type Route = 'home' | 'new-story' | 'game' | 'rag' | 'batch'

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
        {/* 簡單導航欄 */}
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
                  <Button
                    variant={route === 'rag' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setRoute('rag')}
                  >
                    RAG
                  </Button>
                  <Button
                    variant={route === 'batch' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setRoute('batch')}
                  >
                    批次任務
                  </Button>
                </div>
              </div>
            </div>
          </nav>
        )}

        {/* 路由內容 */}
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

        {route === 'rag' && <RAGManagement />}

        {route === 'batch' && <BatchMonitor />}
      </div>

      {/* React Query DevTools (only in development) */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

export default App
