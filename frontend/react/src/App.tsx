import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient } from './config/query.config'
import { AuthGate } from './features/auth/AuthGate'
import { StoryWorkbenchV2 } from './features/story-v2/StoryWorkbenchV2'

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthGate>
        <StoryWorkbenchV2 />
      </AuthGate>
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  )
}

export default App
