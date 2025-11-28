import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient } from './config/query.config'
import { useUiStore } from './stores/uiStore'
import { Button } from './components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'

function App() {
  const { theme, setTheme, addNotification } = useUiStore()

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Anime Adventure Lab - React Frontend</CardTitle>
              <CardDescription>
                Modern React + TypeScript frontend for the comprehensive backend
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-2">環境搭建完成 ✅</h3>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>React 18 + TypeScript + Vite</li>
                  <li>TanStack Query + Router + Zustand</li>
                  <li>shadcn/ui + Tailwind CSS</li>
                  <li>API Client + Error Handling</li>
                  <li>SSE Streaming Support</li>
                </ul>
              </div>

              <div className="flex gap-2">
                <Button
                  variant={theme === 'dark' ? 'default' : 'outline'}
                  onClick={() => setTheme('dark')}
                >
                  Dark
                </Button>
                <Button
                  variant={theme === 'light' ? 'default' : 'outline'}
                  onClick={() => setTheme('light')}
                >
                  Light
                </Button>
                <Button
                  variant="secondary"
                  onClick={() =>
                    addNotification({
                      type: 'success',
                      title: 'Test Notification',
                      message: 'UI Store is working!',
                    })
                  }
                >
                  Test Notification
                </Button>
              </div>

              <div className="p-4 bg-muted rounded-md">
                <p className="text-sm">
                  <strong>Next Steps:</strong> Implement Story gameplay components,
                  create API hooks, and connect to backend endpoints.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>重要提醒</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                ⚠️ 由於 GPU 正在進行訓練，所有涉及模型的功能將使用 mock/stub 進行測試，不會加載真實模型。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* React Query DevTools (only in development) */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

export default App
