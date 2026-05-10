import { Component, type ErrorInfo, type ReactNode } from 'react'
import { Button } from './ui/button'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

/**
 * Error Boundary Component
 * Catches JavaScript errors anywhere in the child component tree
 */
export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.setState({
      error,
      errorInfo,
    })
  }

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    })
  }

  private handleReload = () => {
    window.location.reload()
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background">
          <Card className="max-w-2xl w-full border-red-500/30">
            <CardHeader>
              <CardTitle className="text-red-400">發生錯誤</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-red-900/20 border border-red-500/30 rounded">
                <p className="text-red-300 font-mono text-sm">
                  {this.state.error?.toString()}
                </p>
              </div>

              {this.state.errorInfo && (
                <details className="text-sm">
                  <summary className="cursor-pointer text-slate-400 hover:text-slate-300 mb-2">
                    查看詳細資訊
                  </summary>
                  <pre className="p-4 bg-slate-800/50 rounded overflow-x-auto text-xs text-slate-400">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}

              <div className="flex gap-2">
                <Button onClick={this.handleReset} variant="outline">
                  重試
                </Button>
                <Button onClick={this.handleReload} variant="default">
                  重新載入頁面
                </Button>
              </div>

              <p className="text-sm text-slate-400">
                如果問題持續發生，請檢查瀏覽器控制台或聯繫系統管理員。
              </p>
            </CardContent>
          </Card>
        </div>
      )
    }

    return this.props.children
  }
}
