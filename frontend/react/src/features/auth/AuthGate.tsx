import { type FormEvent, type ReactNode, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { v2Get, v2Post, type V2Capabilities } from '@/api/v2'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

interface AuthGateProps {
  children: ReactNode
}

export function AuthGate({ children }: AuthGateProps) {
  const queryClient = useQueryClient()
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('')
  const capabilities = useQuery({
    queryKey: ['v2', 'capabilities'],
    queryFn: () => v2Get<V2Capabilities>('/system/capabilities'),
    retry: false,
  })
  const session = useQuery({
    queryKey: ['v2', 'auth-session'],
    queryFn: () => v2Get<{ authenticated: boolean }>('/auth/session'),
    enabled: capabilities.data?.auth_required === true,
    retry: false,
  })
  const login = useMutation({
    mutationFn: () =>
      v2Post('/auth/session', {
        username,
        password,
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['v2', 'auth-session'] }),
  })

  if (capabilities.isLoading || (capabilities.data?.auth_required && session.isLoading)) {
    return <FullPageStatus message="正在確認工作台狀態…" />
  }
  if (capabilities.isError) {
    return <FullPageStatus message="無法連線到 SagaForge API。請確認服務是否已啟動。" error />
  }
  if (!capabilities.data?.auth_required || session.data?.authenticated) {
    return children
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    login.mutate()
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md border-slate-700/80 bg-slate-950/90 shadow-2xl">
        <CardHeader>
          <p className="text-xs font-medium uppercase tracking-[0.24em] text-cyan-400">
            Private creator workspace
          </p>
          <CardTitle className="text-2xl">登入 SagaForge</CardTitle>
          <p className="text-sm text-slate-400">管理世界設定、故事狀態與 AI 工作佇列。</p>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={submit}>
            <div className="space-y-2">
              <Label htmlFor="username">管理員帳號</Label>
              <Input
                id="username"
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">密碼</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </div>
            {login.isError && (
              <p role="alert" className="text-sm text-red-400">
                登入失敗，請確認帳號與密碼。
              </p>
            )}
            <Button className="w-full" type="submit" disabled={login.isPending || password.length < 8}>
              {login.isPending ? '登入中…' : '進入工作台'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </main>
  )
}

function FullPageStatus({ message, error = false }: { message: string; error?: boolean }) {
  return (
    <div
      className={`flex min-h-screen items-center justify-center p-6 text-center ${error ? 'text-red-400' : 'text-slate-400'}`}
      role={error ? 'alert' : 'status'}
      aria-live="polite"
    >
      {message}
    </div>
  )
}
