# Anime Adventure Lab - React Frontend

Modern React + TypeScript frontend for the Anime Adventure Lab backend.

## 📋 Overview

This is the primary React-based UI for the backend (Gradio 已移除), providing:
- Story gameplay interface (Phase 1 MVP)
- Admin dashboard for RAG, Agent, Batch, T2I management
- Real-time updates via SSE streaming
- Type-safe API integration with auto-generated types
- Modern UI with shadcn/ui and Tailwind CSS

## 🏗️ Tech Stack

### Core
- **React 18.2** - UI library
- **TypeScript 5.0** - Type safety
- **Vite 5.0** - Build tool
- **TanStack Query v5** - Server state management
- **TanStack Router** - Type-safe routing
- **Zustand 4.4** - Client state management

### UI
- **shadcn/ui** - Component library
- **Tailwind CSS 3.4** - Styling
- **lucide-react** - Icons
- **class-variance-authority** - Component variants

### API Integration
- **Axios 1.6** - HTTP client
- **openapi-typescript** - Type generation from OpenAPI schema
- **@microsoft/fetch-event-source** - SSE streaming

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Generate API types from backend
npm run generate:api

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`.

## 📁 Project Structure

```
frontend/react/
├── src/
│   ├── api/                    # API layer
│   │   ├── client.ts           # Axios configuration
│   │   ├── generated/          # Auto-generated (DO NOT EDIT)
│   │   │   └── api.ts          # OpenAPI types
│   │   └── hooks/              # React Query hooks (planned)
│   │
│   ├── components/             # UI components
│   │   └── ui/                 # shadcn/ui base components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── textarea.tsx
│   │       └── label.tsx
│   │
│   ├── features/               # Feature modules (planned)
│   │   ├── story/              # Story gameplay
│   │   ├── rag/                # RAG management
│   │   ├── agent/              # Agent system
│   │   ├── batch/              # Batch processing
│   │   └── t2i/                # Text-to-Image
│   │
│   ├── stores/                 # Zustand stores
│   │   ├── sessionStore.ts     # Story session state
│   │   └── uiStore.ts          # UI state (theme, sidebar, notifications)
│   │
│   ├── hooks/                  # Custom hooks
│   │   └── useSSE.ts           # SSE streaming hook
│   │
│   ├── lib/                    # Utilities
│   │   ├── utils.ts            # cn(), formatters, etc.
│   │   └── api-error.ts        # API error handling
│   │
│   ├── config/                 # Configuration
│   │   └── query.config.ts     # React Query config
│   │
│   ├── App.tsx                 # Root component
│   ├── main.tsx                # Entry point
│   └── index.css               # Global styles
│
├── scripts/
│   └── generate-api-types.sh  # API type generation script
│
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
├── tsconfig.json               # TypeScript configuration
└── package.json
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```bash
# API endpoint (optional, defaults to http://localhost:8000)
VITE_API_BASE=http://localhost:8000/api/v1

# WebSocket endpoint (optional, for future use)
VITE_WS_BASE=ws://localhost:8000/ws
```

### Proxy Configuration

The Vite dev server proxies `/api/v1` requests to the backend server (configured in `vite.config.ts`):

```typescript
server: {
  port: 3000,
  proxy: {
    '/api/v1': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
},
```

## 🛠️ Development Workflow

### 1. Generate API Types

Whenever the backend API changes, regenerate TypeScript types:

```bash
npm run generate:api
```

This script:
1. Runs `scripts/generate_openapi_schema.py` (generates `openapi.json`)
2. Runs `openapi-typescript` to generate `src/api/generated/api.ts`

### 2. Type Checking

Check for TypeScript errors without building:

```bash
npm run type-check
```

### 3. Build for Production

```bash
npm run build
```

Build output goes to `dist/`.

### 4. Preview Production Build

```bash
npm run preview
```

## 📦 Key Features

### Type-Safe API Client

```typescript
import { apiGet, apiPost } from '@/api/client'

// Type-safe requests
const session = await apiGet<StorySession>('/story/session/abc123')
const result = await apiPost<TurnResult>('/story/turn', {
  player_input: 'explore the forest',
  session_id: 'abc123',
})
```

### React Query Integration

```typescript
import { useQuery } from '@tanstack/react-query'
import { CACHE_KEYS } from '@/config/query.config'

function useStorySession(sessionId: string) {
  return useQuery({
    queryKey: CACHE_KEYS.story.session(sessionId),
    queryFn: () => apiGet(`/story/session/${sessionId}`),
  })
}
```

### SSE Streaming

```typescript
import { useSSE } from '@/hooks/useSSE'

function ChatStream() {
  const { data, isConnected } = useSSE({
    url: '/api/v1/chat/stream',
    body: { messages: [...] },
  })

  return <div>{data?.content}</div>
}
```

### State Management

```typescript
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'

function Component() {
  // Story session state
  const { currentSessionId, setCurrentSessionId } = useSessionStore()

  // UI state
  const { theme, setTheme, addNotification } = useUiStore()

  return <div>...</div>
}
```

## 🎨 UI Components

We use shadcn/ui components. Already implemented:
- `Button` - All variants (default, outline, ghost, destructive, etc.)
- `Card` - Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- `Input` - Text input
- `Textarea` - Multi-line text input
- `Label` - Form label

More components will be added as needed.

## 🔄 Migration Plan

### Phase 1: Story MVP (Weeks 1-3) ✅ Week 1-2 Complete
- [x] Environment setup
- [x] API client and type generation
- [x] React Query configuration
- [x] Zustand stores
- [x] SSE hooks
- [x] Base UI components
- [ ] Story gameplay components
- [ ] Story API hooks

### Phase 2: Admin Dashboard (Weeks 4-7)
- [ ] RAG management UI
- [ ] Agent task runner
- [ ] Batch job monitoring
- [ ] T2I generation interface
- [ ] System monitoring

### Phase 3: Optimization (Weeks 8-9)
- [ ] Code splitting
- [ ] Performance optimization
- [ ] Production deployment

## ⚠️ Important Notes

### GPU Training in Progress
Due to ongoing GPU training, **all model-related features will use mocks/stubs** and will NOT load real models. This avoids conflicts with training processes.

### API Type Generation
The `src/api/generated/` directory is **auto-generated**. DO NOT manually edit files in this directory. Always regenerate types after backend changes.

### CORS Configuration
The backend is already configured to allow requests from `http://localhost:3000` (see `configs/app.yaml`).

## 📚 Resources

- [React Query Documentation](https://tanstack.com/query/latest)
- [TanStack Router Documentation](https://tanstack.com/router/latest)
- [shadcn/ui Documentation](https://ui.shadcn.com/)
- [Zustand Documentation](https://docs.pmnd.rs/zustand)

## 🐛 Troubleshooting

### API Types Not Generated
```bash
# Ensure backend server is importable
cd ../../../
python -c "from api.main import app; print('OK')"

# Regenerate types
cd frontend/react
npm run generate:api
```

### Port Already in Use
```bash
# Change port in vite.config.ts
server: {
  port: 3001,  // Use different port
}
```

### Backend Connection Failed
- Ensure backend is running on `http://localhost:8000`
- Check CORS configuration in `configs/app.yaml`
- Verify proxy configuration in `vite.config.ts`

## 🤝 Contributing

1. Follow the established project structure
2. Use TypeScript for all new files
3. Follow the component naming conventions (PascalCase)
4. Update types after backend changes (`npm run generate:api`)
5. Run type checking before committing (`npm run type-check`)

## 📄 License

Part of the Anime Adventure Lab project.
