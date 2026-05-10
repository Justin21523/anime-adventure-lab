import { QueryClient, type QueryClientConfig } from '@tanstack/react-query'

/**
 * Default query options for React Query
 * Optimized for performance with longer stale times and smart refetching
 */
const defaultOptions: QueryClientConfig['defaultOptions'] = {
  queries: {
    // Increase stale time to reduce unnecessary refetches
    staleTime: 2 * 60_000, // 2 minutes (was 30s)

    // Keep data in cache longer for better UX when navigating back
    gcTime: 10 * 60_000, // 10 minutes (was 5 minutes)

    // Disable window focus refetch to avoid unnecessary API calls
    refetchOnWindowFocus: false,

    // Refetch on reconnect to get fresh data after network issues
    refetchOnReconnect: true,

    // Retry failed requests with exponential backoff
    retry: 2, // Increased from 1 for better reliability
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

    // Network mode configuration
    networkMode: 'online' as const,
  },
  mutations: {
    retry: 0,
    // Network mode for mutations
    networkMode: 'online' as const,
  },
}

/**
 * Create Query Client instance
 */
export const queryClient = new QueryClient({
  defaultOptions,
})

/**
 * Cache key factory for type-safe query keys
 * Following React Query best practices with hierarchical keys
 */
export const CACHE_KEYS = {
  // Story module
  story: {
    all: ['story'] as const,
    sessions: () => ['story', 'sessions'] as const,
    session: (id: string) => ['story', 'session', id] as const,
    context: (id: string) => ['story', 'context', id] as const,
    agentProfile: (id: string) => ['story', 'agent_profile', id] as const,
    personas: () => ['story', 'personas'] as const,
    persona: (id: string) => ['story', 'persona', id] as const,
    turns: (sessionId: string) => ['story', 'turns', sessionId] as const,
  },

  // RAG module
  rag: {
    all: ['rag'] as const,
    documents: (worldId?: string) =>
      worldId ? (['rag', 'documents', worldId] as const) : (['rag', 'documents'] as const),
    document: (id: string) => ['rag', 'document', id] as const,
    search: (query: string, worldId?: string) =>
      ['rag', 'search', query, worldId] as const,
    stats: (worldId?: string) =>
      worldId ? (['rag', 'stats', worldId] as const) : (['rag', 'stats'] as const),
  },

  // Worlds (World Studio)
  worlds: {
    all: ['worlds'] as const,
    list: () => ['worlds', 'list'] as const,
    detail: (worldId: string) => ['worlds', 'detail', worldId] as const,
  },

  // T2I module
  t2i: {
    all: ['t2i'] as const,
    generations: () => ['t2i', 'generations'] as const,
    generation: (id: string) => ['t2i', 'generation', id] as const,
    history: (sessionId?: string) =>
      sessionId ? (['t2i', 'history', sessionId] as const) : (['t2i', 'history'] as const),
    loras: () => ['t2i', 'loras'] as const,
    controlnets: () => ['t2i', 'controlnets'] as const,
  },

  // Agent module
  agent: {
    all: ['agent'] as const,
    tasks: () => ['agent', 'tasks'] as const,
    task: (id: string) => ['agent', 'task', id] as const,
    tools: () => ['agent', 'tools'] as const,
    tool: (name: string) => ['agent', 'tool', name] as const,
    catalog: () => ['agent', 'catalog'] as const,
  },

  // Batch module
  batch: {
    all: ['batch'] as const,
    jobs: () => ['batch', 'jobs'] as const,
    job: (id: string) => ['batch', 'job', id] as const,
    status: (id: string) => ['batch', 'status', id] as const,
  },

  // Jobs module
  jobs: {
    all: ['jobs'] as const,
    list: () => ['jobs', 'list'] as const,
    job: (id: string) => ['jobs', 'job', id] as const,
  },

  // Runtime presets
  runtime: {
    all: ['runtime'] as const,
    presets: () => ['runtime', 'presets'] as const,
  },

  // Monitoring module
  monitoring: {
    all: ['monitoring'] as const,
    health: () => ['monitoring', 'health'] as const,
    metrics: () => ['monitoring', 'metrics'] as const,
    performance: () => ['monitoring', 'performance'] as const,
  },

  // Models module
  models: {
    all: ['models'] as const,
    list: () => ['models', 'list'] as const,
    loaded: () => ['models', 'loaded'] as const,
    model: (name: string) => ['models', 'model', name] as const,
  },

  // System
  system: {
    health: () => ['system', 'health'] as const,
  },
} as const

/**
 * Cache time constants (in milliseconds)
 */
export const CACHE_TIME = {
  SHORT: 30_000, // 30 seconds
  MEDIUM: 5 * 60_000, // 5 minutes
  LONG: 30 * 60_000, // 30 minutes
  VERY_LONG: 24 * 60 * 60_000, // 24 hours
} as const

/**
 * Stale time constants (in milliseconds)
 */
export const STALE_TIME = {
  INSTANT: 0, // Always stale
  SHORT: 10_000, // 10 seconds
  MEDIUM: 30_000, // 30 seconds
  LONG: 5 * 60_000, // 5 minutes
  NEVER: Infinity, // Never stale
} as const
