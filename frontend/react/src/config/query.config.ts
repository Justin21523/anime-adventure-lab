import { QueryClient, DefaultOptions } from '@tanstack/react-query'

/**
 * Default query options for React Query
 */
const defaultOptions: DefaultOptions = {
  queries: {
    staleTime: 30_000, // 30 seconds
    gcTime: 5 * 60_000, // 5 minutes (formerly cacheTime)
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
    retry: 1,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  },
  mutations: {
    retry: 0,
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
  },

  // Batch module
  batch: {
    all: ['batch'] as const,
    jobs: () => ['batch', 'jobs'] as const,
    job: (id: string) => ['batch', 'job', id] as const,
    status: (id: string) => ['batch', 'status', id] as const,
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
