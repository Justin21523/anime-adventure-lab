import type { ReviewQueueItem } from '../types/reviewQueue.types'

export function reviewQueueStorageKey(sessionId: string): string {
  return `story_review_queue_${String(sessionId || '').trim()}`
}

export function emitReviewQueueUpdated(sessionId: string): void {
  try {
    const sid = String(sessionId || '').trim()
    if (!sid) return
    if (typeof window === 'undefined') return
    window.dispatchEvent(new CustomEvent('reviewqueue:update', { detail: { sessionId: sid } }))
  } catch {
    // ignore
  }
}

export function loadReviewQueue(sessionId: string): ReviewQueueItem[] {
  const key = reviewQueueStorageKey(sessionId)
  if (!sessionId) return []
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []

    const normalized: ReviewQueueItem[] = []
    for (const item of parsed) {
      if (!item || typeof item !== 'object') continue
      const kind = (item as any).kind
      if (!kind && (item as any).response && (item as any).selection) {
        // Backwards compatibility: old queue items were writeback-only without a `kind` field.
        normalized.push({ ...(item as any), kind: 'world_writeback' } as ReviewQueueItem)
        continue
      }
      if (kind === 'world_writeback' || kind === 'world_ai' || kind === 'state_delta') {
        normalized.push(item as ReviewQueueItem)
      }
    }
    return normalized
  } catch {
    return []
  }
}

export function saveReviewQueue(sessionId: string, queue: ReviewQueueItem[], maxItems: number = 20): void {
  if (!sessionId) return
  const key = reviewQueueStorageKey(sessionId)
  try {
    localStorage.setItem(key, JSON.stringify((queue || []).slice(0, Math.max(0, maxItems))))
    emitReviewQueueUpdated(sessionId)
  } catch {
    // ignore
  }
}

export function enqueueReviewQueueItem(sessionId: string, item: ReviewQueueItem, maxItems: number = 20): ReviewQueueItem[] {
  const current = loadReviewQueue(sessionId)
  const next = [item, ...current].slice(0, Math.max(0, maxItems))
  saveReviewQueue(sessionId, next, maxItems)
  return next
}
