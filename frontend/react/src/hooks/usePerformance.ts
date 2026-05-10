import { useEffect } from 'react'

/**
 * Performance monitoring hook
 * Tracks component render times and reports to console in development
 */
export function usePerformance(componentName: string) {
  useEffect(() => {
    if (import.meta.env.DEV) {
      const startTime = performance.now()

      return () => {
        const endTime = performance.now()
        const renderTime = endTime - startTime

        // Only log if render time is significant (> 16ms for 60fps)
        if (renderTime > 16) {
          console.log(`[Performance] ${componentName} rendered in ${renderTime.toFixed(2)}ms`)
        }
      }
    }
  })
}

/**
 * Debounce hook for expensive operations
 * Returns a debounced version of the callback
 */
export function useDebounce<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    timeoutId = setTimeout(() => {
      callback(...args)
      timeoutId = null
    }, delay)
  }
}

/**
 * Throttle hook for limiting function execution frequency
 * Returns a throttled version of the callback
 */
export function useThrottle<T extends (...args: any[]) => any>(
  callback: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      callback(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}
