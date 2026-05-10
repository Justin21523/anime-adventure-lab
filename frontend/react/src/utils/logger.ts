/**
 * Structured Logging System
 *
 * Provides centralized logging with different levels and context tracking.
 * Stores logs locally for debugging and can be extended for production monitoring.
 */

export const LogLevel = {
  DEBUG: 'debug',
  INFO: 'info',
  WARN: 'warn',
  ERROR: 'error',
} as const

export type LogLevel = typeof LogLevel[keyof typeof LogLevel]

export interface LogContext {
  component?: string
  action?: string
  sessionId?: string
  userId?: string
  url?: string
  method?: string
  status?: number
  error?: string
  [key: string]: any
}

interface LogEntry {
  timestamp: string
  level: LogLevel
  message: string
  context?: LogContext
}

class Logger {
  private isDev = import.meta.env.DEV
  private maxLocalLogs = 100

  /**
   * Core logging method
   */
  log(level: LogLevel, message: string, context?: LogContext): void {
    const timestamp = new Date().toISOString()
    const logEntry: LogEntry = {
      timestamp,
      level,
      message,
      ...(context && { context }),
    }

    // Development: Output to console with styling
    if (this.isDev) {
      this.logToConsole(level, message, context)
    }

    // Store locally for debugging (both dev and prod)
    this.storeLocally(logEntry)

    // Production: Send to monitoring service (future implementation)
    if (!this.isDev && level === LogLevel.ERROR) {
      this.sendToMonitoring(logEntry)
    }
  }

  /**
   * Log to browser console with appropriate styling
   */
  private logToConsole(level: LogLevel, message: string, context?: LogContext): void {
    const styles = {
      [LogLevel.DEBUG]: 'color: #888; font-weight: normal',
      [LogLevel.INFO]: 'color: #4A90E2; font-weight: normal',
      [LogLevel.WARN]: 'color: #F5A623; font-weight: bold',
      [LogLevel.ERROR]: 'color: #D0021B; font-weight: bold',
    }

    const consoleMethod = level === LogLevel.ERROR ? 'error'
                        : level === LogLevel.WARN ? 'warn'
                        : level === LogLevel.INFO ? 'info'
                        : 'log'

    if (context) {
      console[consoleMethod](`%c[${level.toUpperCase()}]`, styles[level], message, context)
    } else {
      console[consoleMethod](`%c[${level.toUpperCase()}]`, styles[level], message)
    }
  }

  /**
   * Store log entries in localStorage for debugging
   */
  private storeLocally(entry: LogEntry): void {
    try {
      const logs = this.getLocalLogs()
      logs.push(entry)

      // Keep only the most recent logs
      if (logs.length > this.maxLocalLogs) {
        logs.splice(0, logs.length - this.maxLocalLogs)
      }

      localStorage.setItem('app_logs', JSON.stringify(logs))
    } catch (error) {
      // Silently fail if localStorage is not available
      console.warn('Failed to store log locally:', error)
    }
  }

  /**
   * Get all logs from localStorage
   */
  getLocalLogs(): LogEntry[] {
    try {
      const stored = localStorage.getItem('app_logs')
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      return []
    }
  }

  /**
   * Clear all local logs
   */
  clearLocalLogs(): void {
    try {
      localStorage.removeItem('app_logs')
    } catch (error) {
      console.warn('Failed to clear local logs:', error)
    }
  }

  /**
   * Export logs as JSON for debugging
   */
  exportLogs(): string {
    const logs = this.getLocalLogs()
    return JSON.stringify(logs, null, 2)
  }

  /**
   * Send error logs to monitoring service (future implementation)
   */
  private sendToMonitoring(entry: LogEntry): void {
    // TODO: Implement integration with monitoring service
    // Examples: Sentry, LogRocket, DataDog, etc.
    //
    // if (window.Sentry) {
    //   window.Sentry.captureMessage(entry.message, {
    //     level: entry.level,
    //     extra: entry.context,
    //   })
    // }
  }

  /**
   * Convenience methods for each log level
   */

  debug(message: string, context?: LogContext): void {
    if (this.isDev) {
      this.log(LogLevel.DEBUG, message, context)
    }
  }

  info(message: string, context?: LogContext): void {
    this.log(LogLevel.INFO, message, context)
  }

  warn(message: string, context?: LogContext): void {
    this.log(LogLevel.WARN, message, context)
  }

  error(message: string, context?: LogContext): void {
    this.log(LogLevel.ERROR, message, context)
  }

  /**
   * Helper to log API requests
   */
  logApiRequest(method: string, url: string, context?: Omit<LogContext, 'method' | 'url'>): void {
    this.debug(`API Request: ${method} ${url}`, {
      method,
      url,
      ...context,
    })
  }

  /**
   * Helper to log API responses
   */
  logApiResponse(
    method: string,
    url: string,
    status: number,
    context?: Omit<LogContext, 'method' | 'url' | 'status'>
  ): void {
    const level = status >= 500 ? LogLevel.ERROR
                : status >= 400 ? LogLevel.WARN
                : LogLevel.DEBUG

    this.log(level, `API Response: ${method} ${url} - ${status}`, {
      method,
      url,
      status,
      ...context,
    })
  }

  /**
   * Helper to log component lifecycle events
   */
  logComponent(component: string, action: string, context?: Omit<LogContext, 'component' | 'action'>): void {
    this.debug(`[${component}] ${action}`, {
      component,
      action,
      ...context,
    })
  }
}

// Export singleton instance
export const logger = new Logger()

// Export for debugging in browser console
if (typeof window !== 'undefined') {
  (window as any).__logger = logger
}
