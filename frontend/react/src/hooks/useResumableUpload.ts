import { useState, useCallback, useRef, useEffect } from 'react'
import {
  ResumableUpload,
  type ResumableUploadConfig,
  type ResumableUploadProgress,
} from '@/lib/resumable-upload'

export type { ResumableUploadProgress }
import { logger } from '@/utils/logger'

/**
 * Upload state
 */
export type UploadState = 'idle' | 'uploading' | 'paused' | 'completed' | 'error' | 'cancelled'

/**
 * Hook result
 */
export interface UseResumableUploadResult {
  /**
   * Current upload state
   */
  state: UploadState

  /**
   * Current progress information
   */
  progress: ResumableUploadProgress | null

  /**
   * Error if upload failed
   */
  error: Error | null

  /**
   * Upload result if completed
   */
  result: any

  /**
   * Whether upload is in progress
   */
  isUploading: boolean

  /**
   * Whether upload is paused
   */
  isPaused: boolean

  /**
   * Whether upload is completed
   */
  isCompleted: boolean

  /**
   * Start upload
   */
  upload: (
    file: File,
    uploadUrl: string,
    additionalData?: Record<string, any>,
    config?: ResumableUploadConfig
  ) => Promise<void>

  /**
   * Pause upload
   */
  pause: () => void

  /**
   * Resume upload
   */
  resume: () => Promise<void>

  /**
   * Cancel upload
   */
  cancel: () => void

  /**
   * Reset hook state
   */
  reset: () => void
}

/**
 * Default progress
 */
const DEFAULT_PROGRESS: ResumableUploadProgress = {
  totalChunks: 0,
  uploadedChunks: 0,
  uploadingChunks: 0,
  failedChunks: 0,
  percentage: 0,
  totalBytes: 0,
  uploadedBytes: 0,
  speed: 0,
  estimatedTimeRemaining: 0,
}

/**
 * Hook for resumable file uploads
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const {
 *     upload,
 *     pause,
 *     resume,
 *     cancel,
 *     state,
 *     progress,
 *     isUploading,
 *   } = useResumableUpload()
 *
 *   const handleUpload = async (file: File) => {
 *     await upload(file, '/api/upload', { category: 'images' })
 *   }
 *
 *   return (
 *     <div>
 *       <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />
 *       {isUploading && (
 *         <div>
 *           <p>Progress: {progress?.percentage.toFixed(2)}%</p>
 *           <p>Speed: {(progress?.speed / 1024 / 1024).toFixed(2)} MB/s</p>
 *           <button onClick={pause}>Pause</button>
 *           <button onClick={cancel}>Cancel</button>
 *         </div>
 *       )}
 *     </div>
 *   )
 * }
 * ```
 */
export function useResumableUpload(): UseResumableUploadResult {
  const [state, setState] = useState<UploadState>('idle')
  const [progress, setProgress] = useState<ResumableUploadProgress | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [result, setResult] = useState<any>(null)

  const uploadInstanceRef = useRef<ResumableUpload | null>(null)

  /**
   * Reset hook state
   */
  const reset = useCallback(() => {
    setState('idle')
    setProgress(null)
    setError(null)
    setResult(null)
    uploadInstanceRef.current = null
  }, [])

  /**
   * Start upload
   */
  const upload = useCallback(
    async (
      file: File,
      uploadUrl: string,
      additionalData: Record<string, any> = {},
      config: ResumableUploadConfig = {}
    ) => {
      try {
        // Reset state
        setState('uploading')
        setProgress(DEFAULT_PROGRESS)
        setError(null)
        setResult(null)

        logger.info('Starting resumable upload', {
          fileName: file.name,
          fileSize: file.size,
          uploadUrl,
        })

        // Create upload instance with callbacks
        const uploadInstance = new ResumableUpload(file, uploadUrl, additionalData, {
          ...config,
          onProgress: (prog) => {
            setProgress(prog)
            config.onProgress?.(prog)
          },
          onComplete: (res) => {
            setState('completed')
            setResult(res)
            setProgress((prev) =>
              prev ? { ...prev, percentage: 100, uploadedChunks: prev.totalChunks } : null
            )
            config.onComplete?.(res)
          },
          onError: (err) => {
            setState('error')
            setError(err)
            config.onError?.(err)
          },
          onChunkStart: config.onChunkStart,
          onChunkComplete: config.onChunkComplete,
          onChunkError: config.onChunkError,
        })

        uploadInstanceRef.current = uploadInstance

        // Start upload
        const uploadResult = await uploadInstance.start()

        logger.info('Upload completed', {
          fileName: file.name,
          result: uploadResult,
        })
      } catch (err: any) {
        logger.error('Upload failed', {
          fileName: file.name,
          error: err.message,
        })

        setState('error')
        setError(err)
      }
    },
    []
  )

  /**
   * Pause upload
   */
  const pause = useCallback(() => {
    if (uploadInstanceRef.current && state === 'uploading') {
      uploadInstanceRef.current.pause()
      setState('paused')
      logger.info('Upload paused')
    }
  }, [state])

  /**
   * Resume upload
   */
  const resume = useCallback(async () => {
    if (uploadInstanceRef.current && state === 'paused') {
      setState('uploading')
      logger.info('Resuming upload')

      try {
        await uploadInstanceRef.current.resume()
      } catch (err: any) {
        logger.error('Resume failed', { error: err.message })
        setState('error')
        setError(err)
      }
    }
  }, [state])

  /**
   * Cancel upload
   */
  const cancel = useCallback(() => {
    if (uploadInstanceRef.current) {
      uploadInstanceRef.current.cancel()
      setState('cancelled')
      logger.info('Upload cancelled')
    }
  }, [])

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      // Cleanup upload instance on unmount
      if (uploadInstanceRef.current) {
        uploadInstanceRef.current.cancel()
      }
    }
  }, [])

  return {
    state,
    progress,
    error,
    result,
    isUploading: state === 'uploading',
    isPaused: state === 'paused',
    isCompleted: state === 'completed',
    upload,
    pause,
    resume,
    cancel,
    reset,
  }
}

/**
 * Hook for managing multiple resumable uploads
 */
export interface MultiUploadItem {
  id: string
  file: File
  state: UploadState
  progress: ResumableUploadProgress | null
  error: Error | null
  result: any
}

export interface UseMultiResumableUploadResult {
  /**
   * All upload items
   */
  uploads: MultiUploadItem[]

  /**
   * Add new upload
   */
  addUpload: (
    file: File,
    uploadUrl: string,
    additionalData?: Record<string, any>,
    config?: ResumableUploadConfig
  ) => string

  /**
   * Remove upload
   */
  removeUpload: (id: string) => void

  /**
   * Pause upload
   */
  pauseUpload: (id: string) => void

  /**
   * Resume upload
   */
  resumeUpload: (id: string) => void

  /**
   * Cancel upload
   */
  cancelUpload: (id: string) => void

  /**
   * Clear all uploads
   */
  clearAll: () => void

  /**
   * Get upload by ID
   */
  getUpload: (id: string) => MultiUploadItem | undefined
}

/**
 * Hook for managing multiple resumable uploads
 */
export function useMultiResumableUpload(): UseMultiResumableUploadResult {
  const [uploads, setUploads] = useState<MultiUploadItem[]>([])
  const uploadInstancesRef = useRef<Map<string, ResumableUpload>>(new Map())

  /**
   * Add new upload
   */
  const addUpload = useCallback(
    (
      file: File,
      uploadUrl: string,
      additionalData: Record<string, any> = {},
      config: ResumableUploadConfig = {}
    ): string => {
      const id = `${file.name}_${Date.now()}`

      // Create new upload item
      const newUpload: MultiUploadItem = {
        id,
        file,
        state: 'uploading',
        progress: DEFAULT_PROGRESS,
        error: null,
        result: null,
      }

      setUploads((prev) => [...prev, newUpload])

      // Create upload instance
      const uploadInstance = new ResumableUpload(file, uploadUrl, additionalData, {
        ...config,
        onProgress: (prog) => {
          setUploads((prev) =>
            prev.map((u) => (u.id === id ? { ...u, progress: prog } : u))
          )
          config.onProgress?.(prog)
        },
        onComplete: (res) => {
          setUploads((prev) =>
            prev.map((u) =>
              u.id === id
                ? {
                    ...u,
                    state: 'completed' as UploadState,
                    result: res,
                    progress: u.progress
                      ? { ...u.progress, percentage: 100, uploadedChunks: u.progress.totalChunks }
                      : null,
                  }
                : u
            )
          )
          config.onComplete?.(res)
        },
        onError: (err) => {
          setUploads((prev) =>
            prev.map((u) =>
              u.id === id ? { ...u, state: 'error' as UploadState, error: err } : u
            )
          )
          config.onError?.(err)
        },
      })

      uploadInstancesRef.current.set(id, uploadInstance)

      // Start upload
      uploadInstance.start().catch((err) => {
        logger.error('Upload failed', { id, error: err.message })
      })

      return id
    },
    []
  )

  /**
   * Remove upload
   */
  const removeUpload = useCallback((id: string) => {
    setUploads((prev) => prev.filter((u) => u.id !== id))
    uploadInstancesRef.current.delete(id)
  }, [])

  /**
   * Pause upload
   */
  const pauseUpload = useCallback((id: string) => {
    const instance = uploadInstancesRef.current.get(id)
    if (instance) {
      instance.pause()
      setUploads((prev) =>
        prev.map((u) => (u.id === id ? { ...u, state: 'paused' as UploadState } : u))
      )
    }
  }, [])

  /**
   * Resume upload
   */
  const resumeUpload = useCallback((id: string) => {
    const instance = uploadInstancesRef.current.get(id)
    if (instance) {
      setUploads((prev) =>
        prev.map((u) => (u.id === id ? { ...u, state: 'uploading' as UploadState } : u))
      )
      instance.resume().catch((err) => {
        logger.error('Resume failed', { id, error: err.message })
      })
    }
  }, [])

  /**
   * Cancel upload
   */
  const cancelUpload = useCallback((id: string) => {
    const instance = uploadInstancesRef.current.get(id)
    if (instance) {
      instance.cancel()
      setUploads((prev) =>
        prev.map((u) => (u.id === id ? { ...u, state: 'cancelled' as UploadState } : u))
      )
    }
  }, [])

  /**
   * Clear all uploads
   */
  const clearAll = useCallback(() => {
    uploadInstancesRef.current.forEach((instance) => instance.cancel())
    uploadInstancesRef.current.clear()
    setUploads([])
  }, [])

  /**
   * Get upload by ID
   */
  const getUpload = useCallback(
    (id: string) => {
      return uploads.find((u) => u.id === id)
    },
    [uploads]
  )

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      uploadInstancesRef.current.forEach((instance) => instance.cancel())
      uploadInstancesRef.current.clear()
    }
  }, [])

  return {
    uploads,
    addUpload,
    removeUpload,
    pauseUpload,
    resumeUpload,
    cancelUpload,
    clearAll,
    getUpload,
  }
}
