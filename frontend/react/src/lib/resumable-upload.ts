import { logger } from '@/utils/logger'
import { apiPost } from '@/api/client'

/**
 * Configuration for resumable upload
 */
export interface ResumableUploadConfig {
  /**
   * Size of each chunk in bytes
   * @default 1MB (1024 * 1024)
   */
  chunkSize?: number

  /**
   * Maximum concurrent chunk uploads
   * @default 3
   */
  maxConcurrent?: number

  /**
   * Whether to enable upload persistence (saves state to localStorage)
   * @default true
   */
  enablePersistence?: boolean

  /**
   * Key prefix for localStorage
   * @default 'resumable_upload_'
   */
  storageKeyPrefix?: string

  /**
   * Maximum retries per chunk
   * @default 3
   */
  maxRetries?: number

  /**
   * Custom headers to include in chunk upload requests
   */
  headers?: Record<string, string>

  /**
   * Callback for overall progress
   */
  onProgress?: (progress: ResumableUploadProgress) => void

  /**
   * Callback when upload completes
   */
  onComplete?: (result: any) => void

  /**
   * Callback when upload fails
   */
  onError?: (error: Error) => void

  /**
   * Callback when chunk upload starts
   */
  onChunkStart?: (chunkIndex: number, totalChunks: number) => void

  /**
   * Callback when chunk upload completes
   */
  onChunkComplete?: (chunkIndex: number, totalChunks: number) => void

  /**
   * Callback when chunk upload fails
   */
  onChunkError?: (chunkIndex: number, error: Error) => void
}

/**
 * Progress information for resumable upload
 */
export interface ResumableUploadProgress {
  /**
   * Total number of chunks
   */
  totalChunks: number

  /**
   * Number of chunks uploaded
   */
  uploadedChunks: number

  /**
   * Number of chunks currently uploading
   */
  uploadingChunks: number

  /**
   * Number of chunks failed
   */
  failedChunks: number

  /**
   * Overall progress percentage (0-100)
   */
  percentage: number

  /**
   * Total bytes
   */
  totalBytes: number

  /**
   * Bytes uploaded
   */
  uploadedBytes: number

  /**
   * Upload speed in bytes per second
   */
  speed: number

  /**
   * Estimated time remaining in seconds
   */
  estimatedTimeRemaining: number
}

/**
 * Chunk metadata
 */
interface ChunkMetadata {
  index: number
  start: number
  end: number
  size: number
  uploaded: boolean
  uploading: boolean
  failed: boolean
  retries: number
}

/**
 * Upload session state
 */
interface UploadSessionState {
  uploadId: string
  fileName: string
  fileSize: number
  fileType: string
  totalChunks: number
  chunks: ChunkMetadata[]
  startTime: number
  lastUpdateTime: number
  uploadedBytes: number
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<ResumableUploadConfig> = {
  chunkSize: 1024 * 1024, // 1MB
  maxConcurrent: 3,
  enablePersistence: true,
  storageKeyPrefix: 'resumable_upload_',
  maxRetries: 3,
  headers: {},
  onProgress: () => {},
  onComplete: () => {},
  onError: () => {},
  onChunkStart: () => {},
  onChunkComplete: () => {},
  onChunkError: () => {},
}

/**
 * Resumable upload manager
 */
export class ResumableUpload {
  private config: Required<ResumableUploadConfig>
  private file: File
  private uploadUrl: string
  private additionalData: Record<string, any>
  private uploadId: string
  private state: UploadSessionState
  private abortController: AbortController
  private isPaused: boolean = false
  private isCancelled: boolean = false

  constructor(
    file: File,
    uploadUrl: string,
    additionalData: Record<string, any> = {},
    config: ResumableUploadConfig = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.file = file
    this.uploadUrl = uploadUrl
    this.additionalData = additionalData
    this.uploadId = this.generateUploadId()
    this.abortController = new AbortController()

    // Initialize or restore state
    this.state = this.loadState() || this.initializeState()
  }

  /**
   * Generate unique upload ID
   */
  private generateUploadId(): string {
    return `${this.file.name}_${this.file.size}_${Date.now()}`
  }

  /**
   * Initialize upload state
   */
  private initializeState(): UploadSessionState {
    const totalChunks = Math.ceil(this.file.size / this.config.chunkSize)
    const chunks: ChunkMetadata[] = []

    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.config.chunkSize
      const end = Math.min(start + this.config.chunkSize, this.file.size)

      chunks.push({
        index: i,
        start,
        end,
        size: end - start,
        uploaded: false,
        uploading: false,
        failed: false,
        retries: 0,
      })
    }

    const state: UploadSessionState = {
      uploadId: this.uploadId,
      fileName: this.file.name,
      fileSize: this.file.size,
      fileType: this.file.type,
      totalChunks,
      chunks,
      startTime: Date.now(),
      lastUpdateTime: Date.now(),
      uploadedBytes: 0,
    }

    this.saveState(state)
    return state
  }

  /**
   * Save state to localStorage
   */
  private saveState(state: UploadSessionState): void {
    if (!this.config.enablePersistence) return

    try {
      const key = this.config.storageKeyPrefix + this.uploadId
      localStorage.setItem(key, JSON.stringify(state))
    } catch (error: any) {
      logger.warn('Failed to save upload state', { error })
    }
  }

  /**
   * Load state from localStorage
   */
  private loadState(): UploadSessionState | null {
    if (!this.config.enablePersistence) return null

    try {
      const key = this.config.storageKeyPrefix + this.uploadId
      const data = localStorage.getItem(key)
      if (!data) return null

      const state = JSON.parse(data) as UploadSessionState

      // Verify state matches current file
      if (
        state.fileName === this.file.name &&
        state.fileSize === this.file.size &&
        state.fileType === this.file.type
      ) {
        logger.info('Resuming upload from saved state', {
          uploadId: this.uploadId,
          uploadedChunks: state.chunks.filter((c) => c.uploaded).length,
          totalChunks: state.totalChunks,
        })
        return state
      }
    } catch (error: any) {
      logger.warn('Failed to load upload state', { error })
    }

    return null
  }

  /**
   * Clear saved state
   */
  private clearState(): void {
    if (!this.config.enablePersistence) return

    try {
      const key = this.config.storageKeyPrefix + this.uploadId
      localStorage.removeItem(key)
    } catch (error: any) {
      logger.warn('Failed to clear upload state', { error })
    }
  }

  /**
   * Calculate current progress
   */
  private calculateProgress(): ResumableUploadProgress {
    const uploadedChunks = this.state.chunks.filter((c) => c.uploaded).length
    const uploadingChunks = this.state.chunks.filter((c) => c.uploading).length
    const failedChunks = this.state.chunks.filter((c) => c.failed).length

    const percentage = (this.state.uploadedBytes / this.file.size) * 100

    const elapsedTime = (Date.now() - this.state.startTime) / 1000 // seconds
    const speed = elapsedTime > 0 ? this.state.uploadedBytes / elapsedTime : 0
    const remainingBytes = this.file.size - this.state.uploadedBytes
    const estimatedTimeRemaining = speed > 0 ? remainingBytes / speed : 0

    return {
      totalChunks: this.state.totalChunks,
      uploadedChunks,
      uploadingChunks,
      failedChunks,
      percentage: Math.min(100, Math.max(0, percentage)),
      totalBytes: this.file.size,
      uploadedBytes: this.state.uploadedBytes,
      speed,
      estimatedTimeRemaining,
    }
  }

  /**
   * Upload a single chunk
   */
  private async uploadChunk(chunk: ChunkMetadata): Promise<void> {
    const { index, start, end } = chunk

    try {
      // Mark chunk as uploading
      chunk.uploading = true
      chunk.failed = false
      this.saveState(this.state)
      this.config.onChunkStart(index, this.state.totalChunks)

      // Extract chunk data
      const chunkBlob = this.file.slice(start, end)
      const formData = new FormData()
      formData.append('file', chunkBlob, this.file.name)
      formData.append('chunkIndex', String(index))
      formData.append('totalChunks', String(this.state.totalChunks))
      formData.append('uploadId', this.uploadId)
      formData.append('fileName', this.file.name)
      formData.append('fileSize', String(this.file.size))

      // Add additional data
      Object.entries(this.additionalData).forEach(([key, value]) => {
        formData.append(key, String(value))
      })

      // Upload chunk with retry
      await apiPost(this.uploadUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          ...this.config.headers,
        },
        retry: {
          maxRetries: this.config.maxRetries,
          baseDelay: 1000,
          onRetry: (attempt) => {
            logger.info('Retrying chunk upload', {
              chunkIndex: index,
              attempt,
              maxRetries: this.config.maxRetries,
            })
          },
        },
      })

      // Mark chunk as uploaded
      chunk.uploaded = true
      chunk.uploading = false
      chunk.failed = false
      this.state.uploadedBytes += chunk.size
      this.state.lastUpdateTime = Date.now()
      this.saveState(this.state)

      this.config.onChunkComplete(index, this.state.totalChunks)

      logger.debug('Chunk uploaded successfully', {
        chunkIndex: index,
        uploadedBytes: this.state.uploadedBytes,
        totalBytes: this.file.size,
      })
    } catch (error: any) {
      chunk.uploading = false
      chunk.failed = true
      chunk.retries++

      logger.error('Chunk upload failed', {
        chunkIndex: index,
        error: error.message,
        retries: chunk.retries,
      })

      this.config.onChunkError(index, error)
      throw error
    }
  }

  /**
   * Upload all chunks with concurrency control
   */
  private async uploadChunks(): Promise<void> {
    const pendingChunks = this.state.chunks.filter((c) => !c.uploaded && !c.failed)

    logger.info('Starting chunk uploads', {
      totalChunks: this.state.totalChunks,
      pendingChunks: pendingChunks.length,
      maxConcurrent: this.config.maxConcurrent,
    })

    // Upload chunks with concurrency control
    const uploadQueue: Promise<void>[] = []
    let currentIndex = 0

    const uploadNext = async (): Promise<void> => {
      while (currentIndex < pendingChunks.length && !this.isPaused && !this.isCancelled) {
        const chunk = pendingChunks[currentIndex++]

        if (chunk.uploaded || chunk.uploading) continue

        try {
          await this.uploadChunk(chunk)

          // Update progress
          const progress = this.calculateProgress()
          this.config.onProgress(progress)
        } catch (error: any) {
          // Chunk upload failed, will be handled by retry logic
        }
      }
    }

    // Start concurrent uploads
    for (let i = 0; i < this.config.maxConcurrent; i++) {
      uploadQueue.push(uploadNext())
    }

    await Promise.all(uploadQueue)
  }

  /**
   * Start or resume upload
   */
  async start(): Promise<any> {
    try {
      this.isPaused = false
      this.isCancelled = false

      logger.info('Starting resumable upload', {
        uploadId: this.uploadId,
        fileName: this.file.name,
        fileSize: this.file.size,
        totalChunks: this.state.totalChunks,
      })

      // Upload all chunks
      await this.uploadChunks()

      // Check if all chunks uploaded successfully
      const failedChunks = this.state.chunks.filter((c) => c.failed)
      if (failedChunks.length > 0) {
        throw new Error(
          `Upload failed: ${failedChunks.length} chunks failed after maximum retries`
        )
      }

      // Finalize upload
      const result = await this.finalizeUpload()

      // Clear state
      this.clearState()

      logger.info('Upload completed successfully', {
        uploadId: this.uploadId,
        fileName: this.file.name,
      })

      this.config.onComplete(result)
      return result
    } catch (error: any) {
      logger.error('Upload failed', {
        uploadId: this.uploadId,
        error: error.message,
      })

      this.config.onError(error)
      throw error
    }
  }

  /**
   * Pause upload
   */
  pause(): void {
    this.isPaused = true
    logger.info('Upload paused', { uploadId: this.uploadId })
  }

  /**
   * Resume upload
   */
  async resume(): Promise<any> {
    logger.info('Resuming upload', { uploadId: this.uploadId })
    return this.start()
  }

  /**
   * Cancel upload
   */
  cancel(): void {
    this.isCancelled = true
    this.abortController.abort()
    this.clearState()
    logger.info('Upload cancelled', { uploadId: this.uploadId })
  }

  /**
   * Get current progress
   */
  getProgress(): ResumableUploadProgress {
    return this.calculateProgress()
  }

  /**
   * Finalize upload (notify server that all chunks are uploaded)
   */
  private async finalizeUpload(): Promise<any> {
    logger.info('Finalizing upload', { uploadId: this.uploadId })

    // Send finalize request to server
    const result = await apiPost(`${this.uploadUrl}/finalize`, {
      uploadId: this.uploadId,
      fileName: this.file.name,
      fileSize: this.file.size,
      totalChunks: this.state.totalChunks,
      ...this.additionalData,
    })

    return result
  }
}

/**
 * Helper function to create and start a resumable upload
 */
export async function uploadFileResumable(
  file: File,
  uploadUrl: string,
  additionalData?: Record<string, any>,
  config?: ResumableUploadConfig
): Promise<any> {
  const upload = new ResumableUpload(file, uploadUrl, additionalData, config)
  return upload.start()
}
