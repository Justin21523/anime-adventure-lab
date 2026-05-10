# Intelligent Retry Mechanism

This document describes the intelligent retry mechanism implemented for API requests in the frontend application.

## Overview

The retry mechanism provides automatic retry with exponential backoff for failed API requests. It includes:

- **Exponential Backoff**: Delays between retries increase exponentially
- **Jitter**: Random jitter to prevent thundering herd problem
- **Configurable Retry Logic**: Customize which errors to retry
- **Rate Limit Respect**: Honors `Retry-After` headers from the server
- **Visual Feedback**: Toast notifications for retry attempts
- **Smart Defaults**: Only retries on network errors and 5xx server errors

## Architecture

### Core Components

1. **`src/lib/retry.ts`**: Retry logic and configuration
2. **`src/api/client.ts`**: Enhanced API client with retry support
3. **`src/hooks/useRetryNotification.ts`**: Hook for visual retry feedback

## Usage

### Basic Usage with Default Configuration

By default, all API requests will automatically retry on network errors and 5xx server errors:

```typescript
import { apiGet } from '@/api/client'

// This will automatically retry on network errors and 5xx errors
const data = await apiGet<User>('/users/123')
```

Default retry configuration:
- **Max Retries**: 3
- **Base Delay**: 1000ms (1 second)
- **Max Delay**: 30000ms (30 seconds)
- **Backoff Multiplier**: 2 (exponential)
- **Jitter**: Enabled

### Custom Retry Configuration

You can customize retry behavior per request:

```typescript
import { apiGet } from '@/api/client'

const data = await apiGet<User>('/users/123', {
  retry: {
    maxRetries: 5,
    baseDelay: 2000,
    maxDelay: 60000,
    backoffMultiplier: 3,
  },
})
```

### Disable Retry for Specific Requests

Some requests should not be retried (e.g., POST requests that create resources):

```typescript
import { apiPost } from '@/api/client'

// Disable retry explicitly
const result = await apiPost<Order>('/orders', orderData, {
  retry: false,
})
```

### Custom Retry Logic

You can provide custom logic to determine which errors should be retried:

```typescript
import { apiGet } from '@/api/client'
import type { AxiosError } from 'axios'

const data = await apiGet<User>('/users/123', {
  retry: {
    shouldRetry: (error: AxiosError) => {
      // Only retry on 503 Service Unavailable
      return error.response?.status === 503
    },
  },
})
```

### With Visual Feedback (Toast Notifications)

Use the `useRetryNotification` hook to show toast notifications during retries:

```typescript
import { apiGet } from '@/api/client'
import { useRetryNotification } from '@/hooks/useRetryNotification'

function MyComponent() {
  const { createRetryConfigWithNotification, showRetrySuccess, showRetryFailure } =
    useRetryNotification()

  const handleFetch = async () => {
    try {
      const data = await apiGet<User>('/users/123', {
        retry: createRetryConfigWithNotification({
          maxRetries: 3,
          baseDelay: 1000,
        }),
      })
      // Optional: Show success notification
      // showRetrySuccess(1)
      return data
    } catch (error) {
      showRetryFailure(3, error.message)
      throw error
    }
  }

  return <button onClick={handleFetch}>Fetch User</button>
}
```

### File Uploads

File uploads have retry **disabled by default** to prevent duplicate uploads:

```typescript
import { apiUploadFile } from '@/api/client'

// Retry is disabled by default for file uploads
const result = await apiUploadFile('/upload', file, { category: 'images' }, (progress) => {
  console.log(`Upload progress: ${progress}%`)
})

// Enable retry explicitly if needed
const result = await apiUploadFile(
  '/upload',
  file,
  { category: 'images' },
  (progress) => {
    console.log(`Upload progress: ${progress}%`)
  },
  {
    retry: {
      maxRetries: 2,
      baseDelay: 5000,
    },
  }
)
```

## Retry Logic Details

### Default Retry Conditions

By default, requests are retried when:

1. **Network Errors**: No response received (connection failed, timeout, etc.)
2. **5xx Server Errors**: Server errors (500-599)
3. **408 Request Timeout**
4. **429 Too Many Requests**

Requests are **NOT retried** when:

1. **4xx Client Errors** (except 408, 429): Bad request, unauthorized, not found, etc.
2. **Request Cancelled**: User cancelled the request
3. **Successful Responses**: 2xx, 3xx status codes

### Exponential Backoff Calculation

The delay between retries is calculated using exponential backoff:

```
delay = baseDelay × (backoffMultiplier ^ attemptNumber)
```

With jitter enabled (default):
```
delay = random(0, delay)
```

Example with default settings:
- Attempt 1: 0-1000ms (random)
- Attempt 2: 0-2000ms (random)
- Attempt 3: 0-4000ms (random)

### Retry-After Header Support

The retry mechanism respects the `Retry-After` header from the server:

```typescript
import { createRetryConfigWithRateLimitRespect } from '@/lib/retry'

// Automatically respects Retry-After headers
const config = createRetryConfigWithRateLimitRespect({
  maxRetries: 5,
})
```

## Advanced Usage

### Retry with Callback

Monitor retry attempts with callbacks:

```typescript
import { apiGet } from '@/api/client'

const data = await apiGet<User>('/users/123', {
  retry: {
    onRetry: (attemptNumber, error, delay) => {
      console.log(`Retry attempt ${attemptNumber}`)
      console.log(`Error: ${error.message}`)
      console.log(`Waiting ${delay}ms before retry`)

      // Track retry metrics
      analytics.track('api_retry', {
        attempt: attemptNumber,
        url: error.config?.url,
        status: error.response?.status,
      })
    },
  },
})
```

### Direct Use of Retry Function

You can use the retry function directly for non-API operations:

```typescript
import { retryWithBackoff } from '@/lib/retry'

const result = await retryWithBackoff(
  async () => {
    // Your async operation
    const response = await fetch('https://api.example.com/data')
    if (!response.ok) throw new Error('Fetch failed')
    return response.json()
  },
  {
    maxRetries: 3,
    baseDelay: 1000,
    onRetry: (attempt, error) => {
      console.log(`Retry ${attempt}: ${error.message}`)
    },
  }
)
```

## Best Practices

1. **Use Default Retry for GET Requests**: Safe to retry, improves reliability
2. **Disable Retry for POST/PUT/DELETE**: Can cause duplicate operations
3. **Enable Jitter**: Prevents thundering herd when many clients retry simultaneously
4. **Respect Server Rate Limits**: Use `createRetryConfigWithRateLimitRespect`
5. **Add Visual Feedback**: Use `useRetryNotification` for user-facing operations
6. **Log Retry Attempts**: Track retries in analytics for monitoring
7. **Set Appropriate Max Retries**: Too many retries can delay error feedback
8. **Consider Idempotency**: If operations are idempotent, retry is safer

## Logging

All retry attempts are automatically logged using the structured logger:

```
[INFO] Retrying request
  - attempt: 2
  - maxRetries: 3
  - delay: 2000
  - error: "Network Error"
  - status: undefined
```

Failed retries are logged as warnings:

```
[WARN] Retry exhausted or error not retryable
  - attempt: 4
  - maxRetries: 3
  - shouldRetry: false
  - error: "404 Not Found"
```

## Examples

### Example 1: Retry with Custom Max Retries

```typescript
const users = await apiGet<User[]>('/users', {
  retry: {
    maxRetries: 5,
  },
})
```

### Example 2: Longer Delays Between Retries

```typescript
const data = await apiGet<Data>('/data', {
  retry: {
    baseDelay: 5000, // Start with 5 second delay
    backoffMultiplier: 3, // Triple the delay each time
  },
})
```

### Example 3: Retry Only on Specific Status Codes

```typescript
const data = await apiGet<Data>('/data', {
  retry: {
    shouldRetry: (error) => {
      // Only retry on 503 or network errors
      return !error.response || error.response.status === 503
    },
  },
})
```

### Example 4: React Query Integration

```typescript
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'

function useUser(userId: string) {
  return useQuery({
    queryKey: ['user', userId],
    queryFn: () =>
      apiGet<User>(`/users/${userId}`, {
        retry: {
          maxRetries: 3,
          baseDelay: 1000,
        },
      }),
    // React Query has its own retry mechanism, so you might want to disable one
    retry: false, // Disable React Query's retry
  })
}
```

## Testing

### Testing Retry Logic

```typescript
import { retryWithBackoff, defaultShouldRetry } from '@/lib/retry'
import { describe, it, expect, vi } from 'vitest'

describe('Retry Mechanism', () => {
  it('should retry on network error', async () => {
    let attempts = 0
    const fn = vi.fn(async () => {
      attempts++
      if (attempts < 3) {
        throw new Error('Network Error')
      }
      return 'success'
    })

    const result = await retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelay: 100,
    })

    expect(result).toBe('success')
    expect(attempts).toBe(3)
  })

  it('should not retry on 404', () => {
    const error = {
      response: { status: 404 },
    } as any

    expect(defaultShouldRetry(error)).toBe(false)
  })

  it('should retry on 503', () => {
    const error = {
      response: { status: 503 },
    } as any

    expect(defaultShouldRetry(error)).toBe(true)
  })
})
```

## Performance Considerations

- **Total Request Time**: With 3 retries and exponential backoff, a failing request could take up to `baseDelay × (2^0 + 2^1 + 2^2) = 7 seconds` plus request time
- **Concurrent Retries**: If multiple requests fail simultaneously, jitter helps spread out retry attempts
- **Memory**: Retry state is not stored; each request manages its own retry logic
- **Network Load**: Retries increase network traffic; set appropriate `maxRetries` to balance reliability and load

## Troubleshooting

### Issue: Requests are not being retried

**Solution**: Check if the error type is in the default retry conditions. You may need to provide a custom `shouldRetry` function.

### Issue: Too many retries

**Solution**: Reduce `maxRetries` or adjust `shouldRetry` logic to be more selective.

### Issue: Delays are too long

**Solution**: Reduce `baseDelay` or `backoffMultiplier`, or set a lower `maxDelay`.

### Issue: Toast notifications not showing

**Solution**: Ensure `useRetryNotification` is used within a component that has access to the toast context (inside `<Toaster />` provider).

## Related Files

- `src/lib/retry.ts` - Retry logic and utilities
- `src/api/client.ts` - API client with retry support
- `src/hooks/useRetryNotification.ts` - Retry notification hook
- `src/lib/api-error.ts` - API error handling
- `src/utils/logger.ts` - Structured logging
