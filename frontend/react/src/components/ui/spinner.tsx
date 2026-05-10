import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg' | 'xl'
}

/**
 * Loading spinner component
 */
export const Spinner = React.forwardRef<HTMLDivElement, SpinnerProps>(
  ({ size = 'md', className, ...props }, ref) => {
    const sizeStyles = {
      sm: 'w-4 h-4 border-2',
      md: 'w-6 h-6 border-2',
      lg: 'w-8 h-8 border-3',
      xl: 'w-12 h-12 border-4',
    }

    return (
      <div
        ref={ref}
        className={cn('inline-block', className)}
        {...props}
      >
        <div
          className={cn(
            'rounded-full border-slate-700 border-t-primary animate-spin',
            sizeStyles[size]
          )}
        />
      </div>
    )
  }
)

Spinner.displayName = 'Spinner'

interface LoadingProps {
  text?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

/**
 * Loading component with spinner and text
 */
export function Loading({ text = 'Loading...', size = 'md', className }: LoadingProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <Spinner size={size} />
      {text && <p className="text-sm text-slate-400">{text}</p>}
    </div>
  )
}
