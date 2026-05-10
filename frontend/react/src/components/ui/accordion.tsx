import * as React from 'react'
import { cn } from '@/lib/utils'

interface AccordionContextValue {
  openItems: string[]
  toggleItem: (value: string) => void
  multiple?: boolean
}

const AccordionContext = React.createContext<AccordionContextValue | undefined>(undefined)

const useAccordionContext = () => {
  const context = React.useContext(AccordionContext)
  if (!context) {
    throw new Error('Accordion components must be used within an Accordion provider')
  }
  return context
}

interface AccordionProps {
  children: React.ReactNode
  type?: 'single' | 'multiple'
  defaultValue?: string | string[]
  className?: string
}

/**
 * Accordion component for collapsible content sections
 */
export function Accordion({
  children,
  type = 'single',
  defaultValue,
  className,
}: AccordionProps) {
  const [openItems, setOpenItems] = React.useState<string[]>(() => {
    if (!defaultValue) return []
    return Array.isArray(defaultValue) ? defaultValue : [defaultValue]
  })

  const toggleItem = (value: string) => {
    if (type === 'single') {
      setOpenItems((prev) => (prev.includes(value) ? [] : [value]))
    } else {
      setOpenItems((prev) =>
        prev.includes(value)
          ? prev.filter((item) => item !== value)
          : [...prev, value]
      )
    }
  }

  return (
    <AccordionContext.Provider value={{ openItems, toggleItem, multiple: type === 'multiple' }}>
      <div className={cn('w-full', className)}>{children}</div>
    </AccordionContext.Provider>
  )
}

interface AccordionItemProps {
  value: string
  children: React.ReactNode
  className?: string
}

export function AccordionItem({ value, children, className }: AccordionItemProps) {
  return (
    <div className={cn('border-b border-slate-700', className)} data-value={value}>
      {children}
    </div>
  )
}

interface AccordionTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
}

export const AccordionTrigger = React.forwardRef<HTMLButtonElement, AccordionTriggerProps>(
  ({ children, className, ...props }, ref) => {
    const { openItems, toggleItem } = useAccordionContext()
    const value = React.useContext(AccordionItemContext)

    if (!value) {
      throw new Error('AccordionTrigger must be used within AccordionItem')
    }

    const isOpen = openItems.includes(value)

    return (
      <button
        ref={ref}
        type="button"
        onClick={() => toggleItem(value)}
        className={cn(
          'flex w-full items-center justify-between py-4 font-medium transition-all hover:underline',
          'text-left text-slate-200',
          className
        )}
        {...props}
      >
        {children}
        <svg
          className={cn(
            'h-4 w-4 shrink-0 transition-transform duration-200',
            isOpen && 'rotate-180'
          )}
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
    )
  }
)

AccordionTrigger.displayName = 'AccordionTrigger'

const AccordionItemContext = React.createContext<string | undefined>(undefined)

interface AccordionContentProps {
  children: React.ReactNode
  className?: string
}

export function AccordionContent({ children, className }: AccordionContentProps) {
  const { openItems } = useAccordionContext()
  const parentItem = React.useContext(AccordionContext)

  // Find the value from parent AccordionItem
  const itemRef = React.useRef<HTMLDivElement>(null)
  const value = itemRef.current?.parentElement?.getAttribute('data-value')

  const isOpen = value && openItems.includes(value)

  return (
    <div ref={itemRef}>
      {isOpen && (
        <div
          className={cn(
            'pb-4 pt-0 text-sm text-slate-400',
            'animate-in slide-in-from-top-2',
            className
          )}
        >
          {children}
        </div>
      )}
    </div>
  )
}
