import { useState, useEffect, useRef } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollText, ChevronRight, FastForward, MousePointer2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion } from 'framer-motion'

interface DialogBoxProps {
  name?: string
  text: string
  onComplete?: () => void
  isTyping?: boolean
  speed?: number // characters per second
  onShowHistory?: () => void
  showHistoryButton?: boolean
}

export function DialogBox({
  name,
  text,
  onComplete,
  speed = 40,
  onShowHistory,
  showHistoryButton = true
}: DialogBoxProps) {
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const textRef = useRef(text)

  useEffect(() => {
    // Reset typing when text changes
    textRef.current = text
    setDisplayedText('')
    setIsTyping(true)
    
    let currentIndex = 0
    const interval = 1000 / speed

    if (timerRef.current) clearInterval(timerRef.current)

    timerRef.current = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayedText(text.slice(0, currentIndex + 1))
        currentIndex++
      } else {
        if (timerRef.current) clearInterval(timerRef.current)
        setIsTyping(false)
        onComplete?.()
      }
    }, interval)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [text, speed, onComplete])

  const skipTyping = () => {
    if (isTyping) {
      if (timerRef.current) clearInterval(timerRef.current)
      setDisplayedText(text)
      setIsTyping(false)
      onComplete?.()
    }
  }

  return (
    <motion.div 
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="relative w-full max-w-4xl mx-auto px-4 pb-12 group"
    >
      <Card 
        className={cn(
          "bg-slate-950/85 backdrop-blur-xl border-indigo-500/30 shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden cursor-pointer min-h-[160px] flex flex-col transition-all",
          isTyping && "ring-1 ring-indigo-500/30",
          !isTyping && "hover:border-indigo-400/50"
        )}
        onClick={skipTyping}
      >
        {/* Name Tag */}
        {name && (
          <motion.div 
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="absolute -top-3 left-8 px-6 py-1.5 bg-gradient-to-r from-indigo-600 to-violet-600 text-white text-sm font-bold rounded-sm shadow-[0_5px_15px_rgba(79,70,229,0.4)] border border-indigo-400/50 z-10"
            style={{ clipPath: "polygon(0% 0%, 100% 0%, 95% 100%, 5% 100%)" }}
          >
            {name}
          </motion.div>
        )}

        <div className="p-8 pt-10 flex-1 relative">
          <p className="text-xl text-slate-100 leading-relaxed font-medium selection:bg-indigo-500/40 tracking-wide drop-shadow-sm">
            {displayedText}
            {isTyping && (
              <motion.span 
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 0.8, repeat: Infinity }}
                className="inline-block w-2.5 h-6 ml-1 bg-indigo-400 align-middle" 
              />
            )}
          </p>
          
          {/* Waiting for click indicator */}
          {!isTyping && (
            <motion.div 
              animate={{ y: [0, 5, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="absolute bottom-4 right-6 text-indigo-400"
            >
              <MousePointer2 className="w-5 h-5 fill-current opacity-80" />
            </motion.div>
          )}
        </div>

        {/* Action Bar */}
        <div className="px-6 py-3 border-t border-indigo-900/30 bg-indigo-950/20 flex items-center justify-between">
          <div className="flex gap-4">
            {showHistoryButton && (
              <Button 
                variant="ghost" 
                size="sm" 
                className="h-8 text-slate-400 hover:text-indigo-300 hover:bg-indigo-500/10 transition-colors px-3"
                onClick={(e) => {
                  e.stopPropagation()
                  onShowHistory?.()
                }}
              >
                <ScrollText className="w-4 h-4 mr-2" />
                回顧紀錄
              </Button>
            )}
          </div>
          
          <div className="flex items-center text-slate-500 text-xs font-bold tracking-widest uppercase gap-4">
            {isTyping ? (
              <span className="flex items-center gap-2 text-slate-400 animate-pulse">
                <FastForward className="w-3.5 h-3.5" />
                正在播放
              </span>
            ) : (
              <span className="flex items-center gap-2 text-indigo-400">
                點擊繼續
                <ChevronRight className="w-4 h-4" />
              </span>
            )}
          </div>
        </div>
      </Card>
      
      {/* Box glow effect */}
      <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 blur-xl -z-10 rounded-2xl opacity-50" />
    </motion.div>
  )
}
