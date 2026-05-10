import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

interface CharacterSpriteProps {
  imageUrl?: string
  name: string
  position?: 'left' | 'center' | 'right'
  isActive?: boolean
  isSpeaking?: boolean
  emotion?: 'happy' | 'sad' | 'angry' | 'surprised' | 'neutral'
}

export function CharacterSprite({
  imageUrl,
  name,
  position = 'center',
  isActive = true,
  isSpeaking = false,
  emotion = 'neutral'
}: CharacterSpriteProps) {
  // Define animation variants based on emotion
  const emotionVariants = {
    neutral: {},
    happy: {
      y: [0, -20, 0],
      transition: { duration: 0.5, repeat: isSpeaking ? Infinity : 0, repeatDelay: 2 }
    },
    angry: {
      x: [0, -5, 5, -5, 5, 0],
      transition: { duration: 0.2, repeat: isSpeaking ? Infinity : 0, repeatDelay: 1 }
    },
    surprised: {
      scale: [1, 1.1, 1],
      transition: { duration: 0.3 }
    },
    sad: {
      y: [0, 5, 0],
      opacity: [1, 0.8, 1],
      transition: { duration: 2, repeat: Infinity }
    }
  }

  const baseContent = !imageUrl ? (
    <div className={cn(
      "flex flex-col items-center transition-all duration-500",
      position === 'left' ? "translate-x-[-20%]" : position === 'right' ? "translate-x-[20%]" : "",
      !isActive && "opacity-0 scale-95",
      isActive && "opacity-100 scale-100",
      isSpeaking ? "brightness-110 drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]" : "brightness-75 grayscale-[20%]"
    )}>
      <div className="w-64 h-96 bg-slate-800/40 backdrop-blur-sm border-2 border-slate-700/50 rounded-2xl flex items-center justify-center relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 to-transparent" />
        <span className="text-slate-500 font-bold text-xl uppercase tracking-widest">{name}</span>
        <div className="absolute bottom-4 left-0 right-0 text-center">
           <span className="text-[10px] text-slate-400 font-mono uppercase">No Sprite Loaded</span>
        </div>
      </div>
    </div>
  ) : (
    <motion.div 
      animate={emotion !== 'neutral' ? emotion : (isSpeaking ? { scale: 1.05, filter: "brightness(1.1)" } : { scale: 1, filter: "brightness(0.8)" })}
      variants={emotionVariants}
      className={cn(
        "relative transition-all duration-700 ease-out transform-gpu",
        position === 'left' ? "translate-x-[-15%]" : position === 'right' ? "translate-x-[15%]" : "",
        !isActive && "opacity-0 translate-y-8 scale-95",
        isActive && "opacity-100 translate-y-0 scale-100",
        isSpeaking ? "z-20 brightness-110 drop-shadow-[0_10px_25px_rgba(0,0,0,0.5)]" : "z-10 brightness-75 grayscale-[10%] blur-[0.5px]"
      )}
    >
      <img 
        src={imageUrl} 
        alt={name}
        className={cn(
          "max-h-[70vh] w-auto object-contain transition-all duration-500",
          isSpeaking && "animate-in fade-in zoom-in-95 duration-300"
        )}
      />
      
      {/* Speaking Indicator Glow */}
      {isSpeaking && (
        <div className="absolute inset-0 bg-indigo-500/10 blur-3xl -z-10 rounded-full animate-pulse" />
      )}
    </motion.div>
  )

  return (
    <AnimatePresence>
      {isActive && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          layout
        >
          {baseContent}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
