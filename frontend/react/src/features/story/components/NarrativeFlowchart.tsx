import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { apiGet } from '@/api/client'
import { GitBranch, ChevronDown, Map, Eye } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

interface FlowNode {
  id: string
  type: string
  label: string
  summary?: string
  turn: number
  image_url?: string
}

interface FlowEdge {
  id: string
  source: string
  target: string
  label?: string
}

interface FlowData {
  nodes: FlowNode[]
  edges: FlowEdge[]
}

interface NarrativeFlowchartProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
}

export function NarrativeFlowchart({ open, onOpenChange, sessionId }: NarrativeFlowchartProps) {
  const [data, setData] = useState<FlowData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (open) {
      fetchFlowchart()
    }
  }, [open, sessionId])

  const fetchFlowchart = async () => {
    setLoading(true)
    try {
      const flowchart = await apiGet<FlowData>(`/story/session/${sessionId}/flowchart`)
      setData(flowchart)
    } catch (error) {
      console.error('Failed to fetch flowchart:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl h-[85vh] flex flex-col bg-slate-950 border-indigo-900/50 text-slate-100 shadow-2xl overflow-hidden p-0">
        <DialogHeader className="p-6 border-b border-white/5 bg-indigo-950/20">
          <DialogTitle className="text-2xl font-bold flex items-center gap-3 text-white">
            <GitBranch className="text-indigo-400 w-7 h-7" />
            劇情軌跡觀測站
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-auto p-12 bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.05)_0%,transparent_100%)]">
          {loading ? (
            <div className="h-full flex items-center justify-center">
               <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                  <Map className="w-12 h-12 text-indigo-500/50" />
               </motion.div>
            </div>
          ) : !data || data.nodes.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-500 italic">
               尚無劇情數據
            </div>
          ) : (
            <div className="flex flex-col items-center gap-0">
              {data.nodes.map((node, idx) => (
                <React.Fragment key={node.id}>
                  {/* Node */}
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.1 }}
                    className={cn(
                      "relative group flex items-center gap-6 p-4 rounded-2xl border transition-all duration-500 w-full max-w-xl",
                      node.type === 'start' 
                        ? "bg-indigo-600/20 border-indigo-500/50 shadow-[0_0_20px_rgba(79,70,229,0.2)]" 
                        : "bg-slate-900/40 border-slate-800 hover:border-indigo-500/30 hover:bg-slate-900/60"
                    )}
                  >
                    {/* Turn Marker */}
                    <div className="absolute -left-12 flex flex-col items-center">
                       <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Turn</span>
                       <span className="text-lg font-black text-indigo-500/50">{node.turn}</span>
                    </div>

                    {/* Thumbnail */}
                    <div className="w-24 h-24 rounded-lg bg-slate-950 flex-shrink-0 overflow-hidden border border-white/5">
                       {node.image_url ? (
                         <img src={node.image_url} alt="" className="w-full h-full object-cover" />
                       ) : (
                         <div className="w-full h-full flex items-center justify-center">
                            <Eye className="w-6 h-6 text-slate-800" />
                         </div>
                       )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                       <h4 className={cn(
                         "font-bold text-sm mb-1",
                         node.type === 'start' ? "text-indigo-300" : "text-slate-200"
                       )}>{node.label}</h4>
                       <p className="text-xs text-slate-400 line-clamp-2 italic leading-relaxed">
                         {node.summary || "故事在此處展開..."}
                       </p>
                    </div>
                  </motion.div>

                  {/* Connector (Edge) */}
                  {idx < data.nodes.length - 1 && (
                    <motion.div 
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 40, opacity: 1 }}
                      transition={{ delay: idx * 0.1 + 0.05 }}
                      className="w-0.5 bg-gradient-to-b from-indigo-500/50 to-transparent my-1" 
                    />
                  )}
                </React.Fragment>
              ))}
              
              <div className="mt-8 px-6 py-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-[10px] font-bold text-indigo-400 uppercase tracking-widest animate-pulse">
                 未來的分支正在編織中...
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
