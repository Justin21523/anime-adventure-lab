import { useState, useMemo, useEffect } from 'react'
import { DialogBox } from './DialogBox'
import { CharacterSprite } from './CharacterSprite'
import { ModelSettingsDialog } from '@/components/ModelSettingsDialog'
import { SaveLoadMenu } from './SaveLoadMenu'
import { NarrativeFlowchart } from './NarrativeFlowchart'
import { StoryWorkbenchDialog } from './StoryWorkbenchDialog'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Settings, 
  History, 
  User as UserIcon, 
  Map as MapIcon, 
  Heart,
  MessageSquare,
  Maximize2,
  Minimize2,
  Sparkles,
  MapPin,
  Heart as HealthIcon,
  Zap,
  Terminal,
  ChevronRight,
  GitBranch,
  Save as SaveIcon,
  LayoutGrid
} from 'lucide-react'
import type { StorySessionDetail } from '../types/story.types'
import type { TimelineTurn } from '../types/timeline.types'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TurnTimeline } from './TurnTimeline'
import { CharacterSheet } from './CharacterSheet'
import { RelationshipPanel } from './RelationshipPanel'
import { QuestExplorationPanel } from './QuestExplorationPanel'
import { PlayerInput } from './PlayerInput'
import { DataIngestionStudio } from '@/features/rag/components/DataIngestionStudio'
import type { WorldCharacterTemplate } from '@/features/worlds/types/world.types'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Progress } from '@/components/ui/progress'

interface VisualNovelScreenProps {
  session: StorySessionDetail
  sessionId: string
  isExecuting: boolean
  timelineTurns: TimelineTurn[]
  onSubmit: (input: string, options?: any) => Promise<void>
  onRefetch: () => Promise<any>
  world: any
  persona: any
  storyContext: any
  contextLoading: boolean
  contextError: any
  onToggleClassic?: () => void
}

export function VisualNovelScreen({
  session,
  sessionId,
  isExecuting,
  timelineTurns,
  onSubmit,
  onRefetch,
  world,
  persona,
  storyContext,
  contextLoading,
  contextError,
  onToggleClassic
}: VisualNovelScreenProps) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [menuTab, setMenuTab] = useState('history')
  const [showInput, setShowInput] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isDialogTyping, setIsDialogTyping] = useState(true)
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false)
  const [saveMenuOpen, setSaveMenuOpen] = useState(false)
  const [flowchartOpen, setFlowchartOpen] = useState(false)
  const [workbenchOpen, setWorkbenchOpen] = useState(false)
  const [workbenchTab, setWorkbenchTab] = useState('world')

  // Reset typing state when session changes or new turn starts
  useEffect(() => {
    setIsDialogTyping(true)
  }, [session.turn_count, timelineTurns.length])

  // Parse current text for speaker name and emotion
  const latestTurn = timelineTurns.length > 0 ? timelineTurns[timelineTurns.length - 1] : null
  const currentRawText = latestTurn ? latestTurn.result : (session.narrative || '故事開始...')
  
  const { currentName, currentText, currentEmotion } = useMemo(() => {
    // Attempt to parse "Name (Emotion): Dialogue" or "Name: Dialogue" format
    const match = currentRawText.match(/^([^:\n(]+)(?:\(([^)]+)\))?:\s*([\s\S]+)$/)
    if (match) {
      return { 
        currentName: match[1].trim(), 
        currentEmotion: (match[2]?.trim() || 'neutral') as any,
        currentText: match[3].trim() 
      }
    }
    return { currentName: persona?.name || '敘事者', currentEmotion: 'neutral', currentText: currentRawText }
  }, [currentRawText, persona?.name])

  // Get background image
  const bgImage = latestTurn?.scene_image?.image_url || session.scene_image?.image_url

  // Character Sprite System
  const presentCharacters = useMemo(() => {
    if (!world?.characters) return []
    
    // Get characters present in context
    const presentIds = new Set((storyContext?.present_characters || []).map((c: any) => c.character_id))
    
    // If speaking character is not in presentIds but is in world.characters, add it
    const speakingChar = world.characters.find((c: WorldCharacterTemplate) => 
      c.name === currentName || c.character_id === currentName
    )
    
    const displayList = world.characters.filter((c: WorldCharacterTemplate) => 
      presentIds.has(c.character_id) || c.character_id === speakingChar?.character_id
    )

    // Assign positions (max 3 for now: left, center, right)
    return displayList.slice(0, 3).map((c: WorldCharacterTemplate, idx: number) => {
      let position: 'left' | 'center' | 'right' = 'center'
      if (displayList.length === 2) {
        position = idx === 0 ? 'left' : 'right'
      } else if (displayList.length === 3) {
        position = idx === 0 ? 'left' : idx === 1 ? 'center' : 'right'
      }
      
      const isSpeaking = c.name === currentName || c.character_id === currentName
      return {
        ...c,
        position,
        isSpeaking,
        emotion: isSpeaking ? currentEmotion : 'neutral'
      }
    })
  }, [world?.characters, storyContext?.present_characters, currentName, currentEmotion])

  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen()
      setIsFullscreen(true)
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
        setIsFullscreen(false)
      }
    }
  }

  const openMenu = (tab: string) => {
    setMenuTab(tab)
    setMenuOpen(true)
  }

  return (
    <div className="relative w-full h-screen bg-slate-950 overflow-hidden font-sans select-none">
      {/* Background Layer with Crossfade */}
      <AnimatePresence mode="wait">
        <motion.div 
          key={bgImage || 'default-bg'}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.5 }}
          className="absolute inset-0 bg-cover bg-center"
          style={{
            backgroundImage: bgImage ? `url(${bgImage})` : 'none',
          }}
        >
          {!bgImage && (
            <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-indigo-950/20 to-slate-950" />
          )}
          {/* Dark Overlay for better text readability */}
          <div className="absolute inset-0 bg-black/40 backdrop-blur-[1px]" />
        </motion.div>
      </AnimatePresence>

      {/* Dynamic Ambient Particles (VFX) */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-30">
        <div className="absolute top-[-10%] left-[-10%] w-[120%] h-[120%] bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.05)_0%,transparent_70%)] animate-pulse" />
      </div>

      {/* Character Sprites Layer */}
      <div className="absolute inset-0 flex items-end justify-center pb-40 pointer-events-none px-12 overflow-hidden z-0">
        <div className="flex items-end justify-center gap-12 w-full max-w-7xl">
          {presentCharacters.map((char: any) => (
            <CharacterSprite
              key={char.character_id}
              name={char.name}
              imageUrl={char.image_url}
              position={char.position as any}
              isSpeaking={char.isSpeaking}
              isActive={true}
              emotion={char.emotion}
            />
          ))}
        </div>
      </div>

      {/* UI Layer - Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-6 flex items-start justify-between z-20 pointer-events-none">
        <motion.div 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex flex-col gap-3 pointer-events-auto"
        >
          <div className="flex gap-3">
            <Badge className="bg-black/60 backdrop-blur-md border-indigo-500/30 text-indigo-100 py-1.5 px-4 rounded-full shadow-lg">
              <Sparkles className="w-3.5 h-3.5 mr-2 text-indigo-400" />
              回合 {session.turn_count}
            </Badge>
            <Badge variant="outline" className="bg-black/60 backdrop-blur-md border-slate-700/50 text-slate-300 py-1.5 px-4 rounded-full shadow-lg flex items-center gap-2">
              <MapPin className="w-3.5 h-3.5 text-emerald-400" />
              {storyContext?.current_location || '探索中...'}
            </Badge>
          </div>
          
          {/* Mini HUD - Health & Energy */}
          <div className="flex gap-4 items-center bg-black/40 backdrop-blur-md p-3 rounded-2xl border border-white/5 shadow-2xl w-fit">
             <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 px-1">
                   <span className="flex items-center gap-1"><HealthIcon className="w-3 h-3 text-red-500 fill-current" /> HP</span>
                   <span>{session.stats?.health || 100} / 100</span>
                </div>
                <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                   <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${session.stats?.health || 100}%` }}
                    className="h-full bg-gradient-to-r from-red-600 to-rose-500" 
                   />
                </div>
             </div>
             <div className="w-px h-6 bg-white/10 mx-1" />
             <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 px-1">
                   <span className="flex items-center gap-1"><Zap className="w-3 h-3 text-amber-500 fill-current" /> SP</span>
                   <span>{session.stats?.energy || 100} / 100</span>
                </div>
                <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                   <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${session.stats?.energy || 100}%` }}
                    className="h-full bg-gradient-to-r from-amber-500 to-orange-400" 
                   />
                </div>
             </div>
          </div>
        </motion.div>

        <motion.div 
          initial={{ y: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="flex gap-3 pointer-events-auto"
        >
          <Button 
            variant="ghost" 
            size="icon" 
            className="w-11 h-11 rounded-full bg-black/50 backdrop-blur-md text-white hover:bg-indigo-600/80 border border-white/10 shadow-xl transition-all"
            onClick={() => setModelSettingsOpen(true)}
            title="模型設定"
          >
            <Terminal className="w-5.5 h-5.5 text-indigo-400" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="w-11 h-11 rounded-full bg-black/50 backdrop-blur-md text-white hover:bg-indigo-600/80 border border-white/10 shadow-xl transition-all"
            onClick={onToggleClassic}
            title="切換到儀表板"
          >
            <Settings className="w-5.5 h-5.5" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="w-11 h-11 rounded-full bg-black/50 backdrop-blur-md text-white hover:bg-indigo-600/80 border border-white/10 shadow-xl transition-all"
            onClick={handleToggleFullscreen}
          >
            {isFullscreen ? <Minimize2 className="w-5.5 h-5.5" /> : <Maximize2 className="w-5.5 h-5.5" />}
          </Button>
        </motion.div>
      </div>

      {/* Quick Access Sidebar */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-4 z-20">
        <SideButton icon={<SaveIcon className="w-5 h-5 text-emerald-400" />} label="存檔" onClick={() => setSaveMenuOpen(true)} position="left" />
        <SideButton icon={<GitBranch className="w-5 h-5 text-indigo-400" />} label="路徑" onClick={() => setFlowchartOpen(true)} position="left" />
        <SideButton icon={<LayoutGrid className="w-5 h-5 text-amber-400" />} label="工具" onClick={() => setWorkbenchOpen(true)} position="left" />
        <div className="h-px w-8 bg-white/10 mx-auto my-1" />
        <SideButton icon={<History className="w-5 h-5" />} label="回顧" onClick={() => openMenu('history')} position="left" />
        <SideButton icon={<UserIcon className="w-5 h-5" />} label="狀態" onClick={() => openMenu('status')} position="left" />
        <SideButton icon={<Heart className="w-5 h-5" />} label="關係" onClick={() => openMenu('relationships')} position="left" />
        <SideButton icon={<MapIcon className="w-5 h-5" />} label="任務" onClick={() => openMenu('quests')} position="left" />
      </div>

      {/* Choices Layer - Now Non-Blocking and appears after typing */}
      <AnimatePresence>
        {!isExecuting && !isDialogTyping && session.choices && session.choices.length > 0 && !showInput && (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="absolute right-12 top-1/4 bottom-1/4 w-80 flex flex-col justify-center z-30 pointer-events-none"
          >
            <div className="flex flex-col gap-4 p-6 bg-slate-950/60 backdrop-blur-2xl rounded-3xl border border-indigo-500/20 shadow-[0_0_40px_rgba(0,0,0,0.5)] pointer-events-auto">
              <h3 className="text-indigo-300 text-[10px] font-bold uppercase tracking-[0.4em] mb-2 px-2">
                抉擇之刻
              </h3>
              {session.choices.map((choice, idx) => (
                <motion.div
                  key={choice.choice_id}
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: idx * 0.1 }}
                >
                  <Button
                    variant="outline"
                    className="w-full py-6 px-6 text-sm font-medium bg-slate-900/80 backdrop-blur-xl border-indigo-500/30 hover:bg-indigo-600 hover:border-indigo-400 text-white shadow-lg hover:scale-[1.05] active:scale-[0.98] transition-all duration-300 group rounded-xl text-left justify-start h-auto"
                    onClick={() => onSubmit(choice.text, { choiceId: choice.choice_id })}
                  >
                    <span className="flex-1 line-clamp-2">{choice.text}</span>
                    <ChevronRight className="w-4 h-4 ml-2 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </Button>
                </motion.div>
              ))}
              
              <Button
                variant="ghost"
                className="w-full text-slate-500 hover:text-indigo-300 mt-2 tracking-widest text-[10px] uppercase h-8"
                onClick={() => setShowInput(true)}
              >
                <MessageSquare className="w-3 h-3 mr-2" />
                自定義行動...
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom Area - Dialog & Input */}
      <div className="absolute bottom-0 left-0 right-0 z-10 flex flex-col items-center">
        <AnimatePresence>
          {isExecuting && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="mb-12 flex flex-col items-center gap-4"
            >
              <div className="relative">
                <div className="w-16 h-16 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-indigo-400 animate-pulse" />
                </div>
              </div>
              <span className="text-indigo-100 text-sm font-bold tracking-[0.2em] bg-black/60 px-6 py-2 rounded-full backdrop-blur-md border border-indigo-500/30 shadow-[0_0_20px_rgba(99,102,241,0.3)]">
                正在編織命運的下一章...
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showInput && !isExecuting && (
            <motion.div 
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 100, opacity: 0 }}
              className="w-full max-w-4xl px-4 pb-8"
            >
              <Card className="bg-slate-950/90 backdrop-blur-2xl border-indigo-500/40 p-6 shadow-[0_-20px_50px_rgba(0,0,0,0.5)] rounded-2xl ring-1 ring-white/10">
                <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full animate-ping" />
                    <span className="text-xs font-bold text-indigo-300 uppercase tracking-[0.2em]">自定義行動輸入</span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="h-7 text-slate-500 hover:text-white hover:bg-white/5 rounded-full px-4"
                    onClick={() => setShowInput(false)}
                  >
                    返回選項
                  </Button>
                </div>
                <PlayerInput
                  sessionId={sessionId}
                  onSubmit={async (input, opts) => {
                    await onSubmit(input, opts)
                    setShowInput(false)
                  }}
                  isExecuting={isExecuting}
                  choices={session.choices}
                  worldId={session.world_id}
                  ragAuto={session.rag_auto}
                  ragMode={session.rag_mode}
                  ragAvailable={session.rag_available}
                  enrichWithRag={session.enrich_with_rag}
                  ragNextTurn={session.rag_next_turn}
                  rerankMode={session.rerank_mode}
                  rerankNextTurn={session.rerank_next_turn}
                />
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {!showInput && (
          <DialogBox 
            name={currentName}
            text={currentText}
            onShowHistory={() => openMenu('history')}
            onComplete={() => setIsDialogTyping(false)}
          />
        )}
      </div>

      {/* Modals */}
      <ModelSettingsDialog open={modelSettingsOpen} onOpenChange={setModelSettingsOpen} />
      <SaveLoadMenu 
        open={saveMenuOpen} 
        onOpenChange={setSaveMenuOpen} 
        sessionId={sessionId} 
        onLoadSuccess={onRefetch} 
      />
      <NarrativeFlowchart 
        open={flowchartOpen} 
        onOpenChange={setFlowchartOpen} 
        sessionId={sessionId} 
      />
      <StoryWorkbenchDialog
        open={workbenchOpen}
        onOpenChange={setWorkbenchOpen}
        sessionId={sessionId}
        worldId={session.world_id}
        tab={workbenchTab}
        onTabChange={setWorkbenchTab}
      />

      {/* Menu Overlay */}
      <Dialog open={menuOpen} onOpenChange={setMenuOpen}>
        <DialogContent className="max-w-5xl h-[85vh] p-0 overflow-hidden bg-slate-950/95 backdrop-blur-2xl border-indigo-900/50 shadow-[0_0_100px_rgba(0,0,0,0.8)]">
          <Tabs defaultValue="history" value={menuTab} onValueChange={setMenuTab} className="h-full flex flex-col">
            <div className="px-8 py-6 border-b border-indigo-900/30 bg-indigo-950/20 flex items-center justify-between">
              <DialogHeader>
                <DialogTitle className="text-2xl font-bold text-white flex items-center gap-3">
                  <div className="p-2 bg-indigo-600/20 rounded-lg">
                    <Settings className="w-6 h-6 text-indigo-400" />
                  </div>
                  冒險選單
                </DialogTitle>
              </DialogHeader>
              <TabsList className="bg-slate-900/80 border border-indigo-900/30 p-1 h-auto">
                <TabsTrigger value="history" className="gap-2 py-2 px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all"><History className="w-4 h-4" /> 歷史紀錄</TabsTrigger>
                <TabsTrigger value="status" className="gap-2 py-2 px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all"><UserIcon className="w-4 h-4" /> 角色狀態</TabsTrigger>
                <TabsTrigger value="relationships" className="gap-2 py-2 px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all"><Heart className="w-4 h-4" /> 關係圖譜</TabsTrigger>
                <TabsTrigger value="quests" className="gap-2 py-2 px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all"><MapIcon className="w-4 h-4" /> 任務目標</TabsTrigger>
                <TabsTrigger value="ingest" className="gap-2 py-2 px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all"><Sparkles className="w-4 h-4" /> 知識錄入</TabsTrigger>
              </TabsList>
            </div>

            <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
              <TabsContent value="history" className="mt-0 h-full">
                <div className="bg-slate-900/40 p-4 rounded-xl border border-indigo-900/20 h-full overflow-y-auto">
                  <TurnTimeline sessionId={sessionId} turns={timelineTurns} isExecuting={isExecuting} />
                </div>
              </TabsContent>
              <TabsContent value="status" className="mt-0 space-y-6">
                <CharacterSheet
                  playerName={session.player_name}
                  stats={session.stats}
                  inventory={session.inventory}
                  flags={session.flags}
                />
              </TabsContent>
              <TabsContent value="relationships" className="mt-0">
                <RelationshipPanel
                  context={storyContext}
                  isLoading={contextLoading}
                  error={contextError}
                  worldpack={world || null}
                  inventory={session.inventory}
                  turnHistory={session.turn_history || null}
                  onQuickAction={() => setMenuOpen(false)}
                />
              </TabsContent>
              <TabsContent value="quests" className="mt-0">
                <QuestExplorationPanel flags={session.flags} worldpack={world || null} />
              </TabsContent>
              <TabsContent value="ingest" className="mt-0">
                <DataIngestionStudio worldId={session.world_id} onIngestSuccess={() => setMenuOpen(false)} />
              </TabsContent>
            </div>
          </Tabs>
        </DialogContent>
      </Dialog>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(79, 70, 229, 0.4);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(79, 70, 229, 0.6);
        }
      `}</style>
    </div>
  )
}

function SideButton({ icon, label, onClick, position = 'right' }: { icon: React.ReactNode, label: string, onClick: () => void, position?: 'left' | 'right' }) {
  return (
    <div className={cn("group flex items-center gap-3", position === 'left' ? "flex-row" : "flex-row-reverse")}>
      <motion.div
        whileHover={{ scale: 1.1, x: position === 'left' ? 5 : -5 }}
        whileTap={{ scale: 0.95 }}
      >
        <Button 
          variant="ghost" 
          size="icon" 
          className="w-14 h-14 rounded-2xl bg-black/60 backdrop-blur-xl text-slate-300 hover:text-white hover:bg-indigo-600/80 border border-indigo-500/20 shadow-[0_8px_30px_rgba(0,0,0,0.4)] transition-all relative overflow-hidden group/btn"
          onClick={onClick}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 opacity-0 group-hover/btn:opacity-100 transition-opacity" />
          {icon}
        </Button>
      </motion.div>
      <span className={cn(
        "opacity-0 group-hover:opacity-100 transition-all bg-indigo-600/90 backdrop-blur-md text-white text-[10px] px-3 py-1.5 rounded-full border border-indigo-400/50 uppercase tracking-[0.2em] font-bold shadow-lg transform",
        position === 'left' ? "-translate-x-2 group-hover:translate-x-0" : "translate-x-2 group-hover:translate-x-0"
      )}>
        {label}
      </span>
    </div>
  )
}
