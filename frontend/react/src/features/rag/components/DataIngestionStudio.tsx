import React, { useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { apiPost, apiUploadFile } from '@/api/client'
import { useToast } from '@/hooks/useToast'
import { 
  UserPlus, 
  Globe, 
  FileText, 
  Image as ImageIcon, 
  Loader2, 
  Save, 
  Sparkles,
  Upload,
  CheckCircle2,
  FileSearch,
  Plus,
  X,
  Tag,
  Lightbulb
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface DataIngestionStudioProps {
  worldId?: string
  onIngestSuccess?: () => void
}

export function DataIngestionStudio({ worldId = 'default', onIngestSuccess }: DataIngestionStudioProps) {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState('character')
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [vlmAnalyzing, setVlmAnalyzing] = useState(false)

  // Character Form State
  const [charData, setCharData] = useState({
    name: '',
    title: '',
    appearance: '',
    personality: '',
    background: '',
    inventory: ''
  })

  // Advanced Analysis State
  const [analysisFile, setAnalysisFile] = useState<File | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [customMetadata, setCustomMetadata] = useState<Array<{ key: string, value: string }>>([
    { key: 'category', value: '' },
    { key: 'tags', value: '' }
  ])

  const commonMetadataKeys = [
    { key: 'category', label: '分類' },
    { key: 'series', label: '作品/系列' },
    { key: 'author', label: '作者' },
    { key: 'era', label: '年代/背景' },
    { key: 'related_character', label: '關聯角色' },
    { key: 'location', label: '地點' },
    { key: 'ability', label: '能力/法術' },
    { key: 'importance', label: '重要度 (1-5)' },
  ]

  const addPresetMetadata = (key: string) => {
    if (!customMetadata.find(m => m.key === key)) {
      setCustomMetadata([...customMetadata, { key, value: '' }])
    }
  }
  
  // Markdown Editor State
  const [markdownContent, setMarkdownContent] = useState('')
  const [markdownTitle, setMarkdownTitle] = useState('')

  // VLM State
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleCharIngest = async () => {
    if (!charData.name) return
    setLoading(true)
    const content = `
## 角色概覽
- **姓名**: ${charData.name}
- **稱號**: ${charData.title}

## 外貌特徵
${charData.appearance}

## 性格描述
${charData.personality}

## 背景故事
${charData.background}

## 關鍵物品
${charData.inventory}
    `.trim()

    try {
      await apiPost('/rag/ingest_structured', {
        title: `角色設定_${charData.name}`,
        content,
        world_id: worldId,
        tags: ['character_setting', charData.name],
        metadata: { type: 'character', character_name: charData.name }
      })
      toast({ title: '角色設定已存入 RAG', description: `${charData.name} 的資料已成功索引。` })
      setCharData({ name: '', title: '', appearance: '', personality: '', background: '', inventory: '' })
      onIngestSuccess?.()
    } catch (e) {
      toast({ title: '錄入失敗', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  const handleFileAnalyze = async () => {
    if (!analysisFile) return
    setAnalyzing(true)
    try {
      const result = await apiUploadFile<any>('/rag/analyze', analysisFile, {
        world_id: worldId
      })
      setAnalysisResult(result)
      setMarkdownContent(result.full_content)
      setMarkdownTitle(result.suggested_metadata?.title || result.original_filename)
      
      const suggested = result.suggested_metadata || {}
      const metaArray: Array<{ key: string, value: string }> = []
      if (suggested.category) metaArray.push({ key: 'category', value: String(suggested.category) })
      if (suggested.entities) metaArray.push({ key: 'entities', value: Array.isArray(suggested.entities) ? suggested.entities.join(', ') : String(suggested.entities) })
      if (suggested.summary) metaArray.push({ key: 'summary', value: String(suggested.summary) })
      if (suggested.tags) metaArray.push({ key: 'tags', value: Array.isArray(suggested.tags) ? suggested.tags.join(', ') : String(suggested.tags) })
      
      setCustomMetadata(metaArray)
      toast({ title: '文件分析完成', description: '已提取關鍵中繼資料，請確認並調整。' })
    } catch (e) {
      toast({ title: '分析失敗', variant: 'destructive' })
    } finally {
      setAnalyzing(false)
    }
  }

  const handleVlmAnalyze = async () => {
    if (!selectedFile) return
    setVlmAnalyzing(true)
    try {
      const result = await apiUploadFile<any>('/vlm/analyze_to_rag', selectedFile, {
        world_id: worldId
      })
      setMarkdownContent(result.markdown_content)
      setMarkdownTitle(result.suggested_title)
      toast({ title: '視覺分析完成', description: '已根據圖片產出描述，請在 [自由編輯] 分頁確認。' })
      setActiveTab('markdown')
    } catch (e) {
      toast({ title: '分析失敗', variant: 'destructive' })
    } finally {
      setVlmAnalyzing(false)
    }
  }

  const handleAdvancedIngest = async () => {
    if (!markdownContent || !markdownTitle) return
    setLoading(true)
    try {
      const finalMetadata: Record<string, any> = {}
      customMetadata.forEach(m => {
        if (m.key.trim()) finalMetadata[m.key.trim()] = m.value
      })

      await apiPost('/rag/ingest_structured', {
        title: markdownTitle,
        content: markdownContent,
        world_id: worldId,
        tags: (finalMetadata.tags || '').split(',').map((t: string) => t.trim()).filter(Boolean),
        metadata: { ...finalMetadata, type: 'advanced_analysis' }
      })
      toast({ title: '高質量資料已存入 RAG' })
      resetAdvancedForm()
      onIngestSuccess?.()
    } catch (e) {
      toast({ title: '錄入失敗', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  const handleMarkdownIngest = async () => {
    if (!markdownContent || !markdownTitle) return
    setLoading(true)
    try {
      await apiPost('/rag/ingest_structured', {
        title: markdownTitle,
        content: markdownContent,
        world_id: worldId,
        tags: ['manual_entry'],
        metadata: { type: 'manual' }
      })
      toast({ title: '資料已存入 RAG' })
      setMarkdownContent('')
      setMarkdownTitle('')
      onIngestSuccess?.()
    } catch (e) {
      toast({ title: '錄入失敗', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  const resetAdvancedForm = () => {
    setAnalysisFile(null)
    setAnalysisResult(null)
    setCustomMetadata([])
    setMarkdownContent('')
    setMarkdownTitle('')
  }

  const addMetadataField = () => setCustomMetadata([...customMetadata, { key: '', value: '' }])
  const removeMetadataField = (index: number) => setCustomMetadata(customMetadata.filter((_, i) => i !== index))
  const updateMetadataField = (index: number, key: string, value: string) => {
    const newMeta = [...customMetadata]
    newMeta[index] = { key, value }
    setCustomMetadata(newMeta)
  }

  return (
    <Card className="w-full bg-slate-950 border-indigo-900/40 shadow-2xl overflow-hidden">
      <CardHeader className="bg-indigo-950/20 border-b border-white/5">
        <div className="flex justify-between items-center">
           <div>
              <CardTitle className="text-xl font-bold flex items-center gap-2 text-white">
                <Sparkles className="text-indigo-400 w-5 h-5" />
                RAG 資料煉金術
              </CardTitle>
              <CardDescription className="text-slate-400">建立高品質的故事背景知識庫</CardDescription>
           </div>
           <Badge variant="outline" className="border-indigo-500/30 text-indigo-400">World: {worldId}</Badge>
        </div>
      </CardHeader>
      
      <Tabs defaultValue="character" value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="px-6 pt-4">
          <TabsList className="bg-slate-900 border border-white/5 w-full justify-start gap-2 h-auto p-1 overflow-x-auto">
            <TabsTrigger value="character" className="gap-2 py-2 flex-shrink-0"><UserPlus className="w-4 h-4" /> 角色設定</TabsTrigger>
            <TabsTrigger value="analyze" className="gap-2 py-2 flex-shrink-0"><FileSearch className="w-4 h-4" /> 文件分析錄入</TabsTrigger>
            <TabsTrigger value="vlm" className="gap-2 py-2 flex-shrink-0"><ImageIcon className="w-4 h-4" /> 視覺轉換</TabsTrigger>
            <TabsTrigger value="markdown" className="gap-2 py-2 flex-shrink-0"><FileText className="w-4 h-4" /> 自由編輯</TabsTrigger>
          </TabsList>
        </div>

        <CardContent className="p-6">
          <AnimatePresence mode="wait">
            {activeTab === 'character' && (
              <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} exit={{opacity: 0, y: -10}} key="char" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                      <Label className="text-slate-300">角色姓名</Label>
                      <Input value={charData.name} onChange={e => setCharData({...charData, name: e.target.value})} placeholder="例如: 愛莉絲" className="bg-slate-900 border-slate-800" />
                  </div>
                  <div className="space-y-2">
                      <Label className="text-slate-300">角色稱號/職業</Label>
                      <Input value={charData.title} onChange={e => setCharData({...charData, title: e.target.value})} placeholder="例如: 晨曦騎士" className="bg-slate-900 border-slate-800" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">外貌特徵</Label>
                  <Textarea value={charData.appearance} onChange={e => setCharData({...charData, appearance: e.target.value})} placeholder="金髮碧眼，身穿銀色鎧甲..." className="bg-slate-900 border-slate-800 h-20" />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">性格與背景</Label>
                  <Textarea value={charData.personality} onChange={e => setCharData({...charData, personality: e.target.value})} placeholder="性格堅毅但內心溫柔..." className="bg-slate-900 border-slate-800 h-24" />
                </div>
                <Button className="w-full bg-indigo-600 hover:bg-indigo-500" onClick={handleCharIngest} disabled={loading || !charData.name}>
                  {loading ? <Loader2 className="animate-spin mr-2" /> : <Save className="mr-2 w-4 h-4" />}
                  將角色存入 RAG 記憶庫
                </Button>
              </motion.div>
            )}

            {activeTab === 'analyze' && (
              <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} exit={{opacity: 0, y: -10}} key="analyze" className="space-y-6">
                 {!analysisResult ? (
                   <div className="border-2 border-dashed border-slate-800 rounded-2xl p-12 text-center hover:border-indigo-500/50 transition-all group">
                      <div className="flex flex-col items-center gap-4">
                        <div className="w-16 h-16 bg-indigo-500/10 rounded-full flex items-center justify-center">
                           <Upload className="w-8 h-8 text-indigo-400" />
                        </div>
                        <div>
                           <p className="text-slate-200 font-bold">上傳文件檔案 (PDF, TXT, MD)</p>
                           <p className="text-slate-500 text-sm mt-1">LLM 將自動提取結構化中繼資料並建議標籤</p>
                        </div>
                        <Input type="file" className="hidden" id="analyze-upload" onChange={e => setAnalysisFile(e.target.files?.[0] || null)} />
                        {analysisFile ? (
                          <div className="flex flex-col items-center gap-4">
                             <Badge className="bg-indigo-600 px-4 py-1">{analysisFile.name}</Badge>
                             <Button onClick={handleFileAnalyze} disabled={analyzing} className="bg-indigo-600">
                                {analyzing ? <Loader2 className="animate-spin mr-2" /> : <FileSearch className="mr-2 w-4 h-4" />}
                                開始分析文件
                             </Button>
                             <Button variant="ghost" onClick={() => setAnalysisFile(null)} className="text-slate-500 text-xs">取消選擇</Button>
                          </div>
                        ) : (
                          <Button variant="outline" className="mt-2 border-indigo-500/30" asChild>
                             <label htmlFor="analyze-upload">選擇檔案</label>
                          </Button>
                        )}
                      </div>
                   </div>
                 ) : (
                   <div className="space-y-6">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                         <div className="space-y-2 flex flex-col h-full">
                            <Label className="text-slate-300 flex items-center gap-2"><FileText className="w-4 h-4" /> 文件內容預覽</Label>
                            <Textarea value={markdownContent} onChange={e => setMarkdownContent(e.target.value)} className="bg-slate-900 border-slate-800 flex-1 min-h-[300px] font-mono text-[11px]" />
                         </div>
                         {/* Metadata Editor */}
                         <div className="space-y-4">
                            <div className="flex items-center justify-between">
                               <Label className="text-slate-300 flex items-center gap-2"><Tag className="w-4 h-4" /> 結構化中繼資料 (Metadata)</Label>
                               <Button variant="ghost" size="sm" onClick={resetAdvancedForm} className="text-xs text-slate-500 h-6">重置全部</Button>
                            </div>
                            
                            {/* Quick Presets */}
                            <div className="flex flex-wrap gap-2">
                               <span className="text-[10px] text-slate-500 flex items-center gap-1"><Lightbulb className="w-3 h-3" /> 快速新增:</span>
                               {commonMetadataKeys.map(k => (
                                 <button 
                                  key={k.key}
                                  onClick={() => addPresetMetadata(k.key)}
                                  className="text-[10px] px-2 py-0.5 bg-indigo-600/20 text-indigo-300 rounded-full hover:bg-indigo-600/40 border border-indigo-500/20 transition-all"
                                 >
                                    + {k.label}
                                 </button>
                               ))}
                            </div>

                            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                               <div className="space-y-1.5 p-3 bg-slate-900/50 rounded-xl border border-white/5">
                                  <Label className="text-[10px] uppercase tracking-widest text-slate-500">文件標題 (必要)</Label>
                                  <Input value={markdownTitle} onChange={e => setMarkdownTitle(e.target.value)} className="bg-slate-950 h-9 border-indigo-900/20" placeholder="輸入此知識點的標題" />
                               </div>
                               
                               {customMetadata.map((m, i) => (
                                 <div key={i} className="flex gap-2 items-start bg-slate-900/30 p-3 rounded-xl border border-white/5 group transition-all hover:border-indigo-500/30">
                                    <div className="flex-1 space-y-2">
                                       <div className="flex items-center justify-between">
                                          <Input 
                                            placeholder="欄位名 (Key)" 
                                            value={m.key} 
                                            onChange={e => updateMetadataField(i, e.target.value, m.value)}
                                            className="bg-transparent border-none p-0 h-auto text-[10px] font-bold uppercase tracking-widest text-indigo-400 focus-visible:ring-0 w-full"
                                          />
                                          <Button variant="ghost" size="icon" onClick={() => removeMetadataField(i)} className="text-slate-600 hover:text-rose-400 h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity">
                                             <X className="w-3 h-3" />
                                          </Button>
                                       </div>
                                       <Textarea 
                                        placeholder="內容 (Value)" 
                                        value={m.value} 
                                        onChange={e => updateMetadataField(i, m.key, e.target.value)}
                                        className="bg-slate-950 min-h-[60px] text-xs border-indigo-900/20 rounded-lg focus:ring-indigo-500/30"
                                       />
                                    </div>
                                 </div>
                               ))}
                               
                               <Button variant="outline" size="sm" onClick={addMetadataField} className="w-full border-dashed border-indigo-500/20 text-indigo-400/70 hover:text-indigo-300 rounded-xl py-4 h-auto">
                                  <Plus className="w-4 h-4 mr-2" /> 新增自定義欄位
                               </Button>
                            </div>
                         </div>
                      </div>
                      <div className="flex gap-3 pt-4">
                         <Button variant="ghost" onClick={resetAdvancedForm} className="flex-1">重置</Button>
                         <Button onClick={handleAdvancedIngest} disabled={loading} className="flex-[2] bg-indigo-600 hover:bg-indigo-500">
                            {loading ? <Loader2 className="animate-spin mr-2" /> : <CheckCircle2 className="mr-2 w-4 h-4" />}
                            確認錄入 RAG 知識庫
                         </Button>
                      </div>
                   </div>
                 )}
              </motion.div>
            )}

            {activeTab === 'vlm' && (
              <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} exit={{opacity: 0, y: -10}} key="vlm" className="space-y-6">
                <div className="border-2 border-dashed border-slate-800 rounded-2xl p-12 text-center hover:border-indigo-500/50 transition-all group">
                   {!selectedFile ? (
                     <div className="flex flex-col items-center gap-4">
                        <div className="w-16 h-16 bg-indigo-500/10 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                           <Upload className="w-8 h-8 text-indigo-400" />
                        </div>
                        <p className="text-slate-200 font-bold">上傳角色或場景圖</p>
                        <p className="text-slate-500 text-sm">VLM 將自動分析圖片並轉換為結構化設定</p>
                        <Input type="file" className="hidden" id="vlm-upload" accept="image/*" onChange={e => setSelectedFile(e.target.files?.[0] || null)} />
                        <Button variant="outline" className="mt-2 border-indigo-500/30" asChild><label htmlFor="vlm-upload">選擇檔案</label></Button>
                     </div>
                   ) : (
                     <div className="space-y-4">
                        <img src={URL.createObjectURL(selectedFile)} className="max-h-48 mx-auto rounded-lg shadow-lg" alt="Preview" />
                        <div className="flex items-center justify-center gap-3">
                           <Badge variant="secondary">{selectedFile.name}</Badge>
                           <Button variant="ghost" size="sm" onClick={() => setSelectedFile(null)} className="text-rose-400">移除</Button>
                        </div>
                        <Button className="w-full bg-emerald-600 hover:bg-emerald-500" onClick={handleVlmAnalyze} disabled={vlmAnalyzing}>
                          {vlmAnalyzing ? <Loader2 className="animate-spin mr-2" /> : <Sparkles className="mr-2 w-4 h-4" />}
                          開始視覺分析 (呼叫 VLM)
                        </Button>
                     </div>
                   )}
                </div>
              </motion.div>
            )}

            {activeTab === 'markdown' && (
              <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} exit={{opacity: 0, y: -10}} key="markdown" className="space-y-4">
                 <div className="space-y-2">
                    <Label className="text-slate-300">資料標題</Label>
                    <Input value={markdownTitle} onChange={e => setMarkdownTitle(e.target.value)} placeholder="例如: 艾澤拉斯歷史卷軸" className="bg-slate-900 border-slate-800" />
                 </div>
                 <div className="space-y-2">
                    <Label className="text-slate-300">Markdown 內容</Label>
                    <Textarea value={markdownContent} onChange={e => setMarkdownContent(e.target.value)} placeholder="# 標題..." className="bg-slate-900 border-slate-800 h-64 font-mono text-sm" />
                 </div>
                 <Button className="w-full bg-indigo-600 hover:bg-indigo-500" onClick={handleMarkdownIngest} disabled={loading || !markdownContent || !markdownTitle}>
                   {loading ? <Loader2 className="animate-spin mr-2" /> : <CheckCircle2 className="mr-2 w-4 h-4" />}
                   確認並匯入 RAG
                 </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Tabs>
    </Card>
  )
}
