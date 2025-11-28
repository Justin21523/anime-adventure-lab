import { useState } from 'react'
import { useT2IHistory } from '../hooks/useT2IHistory'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import type { T2IHistoryItem } from '../types/t2i.types'

interface ImageGalleryProps {
  sessionId?: string // Optional: filter by story session
  onSelectImage?: (imageUrl: string) => void
}

export function ImageGallery({ sessionId, onSelectImage }: ImageGalleryProps) {
  const { data, isLoading } = useT2IHistory(sessionId)
  const [selectedItem, setSelectedItem] = useState<T2IHistoryItem | null>(null)

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-slate-400">加載圖像歷史中...</p>
        </CardContent>
      </Card>
    )
  }

  const history = data?.history || []

  const handleImageClick = (item: T2IHistoryItem) => {
    setSelectedItem(item)
  }

  const handleUseImage = (imageUrl: string) => {
    if (onSelectImage) {
      onSelectImage(imageUrl)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>
            生成歷史 ({history.length})
            {sessionId && <span className="text-sm text-slate-400 ml-2">當前故事場景</span>}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <p className="text-center text-slate-400 py-8">尚無生成記錄</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {history.map((item) =>
                item.images.map((img, idx) => (
                  <div
                    key={`${item.id}-${idx}`}
                    className="group relative aspect-square rounded overflow-hidden cursor-pointer border-2 border-transparent hover:border-primary transition-all"
                    onClick={() => handleImageClick(item)}
                  >
                    <img
                      src={img.image_url}
                      alt={item.prompt}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <p className="text-xs text-white text-center px-2 line-clamp-3">
                        {item.prompt}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Selected Image Detail */}
      {selectedItem && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>圖像詳情</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setSelectedItem(null)}>
                關閉
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Images */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {selectedItem.images.map((img, idx) => (
                <div key={idx} className="space-y-2">
                  <img
                    src={img.image_url}
                    alt={selectedItem.prompt}
                    className="w-full rounded border border-slate-700"
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">
                      Seed: {img.seed}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleUseImage(img.image_url)}
                    >
                      使用此圖
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {/* Prompt */}
            <div>
              <h4 className="text-sm font-semibold mb-1">提示詞</h4>
              <p className="text-sm text-slate-300 p-2 bg-slate-800/50 rounded">
                {selectedItem.prompt}
              </p>
            </div>

            {/* Negative Prompt */}
            {selectedItem.negative_prompt && (
              <div>
                <h4 className="text-sm font-semibold mb-1">負面提示詞</h4>
                <p className="text-sm text-slate-400 p-2 bg-slate-800/50 rounded">
                  {selectedItem.negative_prompt}
                </p>
              </div>
            )}

            {/* Settings */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-slate-400">尺寸: </span>
                <span className="text-slate-200">
                  {selectedItem.settings.width}x{selectedItem.settings.height}
                </span>
              </div>
              <div>
                <span className="text-slate-400">步數: </span>
                <span className="text-slate-200">{selectedItem.settings.steps}</span>
              </div>
              <div>
                <span className="text-slate-400">CFG: </span>
                <span className="text-slate-200">{selectedItem.settings.cfg_scale}</span>
              </div>
              <div>
                <span className="text-slate-400">時間: </span>
                <span className="text-slate-200">
                  {new Date(selectedItem.timestamp).toLocaleString()}
                </span>
              </div>
            </div>

            {/* LoRAs */}
            {selectedItem.settings.loras && selectedItem.settings.loras.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold mb-2">使用的 LoRA</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedItem.settings.loras.map((lora, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-primary/20 text-primary rounded text-xs"
                    >
                      {lora.name} ({lora.weight})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
