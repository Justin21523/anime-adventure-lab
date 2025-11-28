import { useState } from 'react'
import { T2IGenerator } from './T2IGenerator'
import { ImageGallery } from './ImageGallery'
import { Button } from '@/components/ui/button'

/**
 * T2I Management - Main component for image generation
 * Integrated with story system for scene visualization
 */
export function T2IManagement() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  return (
    <div className="container mx-auto p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">場景生成系統</h1>
          <p className="text-slate-400 mt-1">
            為故事場景生成視覺畫面，提升沉浸體驗
          </p>
        </div>
        <Button variant="outline" onClick={() => (window.location.href = '/')}>
          返回首頁
        </Button>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <T2IGenerator onImageGenerated={(url) => setSelectedImage(url)} />
        </div>
        <div>
          <ImageGallery onSelectImage={(url) => setSelectedImage(url)} />
        </div>
      </div>

      {/* Selected Image Preview */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="relative max-w-4xl max-h-[90vh]">
            <img
              src={selectedImage}
              alt="Selected"
              className="w-full h-full object-contain rounded"
            />
            <Button
              variant="ghost"
              size="sm"
              className="absolute top-4 right-4 bg-black/50"
              onClick={() => setSelectedImage(null)}
            >
              關閉
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
