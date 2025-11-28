/**
 * SceneVisualizer Component
 *
 * Displays auto-generated scene images from T2I integration.
 * Shows generation metadata and provides image controls.
 */

import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface SceneImage {
  image_url: string;
  prompt: string;
  negative_prompt: string;
  generation_time: number;
  seed?: number;
  width: number;
  height: number;
}

interface SceneVisualizerProps {
  sceneImage: SceneImage | null;
  showMetadata?: boolean;
  className?: string;
}

export function SceneVisualizer({
  sceneImage,
  showMetadata = true,
  className = '',
}: SceneVisualizerProps) {
  if (!sceneImage) {
    return null;
  }

  return (
    <Card className={`relative overflow-hidden ${className}`}>
      {/* Scene Image */}
      <div className="relative w-full aspect-square">
        <img
          src={sceneImage.image_url}
          alt="Generated scene"
          className="w-full h-full object-cover rounded-t-lg"
          loading="lazy"
        />

        {/* Generation Time Badge */}
        <div className="absolute bottom-2 right-2">
          <Badge variant="secondary" className="backdrop-blur-sm bg-black/60 text-white">
            生成時間: {sceneImage.generation_time.toFixed(1)}s
          </Badge>
        </div>

        {/* Seed Badge (if available) */}
        {sceneImage.seed && (
          <div className="absolute top-2 right-2">
            <Badge variant="outline" className="backdrop-blur-sm bg-black/60 text-white border-white/20">
              Seed: {sceneImage.seed}
            </Badge>
          </div>
        )}
      </div>

      {/* Metadata Section (Optional) */}
      {showMetadata && (
        <div className="p-4 space-y-2 border-t border-border">
          {/* Positive Prompt */}
          <div className="space-y-1">
            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              場景提示詞
            </div>
            <div className="text-sm text-foreground/80 line-clamp-2">
              {sceneImage.prompt}
            </div>
          </div>

          {/* Image Dimensions */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>
              尺寸: {sceneImage.width}×{sceneImage.height}
            </span>
            <span>•</span>
            <span>
              生成耗時: {sceneImage.generation_time.toFixed(2)}秒
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}

/**
 * SceneVisualizerSkeleton
 *
 * Loading skeleton for scene image generation
 */
export function SceneVisualizerSkeleton({ className = '' }: { className?: string }) {
  return (
    <Card className={`relative overflow-hidden ${className}`}>
      <div className="relative w-full aspect-square bg-muted animate-pulse">
        {/* Loading indicator */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="space-y-2 text-center">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm text-muted-foreground">生成場景圖像中...</p>
          </div>
        </div>
      </div>

      {/* Metadata skeleton */}
      <div className="p-4 space-y-2 border-t border-border">
        <div className="h-3 bg-muted rounded w-20 animate-pulse" />
        <div className="h-4 bg-muted rounded w-full animate-pulse" />
        <div className="h-3 bg-muted rounded w-32 animate-pulse" />
      </div>
    </Card>
  );
}

/**
 * SceneVisualizerError
 *
 * Error state when scene image generation fails
 */
export function SceneVisualizerError({ className = '' }: { className?: string }) {
  return (
    <Card className={`relative overflow-hidden ${className}`}>
      <div className="relative w-full aspect-square bg-muted/50 border-2 border-dashed border-border">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="space-y-2 text-center p-4">
            <div className="text-4xl">🖼️</div>
            <p className="text-sm text-muted-foreground">
              場景圖像生成失敗
            </p>
            <p className="text-xs text-muted-foreground/60">
              請繼續遊戲，系統會在下次場景轉換時重新嘗試生成
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}
