/**
 * MemoryIndicator Component
 *
 * Displays story memory status and statistics.
 * Shows short-term, mid-term, and long-term memory availability.
 */

import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface MemoryStats {
  short_term_count: number;
  summaries_count: number;
  total_turns_covered: number;
  turns_since_last_summary: number;
  rag_available: boolean;
}

interface MemoryIndicatorProps {
  memoryStats?: MemoryStats | null;
  className?: string;
}

export function MemoryIndicator({ memoryStats, className = '' }: MemoryIndicatorProps) {
  if (!memoryStats) {
    return null;
  }

  return (
    <Card className={`p-3 space-y-2 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">記憶系統</h3>
        <Badge variant={memoryStats.rag_available ? 'default' : 'secondary'}>
          {memoryStats.rag_available ? '運行中' : '離線'}
        </Badge>
      </div>

      {/* Memory Layers */}
      <div className="space-y-1.5">
        {/* Short-term Memory */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">短期記憶</span>
          <div className="flex items-center gap-1.5">
            <div className="flex gap-0.5">
              {Array.from({ length: 10 }).map((_, i) => (
                <div
                  key={i}
                  className={`w-1.5 h-3 rounded-sm ${
                    i < memoryStats.short_term_count
                      ? 'bg-blue-500'
                      : 'bg-muted'
                  }`}
                  title={`回合 ${i + 1}`}
                />
              ))}
            </div>
            <span className="text-foreground font-medium min-w-[2ch] text-right">
              {memoryStats.short_term_count}
            </span>
          </div>
        </div>

        {/* Mid-term Memory (Summaries) */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">中期記憶</span>
          <div className="flex items-center gap-1.5">
            <span className="text-foreground font-medium">
              {memoryStats.summaries_count} 個摘要
            </span>
          </div>
        </div>

        {/* Long-term Memory (RAG) */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">長期記憶</span>
          <div className="flex items-center gap-1.5">
            <span className={`font-medium ${memoryStats.rag_available ? 'text-green-500' : 'text-muted-foreground'}`}>
              {memoryStats.rag_available ? '向量搜尋啟用' : '未啟用'}
            </span>
          </div>
        </div>
      </div>

      {/* Compression Progress */}
      <div className="pt-2 border-t border-border">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>下次壓縮</span>
          <span>
            {5 - memoryStats.turns_since_last_summary} 回合後
          </span>
        </div>
        <div className="mt-1 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{
              width: `${(memoryStats.turns_since_last_summary / 5) * 100}%`
            }}
          />
        </div>
      </div>

      {/* Total Coverage */}
      <div className="pt-1 text-xs text-center text-muted-foreground">
        已記錄 {memoryStats.total_turns_covered} 回合
      </div>
    </Card>
  );
}

/**
 * MemoryIndicatorCompact
 *
 * Compact version showing just memory status badge
 */
export function MemoryIndicatorCompact({ memoryStats }: MemoryIndicatorProps) {
  if (!memoryStats) {
    return null;
  }

  const memoryLevel = memoryStats.short_term_count >= 8 ? 'high' :
                      memoryStats.short_term_count >= 4 ? 'medium' : 'low';

  const colors = {
    high: 'bg-green-500/10 text-green-500 border-green-500/20',
    medium: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20',
    low: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
  };

  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md border ${colors[memoryLevel]}`}>
      <div className="w-2 h-2 rounded-full bg-current animate-pulse" />
      <span className="text-xs font-medium">
        記憶: {memoryStats.short_term_count}/10
      </span>
    </div>
  );
}
