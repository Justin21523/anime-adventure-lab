/**
 * RecentMemories Component
 *
 * Displays recent story memories with expandable details.
 * Shows short-term memories and summaries.
 */

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ShortTermMemory {
  turn: number;
  action: string;
  result: string;
  scene?: string;
}

interface MemorySummary {
  turn_range: string;
  summary: string;
  key_events: string[];
}

interface RecentMemoriesProps {
  shortTerm?: ShortTermMemory[];
  summaries?: MemorySummary[];
  className?: string;
}

export function RecentMemories({
  shortTerm = [],
  summaries = [],
  className = ''
}: RecentMemoriesProps) {
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null);
  const [expandedSummary, setExpandedSummary] = useState<string | null>(null);

  if (shortTerm.length === 0 && summaries.length === 0) {
    return (
      <Card className={`p-4 ${className}`}>
        <div className="text-center text-sm text-muted-foreground">
          尚無記憶記錄
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-4 space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">最近記憶</h3>
        <Badge variant="outline">
          {shortTerm.length} 回合
        </Badge>
      </div>

      {/* Recent Summaries */}
      {summaries.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            歷史摘要
          </div>
          {summaries.map((summary) => (
            <div
              key={summary.turn_range}
              className="border border-border rounded-lg overflow-hidden"
            >
              <button
                onClick={() => setExpandedSummary(
                  expandedSummary === summary.turn_range ? null : summary.turn_range
                )}
                className="w-full p-2 flex items-center justify-between hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    回合 {summary.turn_range}
                  </Badge>
                  <span className="text-xs text-foreground/80 truncate">
                    {summary.summary.substring(0, 50)}...
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {expandedSummary === summary.turn_range ? '▼' : '▶'}
                </span>
              </button>

              {expandedSummary === summary.turn_range && (
                <div className="p-3 bg-muted/30 border-t border-border space-y-2">
                  <div className="text-xs text-foreground/90">
                    {summary.summary}
                  </div>
                  {summary.key_events.length > 0 && (
                    <div className="pt-2 border-t border-border/50">
                      <div className="text-xs font-medium text-muted-foreground mb-1">
                        關鍵事件:
                      </div>
                      <ul className="space-y-1">
                        {summary.key_events.map((event, idx) => (
                          <li key={idx} className="text-xs text-foreground/80 flex items-start gap-1.5">
                            <span className="text-muted-foreground">•</span>
                            <span>{event}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Short-term Memories */}
      {shortTerm.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            最近回合
          </div>
          <div className="space-y-1.5">
            {shortTerm.slice().reverse().map((memory) => (
              <div
                key={memory.turn}
                className="border border-border rounded-lg overflow-hidden"
              >
                <button
                  onClick={() => setExpandedTurn(
                    expandedTurn === memory.turn ? null : memory.turn
                  )}
                  className="w-full p-2 flex items-center justify-between hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Badge variant="default" className="text-xs">
                      #{memory.turn}
                    </Badge>
                    <span className="text-xs text-foreground/80 truncate">
                      {memory.action}
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {expandedTurn === memory.turn ? '▼' : '▶'}
                  </span>
                </button>

                {expandedTurn === memory.turn && (
                  <div className="p-3 bg-muted/30 border-t border-border space-y-2">
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-1">
                        玩家行動:
                      </div>
                      <div className="text-xs text-foreground/90">
                        {memory.action}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-1">
                        結果:
                      </div>
                      <div className="text-xs text-foreground/90">
                        {memory.result}
                      </div>
                    </div>
                    {memory.scene && (
                      <div className="pt-2 border-t border-border/50">
                        <div className="text-xs text-muted-foreground">
                          場景: <span className="text-foreground/80">{memory.scene}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

/**
 * RecentMemoriesCompact
 *
 * Compact timeline view of recent memories
 */
export function RecentMemoriesCompact({ shortTerm = [] }: Pick<RecentMemoriesProps, 'shortTerm'>) {
  if (shortTerm.length === 0) {
    return null;
  }

  return (
    <div className="space-y-1">
      {shortTerm.slice(-3).reverse().map((memory, idx) => (
        <div
          key={memory.turn}
          className="flex items-start gap-2 text-xs"
          style={{ opacity: 1 - (idx * 0.3) }}
        >
          <Badge variant="outline" className="text-xs shrink-0">
            #{memory.turn}
          </Badge>
          <span className="text-foreground/70 line-clamp-1">
            {memory.action}
          </span>
        </div>
      ))}
    </div>
  );
}
