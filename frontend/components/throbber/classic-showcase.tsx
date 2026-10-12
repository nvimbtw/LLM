"use client"

import * as React from "react"

import { ThinkingLabel } from "@/components/chat/thinking-label"
import { TokenFlow } from "@/components/chat/token-flow"
import { TurnClock, workedFor } from "@/components/chat/turn-clock"
import { Panel } from "@/components/ui/panel"
import { Button } from "@/components/ui/button"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"

const MODES: ThrobberMode[] = [
  "idle",
  "loading",
  "thinking",
  "generating",
  "tool",
  "training",
  "benchmarking",
  "tokenizing",
]

/** Roughly the rate a reply streams at, so the wheels here turn at the speed
 * they turn at in a real thread. */
const FIXTURE_TOKENS_PER_S = 24
const FIXTURE_TICK_MS = 90

export function ClassicShowcase() {
  const [mode, setMode] = React.useState<ThrobberMode>("generating")
  // A count that climbs, because a counter that rolls is the only part of the
  // status row that cannot be judged from a still. Started just under the
  // thousand, where the wheels give way to `1.0k` — the one moment in the
  // figure's life worth not having to wait seven minutes for.
  const [tokens, setTokens] = React.useState(940)

  React.useEffect(() => {
    const id = setInterval(
      () =>
        setTokens(
          (n) =>
            (n + Math.round((FIXTURE_TOKENS_PER_S * FIXTURE_TICK_MS) / 1000)) %
            100000
        ),
      FIXTURE_TICK_MS
    )
    return () => clearInterval(id)
  }, [])

  return (
    <div className="flex flex-col gap-6">
      <Panel variant="open" title="Morph">
        <div className="flex justify-center py-6">
          <Throbber mode={mode} />
        </div>
        <div
          className="mt-2 flex flex-wrap gap-1.5"
          role="group"
          aria-label="Throbber state"
        >
          {MODES.map((option) => (
            <Button
              key={option}
              type="button"
              onClick={() => setMode(option)}
              size="sm"
              variant={option === mode ? "secondary" : "ghost"}
              aria-pressed={option === mode}
              className="capitalize"
            >
              {option}
            </Button>
          ))}
        </div>
      </Panel>

      {/* The row the chat thread builds out of these parts, with a live count
          behind it — see `chat-thread.tsx`. Here so the two movements in it can
          be watched without a GPU behind them: the wheels rolling as tokens
          arrive, and the word typing itself in and then walking into the next
          word. */}
      <Panel variant="open" title="Status row">
        <div className="flex flex-col gap-4 py-2">
          <div className="flex items-center gap-3 px-1 text-sm">
            <Throbber mode="thinking" />
            <TurnClock />
            <span aria-hidden className="-mx-1 shrink-0 text-border">
              |
            </span>
            <TokenFlow direction="out" count={tokens} />
            <span aria-hidden className="-mx-1 shrink-0 text-border">
              |
            </span>
            <ThinkingLabel />
          </div>
          <div className="flex items-center gap-3 px-1 text-sm">
            <Throbber mode="tool" />
            <TurnClock />
            <span aria-hidden className="-mx-1 shrink-0 text-border">
              |
            </span>
            <TokenFlow direction="in" count={4210 + Math.floor(tokens / 8)} />
            <span aria-hidden className="-mx-1 shrink-0 text-border">
              |
            </span>
            <span className="text-shimmer font-medium">Search:</span>
            <span className="truncate text-muted-foreground">
              tallest mountain in europe
            </span>
          </div>
          {/* The row a turn leaves behind: the mark at rest, and how long it
              took, in words. */}
          <div className="flex items-center gap-3 px-1 text-sm">
            <Throbber mode="idle" />
            <span className="shrink-0 text-xs text-muted-foreground">
              {workedFor(125)}
            </span>
          </div>
        </div>
      </Panel>

      {/* Every state at once. A figure is not judged by how it looks while
          loading — any of them look fine loading — but by whether eight states
          are eight recognisably different pictures, which can only be seen with
          them side by side. */}
      <Panel variant="open" title="Every state">
        <div className="grid grid-cols-1 gap-x-8 sm:grid-cols-2">
          {MODES.map((option) => (
            <div
              key={option}
              className="flex items-center gap-4 border-b border-border/60 py-4"
            >
              <Throbber mode={option} />
              <p className="min-w-0 text-sm font-medium capitalize">{option}</p>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}
