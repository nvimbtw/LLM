"use client"

import * as React from "react"

import { ThinkingLabel } from "@/components/chat/thinking-label"
import { TokenFlow } from "@/components/chat/token-flow"
import { TurnClock, workedFor } from "@/components/chat/turn-clock"
import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

const MODES: { mode: ThrobberMode; description: string }[] = [
  { mode: "idle", description: "Nothing grown — the root dot, breathing." },
  { mode: "loading", description: "Three limbs, run out and cleared." },
  {
    mode: "thinking",
    description: "A broad, shallow bush, grown and withdrawn.",
  },
  {
    mode: "generating",
    description: "One long route, a different one every pass.",
  },
  { mode: "tool", description: "The whole net struck at once." },
  { mode: "training", description: "The whole net held, breathing." },
  { mode: "benchmarking", description: "Limbs arriving one at a time, kept." },
  {
    mode: "tokenizing",
    description: "The grown tree, cut into travelling pieces.",
  },
]

/** Roughly the rate a reply streams at, so the wheels here turn at the speed
 * they turn at in a real thread. */
const FIXTURE_TOKENS_PER_S = 24
const FIXTURE_TICK_MS = 90

export function ThrobberShowcase() {
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
      <Panel
        title="Morph"
        description="Switch state — the figure changes pace and regrows in the new character"
      >
        <div className="flex justify-center py-6">
          <Throbber mode={mode} />
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {MODES.map(({ mode: option }) => (
            <button
              key={option}
              type="button"
              onClick={() => setMode(option)}
              className={cn(
                "rounded-(--radius-marker) border px-3 py-1 text-xs capitalize transition-colors",
                option === mode
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border text-muted-foreground hover:border-input hover:text-foreground"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </Panel>

      {/* The row the chat thread builds out of these parts, with a live count
          behind it — see `chat-thread.tsx`. Here so the two movements in it can
          be watched without a GPU behind them: the wheels rolling as tokens
          arrive, and the word typing itself in and then walking into the next
          word. */}
      <Panel
        title="Status row"
        description="Mark, count, rule, label — as the chat thread assembles it"
      >
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
      <Panel
        title="Every state"
        description="Eight different pictures, not one picture at eight speeds"
      >
        <div className="grid grid-cols-1 gap-px overflow-hidden rounded-(--radius-control-sm) border border-border bg-border/60 sm:grid-cols-2">
          {MODES.map(({ mode: option, description }) => (
            <div
              key={option}
              className="flex items-center gap-4 bg-background px-5 py-4"
            >
              <Throbber mode={option} />
              <div className="min-w-0">
                <p className="text-sm font-medium capitalize">{option}</p>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  {description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}
