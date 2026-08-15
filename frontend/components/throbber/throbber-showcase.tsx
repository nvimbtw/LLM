"use client"

import * as React from "react"

import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

const MODES: { mode: ThrobberMode; description: string }[] = [
  { mode: "idle", description: "A short segment resting at the centre." },
  {
    mode: "loading",
    description: "A wide segment crossing the rail and wrapping.",
  },
  {
    mode: "thinking",
    description: "Colour blooms out of the middle and draws back in.",
  },
  {
    mode: "generating",
    description: "A short segment running the rail, fast and warm.",
  },
  {
    mode: "tool",
    description: "The whole rail lights at once and falls back.",
  },
  {
    mode: "training",
    description: "The rail is held full, pulsing warm and cool as one.",
  },
  {
    mode: "benchmarking",
    description: "A fill advancing in discrete steps, then resetting.",
  },
  {
    mode: "tokenizing",
    description: "The fill is cut into pieces, and the cuts travel.",
  },
]

export function ThrobberShowcase() {
  // The morph is the whole point of the rewrite, so the page leads with a control
  // that exercises it rather than only showing the states in isolation.
  const [mode, setMode] = React.useState<ThrobberMode>("loading")

  return (
    <div className="flex flex-col gap-6">
      <Panel
        title="Morph"
        description="Switch state — the segment changes pace, it never re-starts"
      >
        <div className="flex justify-center py-2">
          <Throbber mode={mode} />
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {MODES.map(({ mode: option }) => (
            <button
              key={option}
              type="button"
              onClick={() => setMode(option)}
              className={cn(
                "rounded-full border px-3 py-1 text-xs capitalize transition-colors",
                option === mode
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border text-muted-foreground hover:text-foreground"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex flex-col gap-px overflow-hidden rounded-lg border border-border bg-border/60">
        {MODES.map(({ mode: option, description }) => (
          <div key={option} className="bg-background px-5 py-4">
            <div className="flex items-baseline justify-between gap-4">
              <span className="text-sm font-medium capitalize">{option}</span>
              <span className="text-xs text-muted-foreground">
                {description}
              </span>
            </div>
            <div className="mt-3 flex justify-center">
              <Throbber mode={option} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
