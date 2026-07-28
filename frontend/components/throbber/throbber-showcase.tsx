"use client"

import * as React from "react"

import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

const MODES: { mode: ThrobberMode; description: string }[] = [
  { mode: "idle", description: "Barely turning. Present, not working." },
  { mode: "loading", description: "A steady, unhurried turn. Smooth surface." },
  {
    mode: "thinking",
    description: "A charge runs the seams. Neurons firing.",
  },
  {
    mode: "generating",
    description: "Fast, with a tight ripple running across the body.",
  },
  {
    mode: "tool",
    description: "Single faces stab out and withdraw. Spiky.",
  },
  {
    mode: "training",
    description: "The whole ball breathes, every face at once.",
  },
  {
    mode: "benchmarking",
    description: "One face at a time lifts, holds, settles.",
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
        description="Switch state — the ball changes pace, it never re-starts"
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
