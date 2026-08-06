"use client"

import * as React from "react"

import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

const MODES: { mode: ThrobberMode; description: string }[] = [
  { mode: "idle", description: "The shape is held. Present, not working." },
  {
    mode: "loading",
    description: "The whole figure turns and reshapes, evenly.",
  },
  {
    mode: "thinking",
    description: "A two-ring band of colour scans outward.",
  },
  {
    mode: "generating",
    description: "Quick, with a swell running out through the rings.",
  },
  {
    mode: "tool",
    description: "Every ring flashes to the second colour at once.",
  },
  {
    mode: "training",
    description: "The whole figure breathes, every ring together.",
  },
  {
    mode: "benchmarking",
    description: "One ring at a time takes the colour and hands it on.",
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
        description="Switch state — the figure changes pace, it never re-starts"
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
