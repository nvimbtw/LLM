"use client"

import * as React from "react"

import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

const MODES: { mode: ThrobberMode; description: string }[] = [
  { mode: "idle", description: "At rest. The rule a page already has, barely breathing." },
  { mode: "loading", description: "A full-width sweep, decelerating into each turn." },
  { mode: "thinking", description: "Hovering near centre, drifting. Undecided, never cyclic." },
  { mode: "generating", description: "Advancing steadily, inking the rule behind it." },
  { mode: "tool", description: "Snapping between fixed stops and holding. Mechanical." },
  { mode: "training", description: "Repeated passes, the rule building weight and releasing." },
  { mode: "benchmarking", description: "Stepping between graduations, dwelling on each reading." },
]

export function ThrobberShowcase() {
  // The morph is the whole point of the rewrite, so the page leads with a control
  // that exercises it rather than only showing the states in isolation.
  const [mode, setMode] = React.useState<ThrobberMode>("loading")

  return (
    <div className="flex flex-col gap-6">
      <Panel title="Morph" description="Switch state — the carriage travels, it never jumps">
        <Throbber mode={mode} className="h-10 w-full" />
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
                  : "border-border/60 text-muted-foreground hover:text-foreground"
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex flex-col gap-px overflow-hidden rounded-xl border border-border/60 bg-border/60">
        {MODES.map(({ mode: option, description }) => (
          <div key={option} className="bg-background px-5 py-4">
            <div className="flex items-baseline justify-between gap-4">
              <span className="text-sm font-medium capitalize">{option}</span>
              <span className="text-[11px] text-muted-foreground/70">{description}</span>
            </div>
            <Throbber mode={option} className="mt-3 h-7 w-full" />
          </div>
        ))}
      </div>
    </div>
  )
}
