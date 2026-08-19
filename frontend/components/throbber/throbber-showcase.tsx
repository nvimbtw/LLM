"use client"

import * as React from "react"

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

export function ThrobberShowcase() {
  const [mode, setMode] = React.useState<ThrobberMode>("generating")

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
                "rounded-sm border px-3 py-1 text-xs capitalize transition-colors",
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

      {/* Every state at once. A figure is not judged by how it looks while
          loading — any of them look fine loading — but by whether eight states
          are eight recognisably different pictures, which can only be seen with
          them side by side. */}
      <Panel
        title="Every state"
        description="Eight different pictures, not one picture at eight speeds"
      >
        <div className="grid grid-cols-1 gap-px overflow-hidden rounded-md border border-border bg-border/60 sm:grid-cols-2">
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
