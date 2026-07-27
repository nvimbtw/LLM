"use client"

import * as React from "react"
import { ChevronRight } from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * A single `<|thinking|>` block, rendered inline at the point in the reply where
 * the model reasoned (there may be several per message).
 *
 * It used to be the raw trace dumped into one `whitespace-pre-wrap` paragraph
 * under a label — which meant a long reasoning block buried the answer it belonged
 * to, and the model's own paragraph breaks were rendered as literal blank lines
 * inside a single run of text. Three things changed:
 *
 * - It is **collapsible**, and collapses itself once it is long. Reasoning is
 *   supporting material; the answer is the thing being read. Short traces stay
 *   open because hiding two lines behind a click is worse than showing them.
 * - The text is **broken into paragraphs** on blank lines, so the model's own
 *   structure survives instead of being flattened.
 * - The header carries the **length**, so a collapsed block says how much is
 *   behind it rather than being an unlabelled door.
 *
 * Still deliberately quiet — a hairline rail and muted text — so it reads as a
 * margin note beside the answer, not a second answer. Streams in place, so
 * partial reasoning renders mid-thought.
 */
export interface ReasoningPanelProps {
  text: string
  /**
   * True while this block is the one currently being written. A streaming trace
   * is never collapsed — it is the only thing happening, and hiding it would
   * make the model look stalled.
   */
  streaming?: boolean
  className?: string
}

// Above this, a trace opens collapsed. Roughly the point where reasoning stops
// being a sentence of context and starts being a wall the answer hides behind.
const LONG_TRACE_WORDS = 70

export function ReasoningPanel({
  text,
  streaming = false,
  className,
}: ReasoningPanelProps) {
  const trimmed = text.trim()
  const words = React.useMemo(
    () => (trimmed ? trimmed.split(/\s+/).length : 0),
    [trimmed]
  )
  const paragraphs = React.useMemo(() => splitParagraphs(trimmed), [trimmed])

  // Only the initial value depends on length — once the reader has opened or
  // closed it, a trace that keeps growing must not yank itself shut again.
  const [open, setOpen] = React.useState(() => words <= LONG_TRACE_WORDS)
  const expanded = streaming || open

  if (!trimmed) return null

  return (
    <div className={cn("my-4 border-l-2 border-chart-5/40 pl-3.5", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={expanded}
        className="group/reasoning -ml-0.5 flex items-center gap-1.5 text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronRight
          aria-hidden
          className={cn(
            "size-3 transition-transform",
            expanded && "rotate-90",
            // A streaming trace cannot be collapsed, so the affordance goes away
            // rather than sitting there inert.
            streaming && "opacity-0"
          )}
        />
        <span className="font-mono text-xs tracking-[0.14em] uppercase">
          {streaming ? "Reasoning…" : "Reasoning"}
        </span>
        <span className="text-xs tabular-nums opacity-70">
          {words.toLocaleString()} words
        </span>
      </button>

      {expanded && (
        <div className="mt-1.5 space-y-2 text-[0.94em] leading-relaxed text-muted-foreground">
          {paragraphs.map((paragraph, i) => (
            <p key={i}>{paragraph}</p>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Blank-line-separated paragraphs. Single newlines are soft-wrapped away: models
 * hard-wrap reasoning at arbitrary widths, and honouring those breaks produces a
 * ragged column that looks like a bug.
 */
function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean)
}
