"use client"

import * as React from "react"

import {
  MarginNote,
  Redacted,
  countWords,
  opensItself,
  splitParagraphs,
} from "./margin-note"

/**
 * A single `<|thinking|>` block, rendered inline at the point in the reply where
 * the model reasoned (there may be several per message).
 *
 * It used to be the raw trace dumped into one `whitespace-pre-wrap` paragraph
 * under a label — which meant a long reasoning block buried the answer it
 * belonged to, and the model's own paragraph breaks were rendered as literal
 * blank lines inside a single run of text. Three things changed:
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
 * margin note beside the answer, not a second answer. It *is* one: the rail, the
 * header and the fold are `MarginNote`, shared with `Trace`, which was the same
 * panel written a second time until the two were pulled together. Streams in
 * place, so partial reasoning renders mid-thought.
 */
export interface ReasoningPanelProps {
  text: string
  /**
   * True while this block is the one currently being written. A streaming trace
   * is never collapsed — it is the only thing happening, and hiding it would
   * make the model look stalled.
   */
  streaming?: boolean
  /**
   * `showReasoning` is off. The note stays, and stays openable, and says how
   * much is behind it; only the words of the thinking go. It used to render
   * nothing at all, which meant a reader who turned reasoning off lost every
   * sign that the model had done any.
   */
  redacted?: boolean
  className?: string
}

export function ReasoningPanel({
  text,
  streaming = false,
  redacted = false,
  className,
}: ReasoningPanelProps) {
  const trimmed = text.trim()
  const words = React.useMemo(() => countWords(trimmed), [trimmed])
  const paragraphs = React.useMemo(
    () => (redacted ? [] : splitParagraphs(trimmed)),
    [trimmed, redacted]
  )

  if (!trimmed) return null

  return (
    <MarginNote
      label={LABEL}
      figures={[`${words.toLocaleString()} words`]}
      streaming={streaming}
      // Nothing here calls a tool, so the pen is always at the end of the rule
      // while this is being written.
      nib={streaming}
      openInitially={opensItself(words)}
      className={className}
      steps={
        redacted
          ? [<Redacted key="redacted" live={streaming} />]
          : [
              // One step, not one per paragraph: this is a single stretch of
              // thought, and staggering its paragraphs in would imply they
              // arrived as separate pieces of work.
              <div
                key="prose"
                // Set a step down from the answer and in the muted ink: this is
                // the working, and it should be legible without ever competing
                // with the conclusion sitting next to it.
                className="space-y-[0.75em] text-[0.92em] leading-[1.7] text-muted-foreground"
              >
                {paragraphs.map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </div>,
            ]
      }
    />
  )
}

/** Reasoning in either tense — unlike a trace, this never becomes "Worked". */
const LABEL = { running: "Reasoning", done: "Reasoning" }
