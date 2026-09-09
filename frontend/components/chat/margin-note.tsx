"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

/**
 * The shell a reply's working is set in: a rule down the left margin, a header
 * that opens it, and the steps standing against the rule.
 *
 * `Trace` — a whole turn's working — and `ReasoningPanel` — one `<|thinking|>`
 * block inside an answer — were the same panel written twice: same rail, same
 * header, same collapse threshold, same paragraph splitter, drifting apart
 * independently. They are two callers of this now, which is what keeps the
 * motion authored once instead of twice.
 *
 * ## What moves, and why any of it does
 *
 * The rule draws itself top to bottom when the note lands, and while the turn
 * is still running it *warms toward its end* and turns right into a foot — an
 * L, whose corner is the nib at the point the pen has reached. That is the
 * whole of "still being written": no spinner, no second row, and the same ink
 * (`--throbber-spark`) the live tick and the status row's throbber are carrying
 * at that moment. The foot is drawn only while nothing else is running, because
 * once a call is in flight *its* tick is the warm mark and two would be two
 * answers to one question.
 *
 * The fold has a duration. It was a bare conditional before, so opening a note,
 * closing it, and the auto-collapse when the answer arrives were all instant
 * swaps — and that last one is the moment the panel is actually judged on,
 * because it is where the working gets out of the way of the conclusion.
 *
 * The movements themselves live in `globals.css` under "The trace, in motion",
 * with the durations and the one curve they all share.
 *
 * ## The header is one line, not a label with a readout after it
 *
 * A leader — the hairline off a table of contents — runs from the word to the
 * figures. It gives the note a top edge, and it is what makes the row read as
 * one composed line rather than two things a middot is holding apart.
 *
 * It also replaced a chevron. That was a lucide icon, which is a 2px stroke,
 * which made it the heaviest mark in a panel whose entire argument is
 * hairlines; and the steps underneath already say whether the note is open.
 */
export interface MarginNoteProps {
  /**
   * The header word in both tenses. A note still being written has not been
   * `Worked` yet — which is what it said, until this took two words.
   */
  label: { running: string; done: string }
  /** Figures after the leader, middot-joined — `17s`, `2 calls`, `217 words`. */
  figures: string[]
  /**
   * True while this note is the thing currently being written. A live note is
   * never collapsed: it is the only thing happening, and hiding it would make
   * the model look stalled.
   */
  streaming?: boolean
  /**
   * Draw the rule's foot — the pen resting at the end of the stroke. Held back
   * by the caller while a step of its own is running.
   */
  nib?: boolean
  /**
   * Whether it starts open. Read once and only once: a note that keeps growing
   * must not yank itself shut again after the reader has opened it.
   */
  openInitially: boolean
  /**
   * The steps, in the order they happened. Each is wrapped here in the
   * positioning and stagger context a tick hangs off, so a caller renders
   * content and nothing else.
   */
  steps: React.ReactNode[]
  className?: string
}

/**
 * Past this many steps the stagger stops growing. Six is roughly where a
 * sequence stops reading as one gesture and starts reading as a queue.
 */
const STAGGER_CAP = 6

/**
 * Above this a note opens collapsed — roughly where reasoning stops being a
 * sentence of context and becomes a wall the answer hides behind.
 */
const LONG_NOTE_WORDS = 70

export function MarginNote({
  label,
  figures,
  streaming = false,
  nib = false,
  openInitially,
  steps,
  className,
}: MarginNoteProps) {
  const [open, setOpen] = React.useState(openInitially)
  const expanded = streaming || open

  return (
    <div
      data-trace-open={expanded ? "" : undefined}
      data-trace-live={streaming ? "" : undefined}
      // `em` throughout, so the rule and its indent scale with whatever size the
      // surface set the reply at — same as every other block.
      className={cn("relative my-[1.5em] pl-[1.15em]", className)}
    >
      <span aria-hidden className="trace-rail" />
      {/* Always mounted, so it retracts when the turn closes rather than
          vanishing — a reasoning-only note stays open through that moment. */}
      <span
        aria-hidden
        className="trace-nib"
        data-drawn={nib ? "" : undefined}
      />

      <button
        type="button"
        onClick={() => !streaming && setOpen((v) => !v)}
        aria-expanded={expanded}
        className="group flex w-full items-baseline gap-[0.7em] text-left"
      >
        <span
          className={cn(
            "shrink-0 tag whitespace-nowrap text-foreground",
            // The same shimmer everything mid-flight in the app carries.
            streaming && "text-shimmer"
          )}
        >
          {streaming ? `${label.running}…` : label.done}
        </span>
        <span
          aria-hidden
          className="h-px flex-1 self-center bg-border/35 transition-colors duration-200 group-hover:bg-border"
        />
        <span className="shrink-0 tag whitespace-nowrap text-muted-foreground/60 tabular-nums">
          {figures.join(" · ")}
        </span>
      </button>

      <div className="trace-fold">
        <div className="trace-clip">
          <div className="flex flex-col gap-[0.95em] pt-[0.75em]">
            {steps.map((step, i) => (
              <div
                key={i}
                className="trace-step relative"
                style={
                  { "--i": Math.min(i, STAGGER_CAP) } as React.CSSProperties
                }
              >
                {step}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * A stretch of a note's thinking with `showReasoning` off: the word, and
 * nothing after it.
 *
 * A count or a rule standing in for the hidden text would be measuring
 * something the reader was not shown. Both kinds of note render it — a whole
 * turn's working, and one `<|thinking|>` block inside an answer — because
 * turning reasoning off should mean the same thing in both places.
 */
export function Redacted({ live }: { live: boolean }) {
  return (
    <span
      className={cn("tag text-muted-foreground/70", live && "text-shimmer")}
    >
      {live ? "Thinking…" : "Thought"}
    </span>
  )
}

/**
 * Whether a note shows itself without being asked.
 *
 * A note that called a tool is over the threshold by construction — it has
 * results under it — and always starts shut, which is why the call count is
 * part of the question and not just the length.
 */
export function opensItself(words: number, calls = 0): boolean {
  return calls === 0 && words <= LONG_NOTE_WORDS
}

/**
 * Blank-line-separated paragraphs. Single newlines are soft-wrapped away:
 * models hard-wrap at arbitrary widths, and honouring those breaks produces a
 * ragged column that looks like a bug.
 */
export function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean)
}

/** Words, with any markup markers taken out first so they are not counted. */
export function countWords(text: string): number {
  const bare = text.replace(/<\|[^]*?\|>/g, " ").trim()
  return bare ? bare.split(/\s+/).length : 0
}
