"use client"

import * as React from "react"
import { ChevronRight } from "lucide-react"

import { cn } from "@/lib/utils"
import {
  isText,
  parseMessage,
  textContent,
  toolOf,
  type MNode,
} from "@/lib/message-parser"
import { TraceStep } from "./trace-step"

/**
 * The working of one turn: everything the model did before the answer.
 *
 * A turn is not always one generation. The models here are too small to stop at
 * a tool call, so a reply that searches runs as *two* rounds — the reply that
 * called, the tool's answer, then the reply that uses it — and the whole of the
 * first round plus what came back is the working. This panel is that phase, in
 * the order it happened; the answer renders beneath it as an ordinary message.
 *
 * ## The rail is the timeline
 *
 * `ReasoningPanel` already sets reasoning apart with a hairline rule down its
 * left edge. Nothing else was needed: reasoning is the prose running *against*
 * that rail, and a tool call is a short tick *off* it — a marginal reference
 * mark, which is the print idiom the rest of the reply is already set in. That
 * is deliberately not the icon-per-row treatment other products use; the rows
 * there float, and need an icon to say what they are. These do not.
 *
 * ## It stays grey, and only the live step is not
 *
 * The rail lost a saturated colour once already (see `ReasoningPanel`) because a
 * margin note must not be the loudest thing in a reply, and that verdict holds
 * here. The one exception is a step *currently running*: its tick warms to
 * `--throbber-spark`, the same token the throbber in the status row is showing
 * at that moment, and cools back to the hairline when the result lands. So the
 * trace is achromatic and the live step is the only warm mark on the page.
 */
export interface ToolAnswer {
  tool: string
  argument: string
  /** The result, in the marker vocabulary — a `<|list|>` or a `<|warn|>`. */
  body: string
}

export interface TraceProps {
  /** The first round's reply: reasoning, prose, and the calls it made. */
  working: string
  /** What each call came back with, in the order the calls closed. */
  answers: ToolAnswer[]
  /** Seconds the whole turn took, summed across its rounds. */
  elapsedSecs?: number
  /** True while the turn is still running. A live trace is never collapsed. */
  streaming?: boolean
  className?: string
}

// Above this a reasoning-only trace opens collapsed — roughly where reasoning
// stops being a sentence of context and becomes a wall the answer hides behind.
// A trace that called a tool is past that by construction and always starts shut.
const LONG_TRACE_WORDS = 70

export function Trace({
  working,
  answers,
  elapsedSecs,
  streaming = false,
  className,
}: TraceProps) {
  const steps = React.useMemo(() => stepsOf(working, answers), [working, answers])
  const words = React.useMemo(() => countWords(working), [working])
  const calls = steps.filter((s) => s.kind === "tool").length

  // Only the initial value depends on the content — once the reader has opened
  // or closed it, a trace that keeps growing must not yank itself shut again.
  const [open, setOpen] = React.useState(() => calls === 0 && words <= LONG_TRACE_WORDS)
  const expanded = streaming || open

  if (steps.length === 0) return null

  return (
    <div
      className={cn(
        // `em` throughout, so the rail and its indent scale with whatever size
        // the surface set the reply at — same as every other block.
        "my-[1.5em] border-l border-border pl-[1.15em]",
        className
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={expanded}
        className="flex items-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronRight
          aria-hidden
          className={cn(
            "size-[0.85em] shrink-0 transition-transform duration-200",
            expanded && "rotate-90",
            // A live trace cannot be collapsed, so the affordance goes away
            // rather than sitting there inert.
            streaming && "opacity-0"
          )}
        />
        <Summary
          calls={calls}
          words={words}
          elapsedSecs={elapsedSecs}
          streaming={streaming}
        />
      </button>

      {expanded && (
        <div className="mt-[0.7em] space-y-[0.9em]">
          {steps.map((step, i) =>
            step.kind === "tool" ? (
              <TraceStep
                key={i}
                tool={step.tool}
                argument={step.argument}
                body={step.body}
                running={step.body === null}
              />
            ) : (
              <Prose key={i} text={step.text} />
            )
          )}
        </div>
      )}
    </div>
  )
}

/**
 * The header line. A turn that called nothing keeps the wording it has always
 * had — the common case must not pay for the feature — and only one that
 * reached for something becomes an account of the work.
 */
function Summary({
  calls,
  words,
  elapsedSecs,
  streaming,
}: {
  calls: number
  words: number
  elapsedSecs?: number
  streaming: boolean
}) {
  const label = calls === 0 ? "Reasoning" : "Worked"
  const parts: string[] = []
  if (calls > 0 && elapsedSecs != null) parts.push(`${Math.round(elapsedSecs)}s`)
  if (calls > 0) parts.push(calls === 1 ? "1 search" : `${calls} searches`)
  if (calls === 0) parts.push(`${words.toLocaleString()} words`)

  return (
    <>
      <span
        className={cn(
          "font-mono text-[0.72em] tracking-[0.18em] uppercase",
          // The same shimmer the thread's status row uses, so a trace still
          // being written says so the way everything else in the app says it.
          streaming && "text-shimmer"
        )}
      >
        {streaming ? `${label}…` : label}
      </span>
      {parts.map((part) => (
        <React.Fragment key={part}>
          {/* Middot-separated, matching the byline under a reply — a gap alone
              read as two labels rather than one line of attribution. */}
          <span aria-hidden className="text-muted-foreground/35">
            ·
          </span>
          <span className="text-[0.72em] tabular-nums">{part}</span>
        </React.Fragment>
      ))}
    </>
  )
}

function Prose({ text }: { text: string }) {
  // Set a step down from the answer and in the muted ink: this is the working,
  // and it should be legible without ever competing with the conclusion.
  return (
    <div className="space-y-[0.75em] text-[0.92em] leading-[1.7] text-muted-foreground">
      {splitParagraphs(text).map((paragraph, i) => (
        <p key={i}>{paragraph}</p>
      ))}
    </div>
  )
}

type Step =
  | { kind: "prose"; text: string }
  | { kind: "tool"; tool: string; argument: string; body: string | null }

/**
 * The working, cut into steps in the order they happened.
 *
 * Reasoning and ordinary prose both become prose steps — inside the trace the
 * distinction has stopped mattering, because everything here is working rather
 * than answer. Consecutive ones are joined so a `<|thinking|>` block followed by
 * a `<|par|>` reads as continuous thought instead of two stacked panels.
 *
 * `answers` is consumed positionally: the nth call in the markup is answered by
 * the nth entry. A call with no entry yet is one still running, which is exactly
 * the state mid-stream.
 */
/**
 * Whether a call in `working` is still waiting on its answer.
 *
 * Exported so the thread's status row can say "Searching…" off the same reading
 * the trace's own live tick uses — two answers to that question would show as
 * one warm mark beside a throbber claiming the model was still writing.
 */
export function isCalling(working: string | undefined, answers: ToolAnswer[]): boolean {
  if (working === undefined) return false
  return stepsOf(working, answers).some((s) => s.kind === "tool" && s.body === null)
}

function stepsOf(working: string, answers: ToolAnswer[]): Step[] {
  const steps: Step[] = []
  let answered = 0

  const pushProse = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    const last = steps[steps.length - 1]
    if (last?.kind === "prose") last.text += `\n\n${trimmed}`
    else steps.push({ kind: "prose", text: trimmed })
  }

  for (const node of parseMessage(working).children) {
    const tool = toolOf(node)
    if (tool) {
      const answer = answers[answered++]
      steps.push({
        kind: "tool",
        tool,
        argument: textContent(node).trim(),
        body: answer?.body ?? null,
      })
      continue
    }
    pushProse(plainText(node))
  }

  return steps
}

/**
 * A node's readable text. Markup inside the working is flattened rather than
 * rendered: the trace is one voice at one size, and a heading or a code block in
 * it would out-shout the answer it is a note beside.
 */
function plainText(node: MNode): string {
  if (isText(node)) return node.value
  // Blocks are separated so two paragraphs do not run into one another as a
  // single unbroken line.
  const separator = node.type === "paragraph" || node.type === "thinking" ? "\n\n" : ""
  return separator + node.children.map(plainText).join("") + separator
}

function countWords(text: string): number {
  const bare = text.replace(/<\|[^]*?\|>/g, " ").trim()
  return bare ? bare.split(/\s+/).length : 0
}

/**
 * Blank-line-separated paragraphs. Single newlines are soft-wrapped away: models
 * hard-wrap at arbitrary widths, and honouring those breaks produces a ragged
 * column that looks like a bug.
 */
function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean)
}
