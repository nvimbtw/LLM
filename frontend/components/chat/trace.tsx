"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import {
  isText,
  parseMessage,
  callOf,
  type MNode,
  type ToolCall,
} from "@/lib/message-parser"
import { useSettings } from "@/components/settings/settings-provider"
import { Inked } from "./inked"
import {
  MarginNote,
  Redacted,
  countWords,
  opensItself,
  splitParagraphs,
} from "./margin-note"
import { TraceStep } from "./trace-step"
import type { ToolAnswer, Interjection } from "@/lib/chat-messages"

/**
 * The working of one turn: everything the model did before the answer.
 *
 * A turn is not always one generation. The models here are too small to stop at
 * a tool call, so a reply that searches runs as several rounds — the reply that
 * called, the tool's answer, then a reply that either uses it or calls again —
 * and every round up to the one that stops calling is the working, together with
 * what each call came back with. This panel is that phase, in the order it
 * happened; the answer renders beneath it as an ordinary message.
 *
 * It is one chain however many rounds it took. A round that calls again is not a
 * new trace under the old one: the reader asked one question, and the model
 * reconsidering after a failed command is the same piece of work continuing.
 *
 * ## The rail is the timeline
 *
 * The shell — rule, header, fold — is `MarginNote`, shared with
 * `ReasoningPanel`. Reasoning is the prose running *against* that rail, and a
 * tool call is a short tick *off* it: a marginal reference mark, which is the
 * print idiom the rest of the reply is already set in. That is deliberately not
 * the icon-per-row treatment other products use; the rows there float, and need
 * an icon to say what they are. These do not.
 *
 * ## It stays grey, and only the live step is not
 *
 * The rail lost a saturated colour once already (see `ReasoningPanel`) because a
 * margin note must not be the loudest thing in a reply, and that verdict holds
 * here. The exception is the point the pen has reached — a step *currently
 * running*, or the rule's own foot when nothing is — which warms to
 * `--throbber-spark`, the same token the throbber in the status row is showing
 * at that moment. So the trace is achromatic and the live edge is the only warm
 * mark on the page.
 *
 * ## Reasoning turned off redacts it; it does not delete it
 *
 * With `showReasoning` off, the prose steps become the single word `Thought`
 * and the calls are untouched — a query and what it returned are not reasoning.
 * The shape of the work is still the thing worth reading, and it is still the
 * honest account of what the reply was built from; only the words of the
 * thinking go. This panel used not to consult the setting at all, so turning
 * reasoning off hid the thinking *inside* an answer and printed the whole of it
 * above one.
 */
export type { ToolAnswer, Interjection } from "@/lib/chat-messages"

export interface TraceProps {
  /** The first round's reply: reasoning, prose, and the calls it made. */
  working: string
  /** What each call came back with, in the order the calls closed. */
  answers: ToolAnswer[]
  /** A person's steer, threaded in after the call it followed. */
  interjections?: Interjection[]
  /** Seconds the whole turn took, summed across its rounds. */
  elapsedSecs?: number
  /** True while the turn is still running. A live trace is never collapsed. */
  streaming?: boolean
  className?: string
}

/** A turn that called nothing is reasoning, in either tense. */
const REASONING = { running: "Reasoning", done: "Reasoning" }
/** One that reached for something is an account of work, and the tense matters
 *  while it is still happening — this said "Worked…" mid-turn. */
const WORKING = { running: "Working", done: "Worked" }

export function Trace({
  working,
  answers,
  interjections = [],
  elapsedSecs,
  streaming = false,
  className,
}: TraceProps) {
  const { settings } = useSettings()
  const steps = React.useMemo(
    () => stepsOf(working, answers, interjections),
    [working, answers, interjections]
  )
  const words = React.useMemo(() => countWords(working), [working])
  const calls = steps.filter((s) => s.kind === "tool").length

  if (steps.length === 0) return null

  // The rule's foot is the pen at rest. A call in flight has a warm tick of its
  // own, and two warm marks would be two answers to the same question.
  const nib = streaming && !steps.some((s) => s.kind === "tool" && !s.body)
  const last = steps.length - 1

  return (
    <MarginNote
      label={calls === 0 ? REASONING : WORKING}
      figures={figuresOf(calls, words, elapsedSecs)}
      streaming={streaming}
      nib={nib}
      openInitially={opensItself(words, calls)}
      className={className}
      steps={steps.map((step, i) =>
        step.kind === "tool" ? (
          <TraceStep
            key={i}
            tool={step.tool}
            argument={step.argument}
            body={step.body}
            running={step.body === null}
          />
        ) : step.kind === "interject" ? (
          <Interject key={i} text={step.text} />
        ) : settings.showReasoning ? (
          <Prose key={i} text={step.text} live={streaming && i === last} />
        ) : (
          <Redacted key={i} live={streaming && i === last} />
        )
      )}
    />
  )
}

/**
 * The header's figures. A turn that called nothing keeps the wording it has
 * always had — the common case must not pay for the feature — and only one that
 * reached for something becomes an account of the work.
 *
 * "call", not "search": a turn can run a shell as well, and the header counts
 * steps of every kind. Which tool each one was is the step's own line.
 */
function figuresOf(
  calls: number,
  words: number,
  elapsedSecs?: number
): string[] {
  if (calls === 0) return [`${words.toLocaleString()} words`]
  const parts: string[] = []
  if (elapsedSecs != null) parts.push(`${Math.round(elapsedSecs)}s`)
  parts.push(calls === 1 ? "1 call" : `${calls} calls`)
  return parts
}

function Prose({ text, live }: { text: string; live: boolean }) {
  const ref = React.useRef<HTMLDivElement>(null)

  // The newest line on the bottom edge of the cap. Written straight to the node
  // rather than through state: this runs on every token of a long think, and a
  // re-render to move a scroll offset is a re-render for nothing.
  React.useEffect(() => {
    const node = ref.current
    if (live && node) node.scrollTop = node.scrollHeight
  })

  // Set a step down from the answer and in the muted ink: this is the working,
  // and it should be legible without ever competing with the conclusion.
  return (
    <div
      ref={ref}
      className={cn(
        "space-y-[0.75em] text-[0.92em] leading-[1.7] text-muted-foreground",
        live && "trace-live-prose"
      )}
    >
      {splitParagraphs(text).map((paragraph, i) => (
        <p key={i}>{live ? <Inked text={paragraph} /> : paragraph}</p>
      ))}
    </div>
  )
}

/**
 * A steer that actually reached the model mid-turn. Same achromatic register
 * as a reasoning step — this is still the model's working, just with someone
 * else's line in it — set apart only by the label, the same "who said this"
 * device `queued-message.tsx` uses before it lands here. It survives reasoning
 * being turned off, because it is not the model's reasoning.
 */
function Interject({ text }: { text: string }) {
  return (
    <div className="space-y-[0.35em]">
      <span className="tag text-muted-foreground/60">you interjected</span>
      <p className="text-[0.92em] leading-[1.7] text-muted-foreground">
        {text}
      </p>
    </div>
  )
}

type Step =
  | { kind: "prose"; text: string }
  | { kind: "tool"; tool: string; argument: string; body: string | null }
  | { kind: "interject"; text: string }

/**
 * The call in `working` that is still waiting on its answer, if there is one.
 *
 * Exported so the thread's status row can name what is running off the same
 * reading the trace's own live tick uses — two answers to that question would
 * show as one warm mark beside a throbber claiming the model was still writing.
 * The *first* unanswered call, because answers are consumed positionally and
 * that is the one the backend is working on.
 */
export function callInFlight(
  working: string | undefined,
  answers: ToolAnswer[]
): ToolCall | null {
  if (working === undefined) return null
  const step = stepsOf(working, answers).find(
    (s) => s.kind === "tool" && s.body === null
  )
  return step?.kind === "tool"
    ? { tool: step.tool, argument: step.argument }
    : null
}

/**
 * The working, cut into steps in the order they happened.
 *
 * Reasoning and ordinary prose both become prose steps — inside the trace the
 * distinction has stopped mattering, because everything here is working rather
 * than answer. Consecutive ones are joined so a `<|thinking|>` block followed by
 * a `<|par|>` reads as continuous thought instead of two stacked panels. It is
 * also why turning reasoning off redacts a whole prose step rather than the
 * `<|thinking|>` inside it: by this point there is no inside.
 *
 * `answers` is consumed positionally: the nth call in the markup is answered by
 * the nth entry. A call with no entry yet is one still running, which is exactly
 * the state mid-stream.
 */
function stepsOf(
  working: string,
  answers: ToolAnswer[],
  interjections: Interjection[] = []
): Step[] {
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
    const call = callOf(node)
    if (call) {
      const answer = answers[answered++]
      steps.push({
        kind: "tool",
        tool: call.tool,
        argument: call.argument,
        body: answer?.body ?? null,
      })
      // Placed the moment the count they were captured against is reached —
      // several can land on the same call, in the order they arrived.
      for (const interjection of interjections) {
        if (interjection.afterAnswers === answered) {
          steps.push({ kind: "interject", text: interjection.text })
        }
      }
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
  const separator =
    node.type === "paragraph" || node.type === "thinking" ? "\n\n" : ""
  return separator + node.children.map(plainText).join("") + separator
}
