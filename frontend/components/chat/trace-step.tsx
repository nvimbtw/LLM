"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import {
  parseMessage,
  textContent,
  type MElement,
  type MNode,
} from "@/lib/message-parser"

/**
 * One tool call inside a `Trace`: what was asked, and what came back.
 *
 * Rendered as a tick off the trace's rail rather than a row with an icon on it.
 * The rail is already there and already means "this is the working"; a mark that
 * meets it says "and here it reached outside" without introducing a second
 * visual language for the same idea.
 *
 * Two levels of disclosure, because a turn can make several calls and five
 * results each would bury the answer: the line names the search and counts what
 * it found, and the results appear on a click.
 */
export interface TraceStepProps {
  /** The tool's wire name — `web-search`. */
  tool: string
  argument: string
  /** The result in the marker vocabulary, or null while the call is running. */
  body: string | null
  running: boolean
  className?: string
}

/**
 * How a completed call is announced. Past tense, because by the time the trace
 * is read it has happened. A tool with no entry here falls back to its own name,
 * so adding one is not a prerequisite for it working.
 */
const TOOL_VERBS: Record<string, string> = { "web-search": "Searched" }

export function TraceStep({
  tool,
  argument,
  body,
  running,
  className,
}: TraceStepProps) {
  const { items, failed } = React.useMemo(() => resultsOf(body), [body])
  const [open, setOpen] = React.useState(false)

  const verb = running ? "Searching" : (TOOL_VERBS[tool] ?? tool)
  // A call that failed is never folded away. Its caveat is one line, and the
  // count that would replace it — "1 result" over a search that found nothing —
  // would say the opposite of what happened.
  const expandable = !failed && items.length > 0
  const shown = failed || open

  return (
    <div className={cn("relative", className)}>
      <Tick running={running} />

      <button
        type="button"
        onClick={() => expandable && setOpen((v) => !v)}
        aria-expanded={expandable ? open : undefined}
        disabled={!expandable}
        className={cn(
          "flex items-baseline gap-[0.6em] text-left",
          expandable && "group cursor-pointer",
          !expandable && "cursor-default"
        )}
      >
        <span
          className={cn(
            "tag shrink-0 whitespace-nowrap",
            "text-muted-foreground/60",
            // The same shimmer everything mid-flight in the app carries.
            running && "text-shimmer"
          )}
        >
          {running ? `${verb}…` : verb}
        </span>

        {/* The query is the half worth reading, so it gets the body face and the
            brighter ink while the label stays a tag. */}
        <span
          className={cn(
            "text-[0.92em] leading-[1.7] text-muted-foreground transition-colors",
            expandable && "group-hover:text-foreground"
          )}
        >
          {argument}
        </span>

        {expandable && (
          <span className="shrink-0 whitespace-nowrap text-muted-foreground/50">
            <span aria-hidden className="text-muted-foreground/35">
              ·
            </span>{" "}
            <span className="text-[0.72em] tabular-nums">
              {items.length === 1 ? "1 result" : `${items.length} results`}
            </span>
          </span>
        )}
      </button>

      {shown && (
        // Flush with the working's prose, not indented under the query. The
        // trace has one left edge and everything in it lines up on that; a
        // second column for results turned the rail into a two-level outline,
        // which is more structure than four lines of supporting text needs.
        <ul className="mt-[0.5em] space-y-[0.35em]">
          {items.map((item, i) => (
            <li
              key={i}
              className={cn(
                "text-[0.85em] leading-[1.6] text-muted-foreground/75",
                // A caveat is a state rather than a finding, so it is set apart
                // from the results a working search would have put here.
                failed && "italic"
              )}
            >
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * The mark where this step meets the trace's rail.
 *
 * Pulled left by the trace's own padding so it starts *on* the hairline. It is
 * the one thing in the trace allowed a colour, and only while the call is in
 * flight — `--throbber-spark` is the token the status row's throbber is using at
 * the same moment, so the two read as one signal rather than two.
 */
function Tick({ running }: { running: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "absolute top-[0.72em] -left-[1.15em] h-px w-[0.7em] transition-colors duration-500",
        running ? "bg-[var(--throbber-spark)]" : "bg-border"
      )}
    />
  )
}

/**
 * What the tool came back with, read out of the marker vocabulary it rendered.
 *
 * A search that worked is a `<|list|>` of items; one that did not — no matches,
 * no API key, a refusal from the provider — is a `<|warn|>` holding the reason.
 * Anything else is flattened to its text rather than dropped: a body this does
 * not recognise is still something the model was told, and hiding it would make
 * the trace a lie about what the reply had to work with.
 */
function resultsOf(body: string | null): { items: string[]; failed: boolean } {
  if (!body?.trim()) return { items: [], failed: false }

  const items: string[] = []
  let failed = false
  const walk = (node: MNode) => {
    if (node.type === "item" || node.type === "warn") {
      const text = textContent(node).trim()
      if (text) items.push(text)
      if (node.type === "warn") failed = true
      return
    }
    if ("children" in node) (node as MElement).children.forEach(walk)
  }
  walk(parseMessage(body))

  if (items.length > 0) return { items, failed }
  const flat = textContent(parseMessage(body)).trim()
  return { items: flat ? [flat] : [], failed: false }
}
