"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import {
  parseMessage,
  textContent,
  type MElement,
  type MNode,
} from "@/lib/message-parser"
import { toolWords } from "@/lib/tools"

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
 *
 * The results are set two different ways, because they are two different kinds
 * of text. A search comes back with *findings*, which take a hanging index in
 * the tag face — the apparatus a footnote is set with, and the cheapest way to
 * say "these are numbered things somebody found". A shell comes back with
 * *output*, which is set in the mono face and carries no index: `01` over the
 * lines of `wc -l` would be a claim about what those lines are. They used to
 * share one class, so terminal output rendered in the body face.
 */
export interface TraceStepProps {
  /** The tool's wire name — `web-search`, `bash`. */
  tool: string
  argument: string
  /** The result in the marker vocabulary, or null while the call is running. */
  body: string | null
  running: boolean
  className?: string
}

/**
 * What the disclosure counts. A search comes back as a list of findings worth
 * counting; a shell comes back as output, where "3 results" would be a lie
 * about what the lines are.
 */
function summaryOf(tool: string, count: number): string {
  if (tool !== "web-search") return "output"
  return count === 1 ? "1 result" : `${count} results`
}

export function TraceStep({
  tool,
  argument,
  body,
  running,
  className,
}: TraceStepProps) {
  const { items, failed, shell } = React.useMemo(() => resultsOf(body), [body])
  const [open, setOpen] = React.useState(false)

  // `lib/tools.ts` owns how a tool is said, so the tick and the thread's status
  // row cannot end up calling the same call two different things.
  const words = toolWords(tool)
  const verb = running ? words.running : words.done
  // A call that failed is never folded away. Its caveat is one line, and the
  // count that would replace it — "1 result" over a search that found nothing —
  // would say the opposite of what happened.
  const expandable = !failed && items.length > 0
  const shown = failed || open

  return (
    <div
      // Both attributes are read by the rules in globals.css: `running` warms
      // and reaches the tick, `open` drives the results' own fold. They are on
      // this element rather than the step wrapper because the wrapper belongs to
      // `MarginNote`, which knows nothing about tools.
      data-trace-running={running ? "" : undefined}
      data-trace-open={shown ? "" : undefined}
      className={className}
    >
      <Tick />

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
            "shrink-0 tag whitespace-nowrap",
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

        {/* Fades in as the answer lands, rather than appearing with it: the
            count is the one part of the row that was not there a moment ago. */}
        <span
          className={cn(
            "shrink-0 tag whitespace-nowrap text-muted-foreground/50 tabular-nums",
            "transition-opacity duration-500",
            !expandable && "opacity-0"
          )}
        >
          {expandable ? summaryOf(tool, items.length) : ""}
        </span>
      </button>

      {/* Flush with the working's prose, not indented under the query. The trace
          has one left edge and everything in it lines up on that; a second
          column for results turned the rail into a two-level outline, which is
          more structure than four lines of supporting text needs. */}
      <div className="trace-fold">
        <div className="overflow-hidden">
          <ul className="mt-[0.5em] space-y-[0.35em]">
            {items.map((item, i) => (
              <li
                key={i}
                className={cn(
                  "leading-[1.6] text-muted-foreground/75",
                  // A caveat is a state rather than a finding, so it is set
                  // apart from the results a working search would have put here.
                  failed && "text-[0.85em] italic",
                  shell && "font-mono text-[0.78em]",
                  !failed &&
                    !shell &&
                    "grid grid-cols-[2.1em_1fr] text-[0.85em]"
                )}
              >
                {!failed && !shell && (
                  <span aria-hidden className="tag text-muted-foreground/35">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                )}
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

/**
 * The mark where this step meets the trace's rail.
 *
 * Pulled left by the trace's own padding so it starts *on* the hairline. Two
 * nested elements, because it carries two independent movements — see "The
 * trace, in motion" in `globals.css`. It is the one thing in the trace allowed
 * a colour, and only while the call is in flight: `--throbber-spark` is the
 * token the status row's throbber is using at the same moment, so the two read
 * as one signal rather than two.
 */
function Tick() {
  return (
    <span
      aria-hidden
      className="trace-tick absolute top-[0.72em] -left-[1.15em] h-px w-[0.7em]"
    >
      <span />
    </span>
  )
}

/**
 * What the tool came back with, read out of the marker vocabulary it rendered.
 *
 * A search that worked is a `<|list|>` of items; a shell command is a
 * `<|codeblock|>`, one line per row so the output reads like the terminal it
 * came from; one that did not work — no matches, no sandbox, a refusal from the
 * provider — is a `<|warn|>` holding the reason.
 * Anything else is flattened to its text rather than dropped: a body this does
 * not recognise is still something the model was told, and hiding it would make
 * the trace a lie about what the reply had to work with.
 */
function resultsOf(body: string | null): {
  items: string[]
  failed: boolean
  shell: boolean
} {
  if (!body?.trim()) return { items: [], failed: false, shell: false }

  const items: string[] = []
  let failed = false
  let shell = false
  const walk = (node: MNode) => {
    if (node.type === "code block") {
      shell = true
      for (const line of textContent(node).split("\n")) {
        if (line.trim()) items.push(line)
      }
      return
    }
    if (node.type === "item" || node.type === "warn") {
      const text = textContent(node).trim()
      if (text) items.push(text)
      if (node.type === "warn") failed = true
      return
    }
    if ("children" in node) (node as MElement).children.forEach(walk)
  }
  walk(parseMessage(body))

  if (items.length > 0) return { items, failed, shell }
  const flat = textContent(parseMessage(body)).trim()
  return { items: flat ? [flat] : [], failed: false, shell: false }
}
