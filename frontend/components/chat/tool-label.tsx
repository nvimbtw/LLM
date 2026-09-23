"use client"

import type { ToolCall } from "@/lib/message-parser"
import { toolArgument, toolWords } from "@/lib/tools"

/**
 * What the thread's status row says while a tool runs: the tool, then what it
 * was handed — `Search: tallest mountain in europe`, `Bash: ls -l /etc`.
 *
 * The row it sits in already had a word for this phase, and the word was
 * "Searching…", from when search was the only tool there was. A shell now runs
 * through the same seam, so the label reads the live call instead
 * (`Trace::callInFlight`) and names it out of the one table that knows how a
 * tool is said (`lib/tools.ts`).
 *
 * Unlike `ThinkingLabel` this is not a guess. The model told us what it wanted
 * and the backend is doing exactly that. Plan operations get a short caption
 * from `lib/tools.ts`; other arguments stay verbatim except for whitespace,
 * because a heredoc arrives with newlines and a status row is one line.
 */
export function ToolLabel({ call }: { call: ToolCall }) {
  const words = toolWords(call.tool)
  const argument = oneLine(toolArgument(call.tool, call.argument))

  // A call with nothing in it is a call the model mangled — the LLM engine drops
  // those, so this is rare, but a bare "Search:" with nothing after the colon
  // would look like a rendering fault rather than the model's own mistake.
  if (!argument) {
    return <span className="text-shimmer font-medium">{words.running}…</span>
  }

  return (
    // min-w-0 and no basis of its own: the spacer beside it has `flex-1` with a
    // zero basis, so it takes the slack and this takes all the shrink — which
    // is what makes a 400-character command truncate here rather than push the
    // token count off the row.
    <span className="flex min-w-0 items-baseline gap-1.5">
      <span className="text-shimmer shrink-0 font-medium">{words.noun}:</span>
      {/* Truncated, not wrapped: this row is a fixed-height slot so the
          thinking → tool → generating swaps don't reflow the thread. */}
      <span className="truncate text-muted-foreground">{argument}</span>
    </span>
  )
}

function oneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim()
}
