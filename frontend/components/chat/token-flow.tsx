"use client"

/**
 * The count in the thread's status row, with the direction the tokens are
 * moving in — `↑ 4,210 tokens` while something is being fed to the model,
 * `↓ 676 tokens` while it writes.
 *
 * The arrow is the point. A bare number beside a working throbber is ambiguous
 * in the one way that matters: a reply eight hundred tokens long and a prompt
 * eight hundred tokens long are not the same event, and the row used to show
 * both as "800 tokens". Which way it points is decided by the caller, from the
 * phase the row is already in — see `ChatThread`.
 *
 * The two figures behind it are not equally solid, and neither is
 * server-verified: the outgoing count is tallied token by token as the socket
 * delivers them, while the incoming one is the context estimate the composer
 * bills from. This is a status row, not an invoice.
 */
import { RollingNumber } from "@/components/ui/rolling-number"

export type TokenDirection = "in" | "out"

export interface TokenFlowProps {
  direction: TokenDirection
  count: number
}

const ARROW: Record<TokenDirection, string> = { in: "↑", out: "↓" }

const FLOW: Record<TokenDirection, string> = {
  in: "being sent to the model",
  out: "being written by the model",
}

export function TokenFlow({ direction, count }: TokenFlowProps) {
  return (
    // inline-flex and items-center, because the count is not text: it is a
    // stack of wheels as tall as a line, and only centring it against the words
    // beside it keeps the three of them sitting on one line.
    <span
      className="inline-flex shrink-0 items-center gap-1 text-xs text-muted-foreground"
      // The exact figure on hover, since the wheels round it off past a
      // thousand — and in full for a screen reader below, for the same reason.
      title={`${count.toLocaleString("en-US")} tokens in this conversation, ${FLOW[direction]}`}
    >
      {/* aria-hidden on the glyph, and the direction spelled out for a reader
        that would otherwise announce the arrow as "up" and nothing else. The
        whole figure is announced here as plain text, once, since the wheels
        themselves are twenty digits deep and unreadable aloud. */}
      <span aria-hidden>{ARROW[direction]}</span>
      <span className="sr-only">
        {count.toLocaleString("en-US")} tokens in this conversation,{" "}
        {FLOW[direction]}
      </span>
      <RollingNumber
        compact
        // An incoming figure is handed over whole — a message sent, a tool's
        // answer back — so it lands rather than climbs. Only the outgoing count
        // arrives a few tokens at a time, and only there is a slower wheel
        // reporting something true about the rate.
        swift={direction === "in"}
        value={count}
      />
      <span aria-hidden>tokens</span>
    </span>
  )
}
