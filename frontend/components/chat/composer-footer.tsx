"use client"

import * as React from "react"
import { Pencil } from "lucide-react"

import { cn } from "@/lib/utils"

export interface ComposerFooterProps {
  title: string
  /** Omitted where there's no chat yet to rename (the new-chat screen) — the
   *  title renders as plain text rather than an edit control. */
  onRenameTitle?: (title: string) => void
  /** Tokens used so far in this conversation — see chat-thread.tsx for how
   *  it's derived; it's an estimate, not a server-verified count. */
  contextTokensUsed: number
  /** The selected model's context window, or null while models are loading. */
  contextWindow: number | null
  className?: string
}

function formatTokens(n: number) {
  if (n < 1000) return `${n}`
  const k = n / 1000
  return `${k.toFixed(k < 10 ? 1 : 0)}k`
}

// Calm well under the window, amber approaching it, red once the next turn is
// likely to start evicting history. Stops rather than a gradient, using only
// tokens already in the palette (chart-1 doubles as the app's one amber) —
// three legible states instead of a scale nobody can read at a glance.
function meterTone(percentUsed: number) {
  if (percentUsed >= 0.9) {
    return { bar: "bg-destructive", text: "text-destructive" }
  }
  if (percentUsed >= 0.7) {
    return { bar: "bg-chart-1", text: "text-chart-1" }
  }
  return { bar: "bg-muted-foreground/60", text: "text-muted-foreground" }
}

/**
 * The composer's bottom band, inside its border and under a hairline: which
 * conversation this is, and how much of its context window is spent.
 *
 * It reads as the quieter line by type and colour rather than by being a
 * separate object — it was one, floating below the card, and the gap between
 * them was a slot of unblurred thread text between two frosted surfaces.
 */
export function ComposerFooter({
  title,
  onRenameTitle,
  contextTokensUsed,
  contextWindow,
  className,
}: ComposerFooterProps) {
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(title)
  const [prevTitle, setPrevTitle] = React.useState(title)

  // Follow the real title while not editing, so a rename that lands from
  // elsewhere (or the initial load) is reflected without touching a draft
  // someone's mid-edit on. Adjusted during render rather than in an Effect —
  // see slash-command-menu.tsx for the same pattern and why.
  if (title !== prevTitle) {
    setPrevTitle(title)
    if (!editing) setDraft(title)
  }

  const commit = () => {
    setEditing(false)
    const next = draft.trim()
    if (next && next !== title) onRenameTitle?.(next)
    else setDraft(title)
  }

  const percentUsed =
    contextWindow && contextWindow > 0
      ? Math.min(1, contextTokensUsed / contextWindow)
      : null
  const tone = percentUsed !== null ? meterTone(percentUsed) : null

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 px-3.5 text-xs text-muted-foreground/80",
        className
      )}
    >
      {!onRenameTitle ? (
        <span className="min-w-0 truncate font-medium text-foreground/80">
          {title}
        </span>
      ) : editing ? (
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              commit()
            }
            if (e.key === "Escape") {
              e.preventDefault()
              setDraft(title)
              setEditing(false)
            }
          }}
          autoFocus
          className="min-w-0 flex-1 truncate bg-transparent font-medium text-foreground/80 outline-none"
        />
      ) : (
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="group/title flex min-w-0 items-center gap-1.5 text-left"
        >
          <span className="truncate font-medium text-foreground/80">
            {title}
          </span>
          <Pencil className="size-3 shrink-0 opacity-0 transition-opacity group-hover/title:opacity-100" />
        </button>
      )}

      {percentUsed !== null && tone && (
        <span
          className={cn(
            "flex shrink-0 items-center gap-1.5 tabular-nums",
            tone.text
          )}
          title={`~${contextTokensUsed.toLocaleString("en-US")} of ${contextWindow!.toLocaleString("en-US")} tokens used (estimated)`}
        >
          <span className="h-1 w-8 overflow-hidden rounded-full bg-muted-foreground/20">
            <span
              className={cn(
                "block h-full rounded-full transition-colors",
                tone.bar
              )}
              style={{ width: `${Math.round(percentUsed * 100)}%` }}
            />
          </span>
          {formatTokens(contextTokensUsed)}/{formatTokens(contextWindow!)}
        </span>
      )}
    </div>
  )
}
