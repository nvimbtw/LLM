"use client"

import * as React from "react"
import { Pencil } from "lucide-react"

import { cn } from "@/lib/utils"
import { VoiceWaveform } from "./voice-waveform"

export interface ComposerFooterProps {
  title: string
  /** Omitted where there's no chat yet to rename (the new-chat screen) — the
   *  title renders as plain text rather than an edit control. */
  onRenameTitle?: (title: string) => void
  /** Tokens used so far in this conversation — see chat-thread.tsx for how
   *  it's derived; it's an estimate, not a server-verified count. */
  contextTokensUsed: number
  /** What the unsent draft would add, counted under the model's own vocabulary
   *  (`hooks/use-draft-tokens.ts`). Kept separate from `contextTokensUsed`
   *  rather than summed by the caller: the two are known to different
   *  standards, and the bar draws them as two segments so what *this* message
   *  costs is legible while it is still being written. */
  draftTokens?: number
  /** The selected model's context window, or null while models are loading. */
  contextWindow: number | null
  /** Dictation is running, which fills the middle column with the waveform.
   *  Nothing else in the band moves — the title and the meter stay exactly
   *  where they are, because the whole point of putting the bars here is that
   *  they cost the composer no space of their own. */
  listening?: boolean
  /** Passed through to the waveform — see `VoiceWaveformProps`. */
  onDeviceDictation?: boolean
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

// The two halves of the figure are known to different standards and the tooltip
// is where that gets said: the draft is a real count under the model's
// vocabulary, the conversation behind it is the running estimate described in
// chat-thread.tsx. Rounding both into one "(estimated)" hid a number that is
// exact, and claiming both were counted would have been the worse half of that
// trade.
function describeUsage(used: number, draft: number, limit: number): string {
  const count = (n: number) => n.toLocaleString("en-US")
  if (draft === 0) {
    return `~${count(used)} of ${count(limit)} tokens used (estimated)`
  }
  return (
    `~${count(used + draft)} of ${count(limit)} tokens — this draft is ` +
    `${count(draft)}, counted under the model's vocabulary; the ~${count(used)} ` +
    `before it is estimated`
  )
}

/**
 * The composer's bottom band, inside its border and under a hairline: which
 * conversation this is, how much of its context window is spent, and — while
 * someone is speaking — the three bars between them.
 *
 * It reads as the quieter line by type and colour rather than by being a
 * separate object — it was one, floating below the card, and the gap between
 * them was a slot of unblurred thread text between two frosted surfaces.
 */
export function ComposerFooter({
  title,
  onRenameTitle,
  contextTokensUsed,
  draftTokens = 0,
  contextWindow,
  listening = false,
  onDeviceDictation = false,
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

  const totalTokens = contextTokensUsed + draftTokens
  const percentUsed =
    contextWindow && contextWindow > 0
      ? Math.min(1, totalTokens / contextWindow)
      : null
  // The draft's own share of the bar, so the tip that moves as you type is the
  // part you are adding. Taken out of the total rather than measured against
  // the window on its own, so the two segments cannot sum past a full bar once
  // the draft alone overruns it.
  const percentDraft =
    percentUsed !== null && totalTokens > 0
      ? percentUsed * (draftTokens / totalTokens)
      : 0
  const tone = percentUsed !== null ? meterTone(percentUsed) : null

  return (
    // Three columns rather than a `justify-between` row, so the middle one is
    // centred on the band itself and not on whatever space the title and the
    // meter happen to leave over. The equal `1fr` sides are what make that
    // true, and they read exactly like the old two-ended row when the middle
    // is empty — which is most of the time.
    <div
      className={cn(
        "grid grid-cols-[1fr_auto_1fr] items-center gap-3 px-3.5 text-xs text-muted-foreground/80",
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

      {/* Always the second child, empty or not: grid places children in
          order, so a cell that comes and goes would move the meter into the
          middle of the band whenever dictation stopped. */}
      <span className="flex justify-center">
        {listening && <VoiceWaveform onDevice={onDeviceDictation} />}
      </span>

      {percentUsed !== null && tone && (
        <span
          className={cn(
            "flex shrink-0 items-center gap-1.5 justify-self-end tabular-nums",
            tone.text
          )}
          title={describeUsage(contextTokensUsed, draftTokens, contextWindow!)}
        >
          <span className="flex h-1 w-8 overflow-hidden rounded-full bg-muted-foreground/20">
            <span
              className={cn("h-full transition-colors", tone.bar)}
              style={{ width: `${(percentUsed - percentDraft) * 100}%` }}
            />
            {/* The draft, in the same colour half-lit: a second hue would read
                as a second meaning, when this is the same quantity at a
                different age. No transition on the width — it tracks typing,
                and an animated one lags a sentence behind the letters. */}
            <span
              className={cn("h-full opacity-45 transition-colors", tone.bar)}
              style={{ width: `${percentDraft * 100}%` }}
            />
          </span>
          {formatTokens(totalTokens)}/{formatTokens(contextWindow!)}
        </span>
      )}
    </div>
  )
}
