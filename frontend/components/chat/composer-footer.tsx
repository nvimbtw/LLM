"use client"

import * as React from "react"
import { Pencil } from "lucide-react"

import { cn } from "@/lib/utils"
import { Skeleton } from "@/components/ui/skeleton"
import { VoiceWaveform } from "./voice-waveform"

export interface ComposerFooterProps {
  isLoading?: boolean
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
  /** The selected model's trained context window, or null while models are
   *  loading. This is always the *trained* figure — never pre-multiplied —
   *  so the tick mark and the overage badge below have a fixed line to
   *  measure from regardless of `contextMultiplier`. */
  contextWindow: number | null
  /** Stretches the real ceiling the bar fills toward, via the LLM engine's
   *  NTK-scaled RoPE tables (`docs/llm-engine/README.md`). 1 (default) renders
   *  identically to before this existed — no tick, no badge. */
  contextMultiplier?: number
  /** Tokens `/prune` has excluded from what the model is prompted with —
   *  `prunedTokens` from `useChat` (`lib/context-floor.ts`). Already backed
   *  out of `contextTokensUsed` by the caller, so this is only what turns on
   *  the "% unseen" badge below — shown alongside the overage one, not
   *  instead of it, since a chat can be both post-prune and past the trained
   *  window again. 0 (default, and the case before any prune) renders exactly
   *  as if this prop didn't exist. */
  unseenTokens?: number
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

// "×4", but "×2.5" for a half-step multiplier — settings.ts allows 0.5 steps,
// and a trailing ".0" on every whole one would be noise the common case pays
// for the rare one.
function formatMultiplier(n: number) {
  return `×${Number.isInteger(n) ? n : n.toFixed(1)}`
}

// Calm well under the *real* ceiling, amber approaching it, red once the next
// turn is likely to start evicting history. Stops rather than a gradient,
// using only tokens already in the palette (chart-1 doubles as the app's one
// amber) — three legible states instead of a scale nobody can read at a
// glance. The real ceiling is the multiplied window when one is in effect, not
// the trained one — crossing into extended territory is what this feature is
// for, not itself a danger sign.
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
function describeUsage(
  used: number,
  draft: number,
  limit: number,
  trained: number,
  multiplier: number,
  unseen: number
): string {
  const count = (n: number) => n.toLocaleString("en-US")
  const base =
    draft === 0
      ? `~${count(used)} of ${count(limit)} tokens used (estimated)`
      : `~${count(used + draft)} of ${count(limit)} tokens — this draft is ` +
        `${count(draft)}, counted under the model's vocabulary; the ~${count(used)} ` +
        `before it is estimated`

  // Independent facts, stated independently — both can be true at once: a
  // prune brings usage back under the trained window the moment it runs, but
  // the conversation can grow past that window again afterwards, at which
  // point some of this chat is unseen *and* what's left has again gone past
  // the trained size.
  const clauses: string[] = []

  if (unseen > 0) {
    const everSeen = used + draft + unseen
    const unseenPercent =
      everSeen > 0 ? Math.round((unseen / everSeen) * 100) : 0
    clauses.push(
      `${count(unseen)} earlier tokens (${unseenPercent}%) were pruned from what the model is prompted with`
    )
  }

  if (multiplier > 1 && used + draft > trained) {
    // Percent *past the trained window*, not percent of the extended one — the
    // number this feature exists to show is how far past what the model
    // actually trained on you are, which the bar's own fill (scaled to the
    // real ceiling) doesn't say on its own.
    const overPercent = Math.round(((used + draft) / trained - 1) * 100)
    clauses.push(
      `${formatMultiplier(multiplier)} extended from ${count(trained)} — ` +
        `${overPercent}% past the model's trained window`
    )
  }

  return clauses.length ? `${base} (${clauses.join("; ")})` : base
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
  isLoading = false,
  title,
  onRenameTitle,
  contextTokensUsed,
  draftTokens = 0,
  contextWindow,
  contextMultiplier = 1,
  unseenTokens = 0,
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
  // The bar's real ceiling. Multiplier at 1 makes this exactly `contextWindow`,
  // so nothing below differs from before extension existed.
  const effectiveLimit =
    contextWindow && contextWindow > 0
      ? contextWindow * contextMultiplier
      : null
  const percentUsed =
    effectiveLimit !== null ? Math.min(1, totalTokens / effectiveLimit) : null
  // The draft's own share of the bar, so the tip that moves as you type is the
  // part you are adding. Taken out of the total rather than measured against
  // the window on its own, so the two segments cannot sum past a full bar once
  // the draft alone overruns it.
  const percentDraft =
    percentUsed !== null && totalTokens > 0
      ? percentUsed * (draftTokens / totalTokens)
      : 0
  const tone = percentUsed !== null ? meterTone(percentUsed) : null
  // Where the trained window ends, as a fraction of the bar's own width —
  // hidden at 1 (equal to a full bar) rather than drawn on top of the end cap,
  // where a tick would just look like the bar's border.
  const trainedTick =
    contextMultiplier > 1 && contextWindow && effectiveLimit
      ? contextWindow / effectiveLimit
      : null
  const overBase =
    contextMultiplier > 1 && contextWindow && totalTokens > contextWindow
      ? Math.round((totalTokens / contextWindow - 1) * 100)
      : null
  // What fraction of everything this chat has ever said is now excluded from
  // what the model is prompted with — the fact `/prune` exists to produce.
  // Shown alongside `overBase`, not instead of it: pruning brings usage back
  // under the trained window at the moment it runs, but the conversation can
  // grow past that window again afterwards, and at that point both facts are
  // true together.
  const everSeen = totalTokens + unseenTokens
  const unseenPercent =
    unseenTokens > 0 && everSeen > 0
      ? Math.round((unseenTokens / everSeen) * 100)
      : null

  return (
    // Three columns rather than a `justify-between` row, so the middle one is
    // centred on the band itself and not on whatever space the title and the
    // meter happen to leave over. The equal `1fr` sides are what make that
    // true, and they read exactly like the old two-ended row when the middle
    // is empty — which is most of the time.
    <div
      className={cn(
        "grid min-h-5 grid-cols-[1fr_auto_1fr] items-center gap-3 px-3.5 text-xs text-muted-foreground/80",
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

      {isLoading ? (
        <span
          className="flex items-center gap-1.5 justify-self-end"
          role="status"
        >
          <span className="sr-only">Loading context window</span>
          <Skeleton
            aria-hidden="true"
            className="h-1 w-8 motion-reduce:animate-none"
          />
          <Skeleton
            aria-hidden="true"
            className="h-3 w-16 motion-reduce:animate-none"
          />
        </span>
      ) : (
        percentUsed !== null &&
        tone && (
          <span
            className={cn(
              "flex shrink-0 items-center gap-1.5 justify-self-end tabular-nums",
              tone.text
            )}
            title={describeUsage(
              contextTokensUsed,
              draftTokens,
              effectiveLimit!,
              contextWindow!,
              contextMultiplier,
              unseenTokens
            )}
          >
            <span className="relative flex h-1 w-8 overflow-hidden rounded-full bg-muted-foreground/20">
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
              {/* Where the trained window ends. A line rather than a third fill
                colour: it marks a boundary in the same bar, not a different
                kind of token — the fill past it is still exactly what the fill
                before it is. */}
              {trainedTick !== null && (
                <span
                  className="absolute inset-y-0 w-px bg-background/70"
                  style={{ left: `${trainedTick * 100}%` }}
                />
              )}
            </span>
            {formatTokens(totalTokens)}/{formatTokens(effectiveLimit!)}
            {/* Independent facts, shown independently: a prune brings usage back
              under the trained window at the moment it runs, but the
              conversation can grow past it again afterwards — at which point
              both are true together (some of this chat is unseen, *and*
              what's left has again gone past the trained size) and neither
              badge should hide the other. */}
            {overBase !== null && (
              <span
                className={cn(
                  "rounded-full px-1 py-px",
                  tone.text,
                  "bg-current/10"
                )}
              >
                +{overBase}% {formatMultiplier(contextMultiplier)}
              </span>
            )}
            {unseenPercent !== null && (
              <span
                className={cn(
                  "rounded-full px-1 py-px",
                  tone.text,
                  "bg-current/10"
                )}
              >
                {unseenPercent}% unseen
              </span>
            )}
          </span>
        )
      )}
    </div>
  )
}
