"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { Skeleton } from "@/components/ui/skeleton"
import { VoiceWaveform } from "./voice-waveform"

export interface ComposerFooterProps {
  isLoading?: boolean
  choices?: React.ReactNode
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

/** Formats a token count compactly for the footer meter. */
function formatTokens(n: number) {
  if (n < 1000) return `${n}`
  const k = n / 1000
  return `${k.toFixed(k < 10 ? 1 : 0)}k`
}

/** Formats the effective context multiplier without redundant decimal zeros. */
function formatMultiplier(n: number) {
  return `×${Number.isInteger(n) ? n : n.toFixed(1)}`
}

/** Selects warning colours against the effective context ceiling. */
function meterTone(percentUsed: number) {
  if (percentUsed >= 0.9) {
    return { bar: "bg-destructive", text: "text-destructive" }
  }
  if (percentUsed >= 0.7) {
    return { bar: "bg-chart-1", text: "text-chart-1" }
  }
  return { bar: "bg-muted-foreground/60", text: "text-muted-foreground" }
}

/** Describes exact draft counts, estimated history, extension and pruning independently. */
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
    const overPercent = Math.round(((used + draft) / trained - 1) * 100)
    clauses.push(
      `${formatMultiplier(multiplier)} extended from ${count(trained)} — ` +
        `${overPercent}% past the model's trained window`
    )
  }

  return clauses.length ? `${base} (${clauses.join("; ")})` : base
}

/** Renders footer choices, dictation waveform and independent draft/history context segments with pruning and extension badges. */
export function ComposerFooter({
  isLoading = false,
  choices,
  contextTokensUsed,
  draftTokens = 0,
  contextWindow,
  contextMultiplier = 1,
  unseenTokens = 0,
  listening = false,
  onDeviceDictation = false,
  className,
}: ComposerFooterProps) {
  const totalTokens = contextTokensUsed + draftTokens
  const effectiveLimit =
    contextWindow && contextWindow > 0
      ? contextWindow * contextMultiplier
      : null
  const percentUsed =
    effectiveLimit !== null ? Math.min(1, totalTokens / effectiveLimit) : null
  const percentDraft =
    percentUsed !== null && totalTokens > 0
      ? percentUsed * (draftTokens / totalTokens)
      : 0
  const tone = percentUsed !== null ? meterTone(percentUsed) : null
  const trainedTick =
    contextMultiplier > 1 && contextWindow && effectiveLimit
      ? contextWindow / effectiveLimit
      : null
  const overBase =
    contextMultiplier > 1 && contextWindow && totalTokens > contextWindow
      ? Math.round((totalTokens / contextWindow - 1) * 100)
      : null
  const everSeen = totalTokens + unseenTokens
  const unseenPercent =
    unseenTokens > 0 && everSeen > 0
      ? Math.round((unseenTokens / everSeen) * 100)
      : null

  return (
    <div
      className={cn(
        "grid min-h-5 grid-cols-[1fr_auto_1fr] items-center gap-3 px-3.5 text-xs text-muted-foreground/80",
        className
      )}
    >
      <div className="min-w-0">{choices}</div>

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

              <span
                className={cn("h-full opacity-45 transition-colors", tone.bar)}
                style={{ width: `${percentDraft * 100}%` }}
              />

              {trainedTick !== null && (
                <span
                  className="absolute inset-y-0 w-px bg-background/70"
                  style={{ left: `${trainedTick * 100}%` }}
                />
              )}
            </span>
            {formatTokens(totalTokens)}/{formatTokens(effectiveLimit!)}
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
