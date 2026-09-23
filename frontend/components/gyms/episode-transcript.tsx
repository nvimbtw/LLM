"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { parseUserMarkdown } from "@/lib/user-parser"
import { RichContent, RichMessage } from "@/components/chat/rich-message"
import { Trace } from "@/components/chat/trace"
import type { Transcript } from "@/hooks/use-gym-run"
import type { GymEpisode } from "@/lib/gyms"

/**
 * One episode, read the way a chat turn is read.
 *
 * Deliberately the chat components and not a view of its own. What a gym shows
 * has to be what the model actually produced — the same markup, the same
 * reasoning panel, the same command output inside the same code block — because
 * the whole point of the gym is that the model is doing in it exactly what it
 * does in a conversation. A bespoke renderer here would be a second opinion
 * about that, and the first time the two disagreed the gym would be the one
 * lying.
 *
 * So: the task is a user bubble, the rounds that called the shell are the
 * `Trace`, and the round that stopped calling is the reply. The verdict is the
 * only thing chat has no equivalent for, and it sits under the answer rather
 * than beside it — it is what the answer turned out to be worth.
 */
export function EpisodeTranscript({
  episode,
  transcript,
  className,
}: {
  episode: GymEpisode
  transcript?: Transcript
  className?: string
}) {
  const rounds = transcript?.rounds ?? []
  // The last round is the answer and everything before it is the working — the
  // same fold `groupTurns` applies to a stored chat turn.
  const working = rounds.slice(0, -1).join("")
  const reply = rounds.at(-1) ?? ""

  return (
    <div className={cn("space-y-6", className)}>
      <div className="flex w-full flex-col items-end">
        <div className="max-w-[min(100%,60ch)] rounded-(--radius-bubble) bg-muted px-4 py-3 text-sm leading-relaxed">
          <RichContent root={parseUserMarkdown(episode.prompt)} />
        </div>
      </div>

      {rounds.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {episode.status === "pending"
            ? "Not run yet."
            : "Nothing was recorded for this episode."}
        </p>
      ) : (
        <div>
          <Trace
            working={working}
            answers={transcript?.answers ?? []}
            elapsedSecs={transcript?.elapsedSecs}
            streaming={transcript?.streaming}
          />
          <RichMessage
            content={reply}
            streaming={transcript?.streaming}
            className="max-w-[68ch] text-base leading-relaxed"
          />
        </div>
      )}

      {episode.status === "graded" && <Verdict episode={episode} />}
    </div>
  )
}

/**
 * What the answer was worth, and — only now — what it should have been.
 *
 * The ground truth arrives with the grade and not before: the gyms engine
 * withholds it while an episode is open, so there is no version of this page
 * that could leak it into a run somebody is watching.
 */
function Verdict({ episode }: { episode: GymEpisode }) {
  const correct = (episode.score ?? 0) >= 1
  return (
    <div
      className={cn(
        "rounded-(--radius-sheet) border px-4 py-3 text-sm",
        correct ? "border-border bg-muted/40" : "border-destructive/40 bg-destructive/5"
      )}
    >
      <div className="flex items-baseline justify-between gap-4">
        <p className="eyebrow text-muted-foreground">
          {correct ? "Correct" : "Wrong"}
        </p>
        <p className="text-xs text-muted-foreground">
          return {episode.reward_total.toFixed(2)}
        </p>
      </div>
      <dl className="mt-2 grid gap-1 text-sm sm:grid-cols-[7rem_1fr]">
        <dt className="text-muted-foreground">Answered</dt>
        <dd className="font-mono break-all">{episode.answer || "—"}</dd>
        {!correct && (
          <>
            <dt className="text-muted-foreground">Expected</dt>
            <dd className="font-mono break-all">{episode.ground_truth || "—"}</dd>
          </>
        )}
      </dl>
    </div>
  )
}
