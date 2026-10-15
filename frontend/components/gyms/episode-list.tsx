"use client"

import { Check, Circle, Dot, X } from "lucide-react"

import { cn } from "@/lib/utils"
import { RunningDot } from "@/components/ui/running-dot"
import type { GymEpisode } from "@/lib/gyms"

/**
 * The run's tasks, one row each, as a way of choosing which to read.
 *
 * A scoreboard rather than a list of links: the number this page exists to
 * produce is "how many of these did the model get", and that is only legible
 * when the pass and the fail are the loudest marks in the column.
 */
export function EpisodeList({
  episodes,
  selected,
  live,
  onSelect,
}: {
  episodes: GymEpisode[]
  selected: string | null
  /** The episode currently being generated, if any. */
  live: string | null
  onSelect: (episodeId: string) => void
}) {
  return (
    <ol className="space-y-px">
      {episodes.map((episode) => {
        const isLive = episode.id === live
        return (
          <li key={episode.id}>
            <button
              type="button"
              onClick={() => onSelect(episode.id)}
              aria-current={episode.id === selected}
              className={cn(
                "flex w-full items-center gap-3 rounded-(--radius-sheet) px-3 py-2 text-left text-sm transition-colors",
                episode.id === selected ? "bg-muted" : "hover:bg-muted/50"
              )}
            >
              <Mark episode={episode} live={isLive} />
              <span className="min-w-0 flex-1 truncate">
                <span className="text-muted-foreground">
                  {episode.index + 1}.
                </span>{" "}
                {episode.task_type}
              </span>
              <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
                {episode.seed}
              </span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

/** Displays the episode’s live, graded or abandoned status without changing its selection behavior. */
function Mark({ episode, live }: { episode: GymEpisode; live: boolean }) {
  if (live) return <RunningDot label="running" />
  if (episode.status === "graded") {
    return (episode.score ?? 0) >= 1 ? (
      <Check aria-label="correct" className="size-4 shrink-0 text-foreground" />
    ) : (
      <X aria-label="wrong" className="size-4 shrink-0 text-destructive" />
    )
  }
  if (episode.status === "closed") {
    return (
      <Dot
        aria-label="abandoned"
        className="size-4 shrink-0 text-muted-foreground"
      />
    )
  }
  return (
    <Circle
      aria-label="not run"
      className="size-3 shrink-0 text-muted-foreground/50"
    />
  )
}
