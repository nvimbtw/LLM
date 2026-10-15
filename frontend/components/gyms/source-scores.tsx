import { Panel } from "@/components/ui/panel"
import { type GymEpisode, type GymSourceScore } from "@/lib/gyms"

type SourceScoresProps =
  | { episodes: GymEpisode[]; scores?: never }
  | { scores: Record<string, GymSourceScore>; episodes?: never }

/** Displays per-gym measured scores from saved totals or the supplied graded episodes. */
export function SourceScores({ episodes, scores }: SourceScoresProps) {
  const sources = scores
    ? Object.keys(scores)
    : [...new Set(episodes.map((episode) => episode.gym_id))]
  return (
    <Panel variant="open" title="By gym">
      <dl className="divide-y divide-border/60 text-sm">
        {sources.map((source) => {
          const graded = (episodes ?? []).filter(
            (episode) => episode.gym_id === source && episode.score !== null
          )
          const measured = scores?.[source]
          const mean = measured
            ? measured.score
            : graded.length
              ? graded.reduce((sum, episode) => sum + (episode.score ?? 0), 0) /
                graded.length
              : null
          return (
            <div
              key={source}
              className="flex justify-between gap-3 py-3 first:pt-0 last:pb-0"
            >
              <dt>{source}</dt>
              <dd className="text-muted-foreground tabular-nums">
                {mean === null
                  ? "Nothing graded"
                  : `${Math.round(mean * 100)}% · ${measured?.graded ?? graded.length} graded`}
              </dd>
            </div>
          )
        })}
      </dl>
    </Panel>
  )
}
