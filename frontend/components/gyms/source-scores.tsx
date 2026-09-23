import { Panel } from "@/components/ui/panel"
import { type GymEpisode, type GymSourceScore } from "@/lib/gyms"

type SourceScoresProps =
  | { episodes: GymEpisode[]; scores?: never }
  | { scores: Record<string, GymSourceScore>; episodes?: never }

export function SourceScores({ episodes, scores }: SourceScoresProps) {
  const sources = scores ? Object.keys(scores) : [...new Set(episodes.map((episode) => episode.gym_id))]
  return (
    <Panel title="By gym">
      <dl className="space-y-3 text-sm">
        {sources.map((source) => {
          const graded = (episodes ?? []).filter((episode) => episode.gym_id === source && episode.score !== null)
          const measured = scores?.[source]
          const mean = measured ? measured.score : graded.length ? graded.reduce((sum, episode) => sum + (episode.score ?? 0), 0) / graded.length : null
          return (
            <div key={source} className="flex justify-between gap-3">
              <dt>{source}</dt>
              <dd className="text-muted-foreground">
                {mean === null ? "Nothing graded" : `${Math.round(mean * 100)}% · ${measured?.graded ?? graded.length} graded`}
              </dd>
            </div>
          )
        })}
      </dl>
    </Panel>
  )
}
