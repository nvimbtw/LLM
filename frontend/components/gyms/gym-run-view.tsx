"use client"

import * as React from "react"
import { Play, Square } from "lucide-react"

import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { EpisodeList } from "@/components/gyms/episode-list"
import { CommandBox } from "@/components/gyms/command-box"
import { EpisodeTranscript } from "@/components/gyms/episode-transcript"
import { SourceScores } from "@/components/gyms/source-scores"
import { RewardChart } from "@/components/gyms/reward-chart"
import { useGymRun } from "@/hooks/use-gym-run"
import { useModelNames } from "@/hooks/use-model-names"
import { modelLabel } from "@/lib/gyms"
import { useSettings } from "@/components/settings/settings-provider"

/**
 * One run, watched.
 *
 * The scoreboard on the left is what the run is *for*; the transcript on the
 * right is why it came out that way. They are side by side rather than one
 * under the other because the question a reader arrives with — "it scored 2 of
 * 5, what went wrong in the other three" — is answered by moving between them.
 *
 * Pressing Start opens the socket that drives the run. Nothing else does: a run
 * is created with its sandboxes seeded and then sits there, so opening this
 * page never spends the GPU by itself.
 */
export interface GymRunViewProps {
  runId: string
}

/** Opens one evaluation run, preserving its scoreboard, transcript and explicit Start/Stop lifetime. */
export function GymRunView({ runId: id }: GymRunViewProps) {
  const { settings } = useSettings()
  const { names } = useModelNames()
  const { run, transcripts, live, isRunning, error, start, stop, open } =
    useGymRun(id)
  const [chosen, setChosen] = React.useState<string | null>(null)

  const selectedId = chosen ?? live ?? run?.episodes[0]?.id ?? null
  const selected = run?.episodes.find((e) => e.id === selectedId) ?? null

  React.useEffect(() => {
    if (selectedId) open(selectedId)
  }, [selectedId, open])

  if (!run) {
    return (
      <PageShell>
        {error ? <LoadError hint={error} /> : <Skeleton className="h-96" />}
      </PageShell>
    )
  }

  const rewards = (selected?.steps ?? []).map((step, i) => ({
    step: step.step_num + 1,
    reward: step.reward ?? 0,
    total: (selected?.steps ?? [])
      .slice(0, i + 1)
      .reduce((sum, s) => sum + (s.reward ?? 0), 0),
  }))

  return (
    <PageShell>
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        title={modelLabel(names, run.model_name)}
        description={[
          run.task_types.join(" · "),
          run.group_size > 1
            ? `${run.episode_count / run.group_size} tasks × ${run.group_size} rollouts`
            : `${run.episode_count} tasks`,
          `seeds from ${run.seed_base}`,
        ].join(" · ")}
        back={{ href: `/gyms/sources/${run.gym_id}`, label: "Runs" }}
        action={
          isRunning ? (
            <Button variant="outline" onClick={stop}>
              <Square className="size-4" />
              Stop
            </Button>
          ) : (
            <Button
              onClick={() => start(sampling(settings))}
              disabled={run.graded >= run.episode_count}
            >
              <Play className="size-4" />
              {run.graded > 0 ? "Resume" : "Start"}
            </Button>
          )
        }
      />

      {error && <FormError className="mb-6">{error}</FormError>}

      <div className="grid gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <div className="space-y-6">
          <Panel
            variant="open"
            title="Tasks"
            description={score(run.score, run.correct, run.graded)}
            bodyClassName="px-1"
          >
            <EpisodeList
              episodes={run.episodes}
              selected={selectedId}
              live={live}
              onSelect={setChosen}
            />
          </Panel>
          {run.gym_id === "general" && <SourceScores episodes={run.episodes} />}
          {rewards.length > 0 && <RewardChart data={rewards} />}
        </div>

        <Panel
          variant="open"
          title={selected ? `Task ${selected.index + 1}` : "Task"}
          description={selected?.task_type}
        >
          {selected ? (
            <div className="space-y-8">
              <EpisodeTranscript
                episode={selected}
                transcript={transcripts[selected.id]}
              />
              {selected.status !== "graded" &&
                selected.status !== "closed" &&
                selected.tools?.includes("bash") &&
                !isRunning && (
                  <div className="border-t border-border pt-6">
                    <p className="mb-2 eyebrow text-muted-foreground">
                      Run a command yourself
                    </p>
                    <CommandBox episodeId={selected.id} />
                  </div>
                )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              This run has no tasks.
            </p>
          )}
        </Panel>
      </div>
    </PageShell>
  )
}

/** The headline, or what stands in for it before anything is graded. */
function score(mean: number | null, correct: number, graded: number): string {
  if (mean === null) return "Nothing graded yet."
  return `${correct} of ${graded} correct — ${Math.round(mean * 100)}%`
}

/**
 * The decoding knobs, taken from the same settings a chat turn uses.
 *
 * Deliberately not a second set of controls on this page: a gym is meant to
 * measure the model as it is served, and a run sampled differently from every
 * conversation would measure something nobody talks to.
 */
function sampling(settings: {
  temperature: number
  topK: number
  topP: number
  contextMultiplier: number
}) {
  return {
    temperature: settings.temperature,
    topK: settings.topK,
    topP: settings.topP,
    contextMultiplier: settings.contextMultiplier,
  }
}
