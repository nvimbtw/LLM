"use client"

import Link from "next/link"

import { ConfigRecap } from "@/components/jobs/config-recap"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { benchmarkSections } from "@/lib/benchmark"
import {
  KIND_LABEL,
  benchmarkSuiteOf,
  isRunning,
  jobHref,
  type Job,
} from "@/lib/jobs"

/**
 * A job that hasn't started yet.
 *
 * Waiting is a real state, not the absence of one, and every kind used to render
 * it as an empty version of its finished self — a benchmark said "No score", a
 * generation run said nothing at all. There are three things worth knowing while a
 * job waits, and none of them need the job to have run: what you asked for, how
 * far down the line it is, and what will appear here when it starts. That is this
 * page.
 */
export interface JobQueuedPanelProps {
  job: Job
}

export function JobQueuedPanel({ job }: JobQueuedPanelProps) {
  const { board, paused, reorder } = useJobsContext()

  // Position within its own lane. The device and network runners proceed
  // independently, so a generation run is not waiting behind a training run and
  // counting it would overstate the wait.
  const lane = board.active.filter((j) => j.lane === job.lane)
  const queued = lane.filter((j) => j.status === "queued")
  const index = queued.findIndex((j) => j.id_job === job.id_job)
  const ahead = index < 0 ? 0 : index
  const running = lane.find(isRunning) ?? null

  const next = whatHappens(job)

  // A follow-up has no place in the line to report: it starts when the job it
  // names finishes, however long that takes and wherever that job currently sits.
  // The honest answer to "when" is the other job's name.
  const after =
    job.status === "waiting"
      ? ([...board.active, ...board.history].find(
          (j) => j.id_job === job.depends_on
        ) ?? null)
      : null

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,18rem)_minmax(0,1fr)]">
      <ConfigRecap job={job} />

      <div className="space-y-6">
        {job.status === "waiting" ? (
          <Panel title="Waiting" description="Queued behind another job">
            <p className="text-sm text-muted-foreground">
              {after ? (
                <>
                  Starts when{" "}
                  <Link
                    href={jobHref(after)}
                    className="text-foreground underline-offset-4 hover:underline"
                  >
                    {after.label}
                  </Link>{" "}
                  finishes, training on the model it produces. If that run fails
                  or is cancelled, this one is cancelled with it.
                </>
              ) : (
                <>
                  Starts when the job it follows finishes, using the model that
                  job produces.
                </>
              )}
            </p>
          </Panel>
        ) : (
          <Panel
            title="In line"
            description={job.lane === "device" ? "GPU queue" : "Network queue"}
          >
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="font-display text-4xl leading-none font-semibold tabular-nums">
                {ahead === 0 ? "Next" : ordinal(ahead + 1)}
              </span>
              <span className="text-sm text-muted-foreground">
                {ahead === 0
                  ? running
                    ? "as soon as the current job ends"
                    : "starting shortly"
                  : `${ahead} job${ahead === 1 ? "" : "s"} ahead`}
              </span>
            </div>

            {running && (
              <p className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
                <RunningDot />
                <span>Running now:</span>
                <Link
                  href={jobHref(running)}
                  className="truncate text-foreground underline-offset-4 hover:underline"
                >
                  {running.label}
                </Link>
              </p>
            )}

            {/* Paused is the one state where "next" means nothing, so it is said
              plainly rather than left to be inferred from a queue that never
              advances. */}
            {paused && (
              <p className="mt-4 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
                The queue is paused. Nothing new will start until it is resumed
                — the running job carries on.
              </p>
            )}

            {ahead > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="mt-5"
                onClick={() => reorder(job.id_job, null)}
              >
                Run this next
              </Button>
            )}
          </Panel>
        )}

        <Panel title={next.title} description={KIND_LABEL[job.kind]}>
          <ol className="space-y-3">
            {next.steps.map((step, i) => (
              <li key={step.name} className="flex gap-3">
                <span className="w-5 shrink-0 pt-0.5 text-xs text-muted-foreground tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="min-w-0">
                  <span className="block text-sm">{step.name}</span>
                  <span className="mt-0.5 block text-xs text-muted-foreground">
                    {step.blurb}
                  </span>
                </span>
              </li>
            ))}
          </ol>
        </Panel>
      </div>
    </div>
  )
}

interface Step {
  name: string
  blurb: string
}

/**
 * What the run will do once it holds the device, in the order it will do it.
 *
 * Written per kind rather than derived, because the point is to describe the work
 * to somebody waiting on it — a stage name lifted from the runner ("Scoring
 * held-out text") says what the code is doing, not what it is for.
 */
function whatHappens(job: Job): { title: string; steps: Step[] } {
  switch (job.kind) {
    case "train":
      return {
        title: "What it will do",
        steps: [
          {
            name: "Build the tokenizer",
            blurb:
              "A BPE train over the corpus, unless this is a fine-tune — those inherit the base model's.",
          },
          {
            name: "Train",
            blurb:
              "Loss per chunk, gradient diagnostics and checkpoints appear here as they land.",
          },
          {
            name: "Write the model",
            blurb: "The weights are saved and the model becomes benchmarkable.",
          },
        ],
      }
    case "benchmark":
      return {
        title: "What it will score",
        // The two suites score different things, so the running order shown to
        // somebody waiting has to be the one their job will actually work through.
        steps: benchmarkSections(benchmarkSuiteOf(job)).map((section) => ({
          name: `${section.name} · ${Math.round(section.weight * 100)}%`,
          blurb: section.blurb,
        })),
      }
    case "chat":
      return {
        title: "What it will do",
        steps: [
          { name: "Reply", blurb: "The turn streams into the conversation." },
        ],
      }
  }
}

/** `2nd`, `3rd`, `11th` — the place in line, said the way a person would. */
function ordinal(n: number): string {
  const rest = n % 100
  if (rest >= 11 && rest <= 13) return `${n}th`
  switch (n % 10) {
    case 1:
      return `${n}st`
    case 2:
      return `${n}nd`
    case 3:
      return `${n}rd`
    default:
      return `${n}th`
  }
}
