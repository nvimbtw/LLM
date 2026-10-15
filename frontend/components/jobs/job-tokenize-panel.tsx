"use client"

import { ButtonLink } from "@/components/ui/button-link"
import { Appendix } from "@/components/training/appendix"
import { Panel } from "@/components/ui/panel"
import { Progress } from "@/components/ui/progress"
import type { useJob } from "@/hooks/use-job"

/**
 * A vocabulary build's body: three counts and a bar.
 *
 * It reports the same `tokenize` frames training's phase 1 does — the run *is*
 * that phase, extracted — so there is no new frame shape here, only fewer of
 * them: a build has no loss to plot and no diagnostics to sample.
 */
export interface JobTokenizePanelProps {
  run: ReturnType<typeof useJob>
}

/** Shows vocabulary build progress and links its completed output to the resource detail page. */
export function JobTokenizePanel({ run }: JobTokenizePanelProps) {
  const { tokenize } = run
  const pct = Math.round((tokenize.progress ?? 0) * 100)

  return (
    <div className="space-y-6">
      <Panel
        variant="open"
        title={tokenize.done ? "Built" : "Learning the merges"}
      >
        <div className="space-y-5">
          <dl className="grid grid-cols-1 gap-x-8 gap-y-5 border-y border-border py-4 sm:grid-cols-3">
            <Figure
              label="Documents"
              value={count(tokenize.documents ?? tokenize.files)}
            />
            <Figure label="Vocab" value={count(tokenize.vocab)} />
            <Figure label="Merges" value={count(tokenize.merges)} />
          </dl>

          <div className="space-y-1.5">
            <Progress value={pct} />
            <div className="text-right text-xs text-muted-foreground tabular-nums">
              {pct}%
            </div>
          </div>
        </div>
      </Panel>

      {run.job?.output.type === "tokenizer" && run.job.output.id && (
        <ButtonLink variant="text" href={`/tokenizers/${run.job.output.id}`}>
          Open vocabulary
        </ButtonLink>
      )}
      <Appendix log={run.log} checkpoints={[]} />
    </div>
  )
}

/** Formats an available build count compactly, retaining the unavailable marker. */
function count(n: number | undefined): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

/** Displays one measured vocabulary-build figure. */
function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="pr-6">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="mt-1.5 figure text-2xl">{value}</dd>
    </div>
  )
}
