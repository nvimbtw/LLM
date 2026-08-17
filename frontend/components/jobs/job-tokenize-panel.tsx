"use client"

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

export function JobTokenizePanel({ run }: JobTokenizePanelProps) {
  const { tokenize } = run
  const pct = Math.round((tokenize.progress ?? 0) * 100)

  return (
    <div className="space-y-6">
      {/* Full width, no rail. The config recap that would sit beside it renders
          nothing — the engine's run row carries no `config` — so the column was
          a quarter page of gutter. */}
      <Panel
        title={tokenize.done ? "Built" : "Learning the merges"}
        description={
          tokenize.done
            ? "Any run can now be pointed at it"
            : "One pass of byte-level BPE"
        }
      >
        <div className="space-y-5">
          <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-border bg-border/60">
            <Figure label="Files" value={count(tokenize.files)} />
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

      <Appendix log={run.log} checkpoints={[]} />
    </div>
  )
}

function count(n: number | undefined): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background px-5 py-4">
      <dt className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-1.5 font-display text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
    </div>
  )
}
