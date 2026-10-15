"use client"

import { PageHeader } from "@/components/ui/page-header"
import { FormError } from "@/components/ui/error-state"
import { ScoreCard } from "./score-card"
import { CorpusPanel } from "./corpus-panel"
import { AblationPanel } from "./ablation-panel"
import { GenerationPanel } from "./generation-panel"
import { MarkupPanel } from "./markup-panel"
import {
  formatDuration,
  isLanguageReport,
  isMarkupReport,
  type BenchmarkRunDetail,
} from "@/lib/benchmark"

export interface BenchmarkReportViewProps {
  run: BenchmarkRunDetail
}

/** Presents a historical report even when its original execution row has been pruned or used a different UUID. */
export function BenchmarkReportView({ run }: BenchmarkReportViewProps) {
  return (
    <>
      <PageHeader
        title={run.model_name}
        back={{ href: "/bench", label: "Bench" }}
        description={`${run.suite} · ${run.status}`}
      />
      <BenchmarkReport run={run} />
    </>
  )
}

/** Renders the same saved report within history, historical UUID pages and durable execution details. */
export function BenchmarkReport({ run }: BenchmarkReportViewProps) {
  const report = run.report
  return (
    <div className="space-y-6">
      {run.error && <FormError>{run.error}</FormError>}
      {report ? (
        <>
          <ScoreCard report={report} />
          {isMarkupReport(report) ? (
            <MarkupPanel report={report} />
          ) : isLanguageReport(report) ? (
            <>
              <div className="grid gap-6 xl:grid-cols-2">
                <CorpusPanel corpus={report.corpus} />
                {report.ablation && (
                  <AblationPanel ablation={report.ablation} />
                )}
              </div>
              {report.surface && (
                <GenerationPanel
                  surface={report.surface}
                  samples={report.samples}
                />
              )}
            </>
          ) : null}
          <p className="text-xs text-muted-foreground tabular-nums">
            {formatDuration(report.total_secs)}
          </p>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">No report recorded.</p>
      )}
    </div>
  )
}
