"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { AdminBoundary } from "@/components/admin/admin-boundary"
import { useValidationJob } from "@/hooks/use-validation-job"
import { jobFromValidation } from "@/lib/jobs"
import { ConfigRecap } from "@/components/jobs/config-recap"
import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { SavedArchitectureGraph } from "@/components/architecture-graph/saved-graph"
import { JobQueuedPanel } from "@/components/jobs/job-queued-panel"
import {
  RetainedSourceDialog,
  type RetainedSourceSelection,
} from "@/components/architecture-graph/retained-source"

export interface ValidationDetailProps {
  id: string
}

/** Gates private validation reports/diagnostics before mounting their session-scoped observer. */
export function ValidationDetail({ id }: ValidationDetailProps) {
  return (
    <AdminBoundary>
      <ValidationContent key={id} id={id} />
    </AdminBoundary>
  )
}

/** Presents actual lifecycle, verified verdict and observed stages, keeping explicit controls separate from navigation cleanup. */
function ValidationContent({ id }: ValidationDetailProps) {
  const router = useRouter()
  const [source, selectSource] = React.useState<RetainedSourceSelection | null>(
    null
  )
  const { observer, state, loading, connectionError } = useValidationJob(id)
  if (loading) return <Skeleton className="h-96 w-full" />
  if (!observer || !state?.run)
    return (
      <LoadError
        title="Validation did not load."
        hint={state?.error ?? undefined}
        onRetry={() => {
          void observer?.refresh()
        }}
      />
    )
  const run = state.run
  const report = state.report
  const terminal = ["done", "failed", "cancelled"].includes(run.status)
  const verdict = report
    ? {
        valid: "Validated",
        invalid: "Invalid",
        inspection_only: "Inspected",
        provisional: "Provisional",
      }[report.verdict]
    : run.status === "done"
      ? "Report unavailable"
      : run.status

  /** Opens only a confirmed fresh retry identity while this observer still owns the requesting session. */
  async function retry() {
    const next = await observer!.retry()
    if (next) router.push(`/arch/validations/${next.run_id}`)
  }

  /** Returns to history only after an explicitly requested terminal deletion succeeds for this owner. */
  async function remove() {
    if (await observer!.remove()) router.push("/admin")
  }

  return (
    <>
      <PageHeader
        title={run.label}
        back={{ href: "/admin", label: "Administration" }}
        meta={`Validation · ${verdict}`}
        action={
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => void observer.refresh()}>
              Refresh
            </Button>
            {terminal ? (
              <>
                <Button
                  disabled={state.controlling || state.retryUncertain}
                  onClick={() => void retry()}
                >
                  Retry saved snapshot
                </Button>
                <Button
                  variant="outline"
                  disabled={state.controlling}
                  onClick={() => void remove()}
                >
                  Delete history
                </Button>
              </>
            ) : (
              <Button
                variant="outline"
                disabled={
                  state.controlling ||
                  state.cancelRequested ||
                  run.status === "cancelling"
                }
                onClick={() => void observer.cancel()}
              >
                {state.cancelRequested || run.status === "cancelling"
                  ? "Stopping…"
                  : "Cancel validation"}
              </Button>
            )}
          </div>
        }
      />
      <div className="grid gap-6">
        <RetainedSourceDialog
          kind="validations"
          id={id}
          selection={source}
          onClose={() => selectSource(null)}
        />
        {state.error && (
          <LoadError
            title="Validation observation needs attention."
            hint={state.error}
            onRetry={() => void observer.refresh()}
          />
        )}
        {state.actionError && <FormError>{state.actionError}</FormError>}
        {run.error && <FormError>{run.error}</FormError>}
        {connectionError && (
          <p className="text-sm text-muted-foreground">
            Live events disconnected. Status continues through REST:{" "}
            {connectionError}
          </p>
        )}
        {run.status === "queued" || run.status === "waiting" ? (
          <JobQueuedPanel job={jobFromValidation(run)} />
        ) : (
          <ConfigRecap job={jobFromValidation(run)} />
        )}
        <Panel
          variant="open"
          title={report ? "Validation stages" : "Last observed stages"}
        >
          <ol className="divide-y divide-border">
            {state.stages.map((stage) => (
              <li
                key={stage.stage}
                className="flex justify-between gap-4 py-2 text-sm"
              >
                <span>{stage.stage}</span>
                <span>
                  {stage.outcome}
                  {stage.elapsed_ms != null ? ` · ${stage.elapsed_ms} ms` : ""}
                </span>
              </li>
            ))}
          </ol>
          {!state.stages.length && (
            <p className="text-sm text-muted-foreground">
              {terminal
                ? "No stage measurements available."
                : "Waiting for execution updates."}
            </p>
          )}
        </Panel>
        {report && (
          <Panel variant="open" title="Published result">
            <div className="grid gap-2 text-sm">
              <p>
                {report.verdict === "inspection_only"
                  ? "Declarations are available. Full validation is required before acceptance."
                  : report.verdict === "provisional"
                    ? "Final tokenizer bindings and numerical preflight are still required."
                    : report.verdict === "invalid"
                      ? "This configuration did not pass validation."
                      : "This exact saved configuration passed its recorded profile."}
              </p>
              <p>
                Unique parameters:{" "}
                {report.unique_parameter_count == null
                  ? "Unavailable"
                  : report.unique_parameter_count.toLocaleString()}
              </p>
              {report.receipt && (
                <>
                  <code className="break-all">
                    Receipt {report.receipt.identity_hash}
                  </code>
                  <details>
                    <summary>
                      Exact source, bindings and runtime profile
                    </summary>
                    <pre className="max-h-96 overflow-auto rounded border border-border p-3 text-xs">
                      {JSON.stringify(report.receipt.identity, null, 2)}
                    </pre>
                  </details>
                </>
              )}
            </div>
          </Panel>
        )}
        {report?.receipt?.graph_manifest_hash && (
          <Panel variant="open" title="Captured architecture">
            <SavedArchitectureGraph
              kind="validations"
              id={id}
              receipt={report.receipt}
              architectureView={report.declaration == null}
              onSource={(revision, line) => selectSource({ revision, line })}
            />
          </Panel>
        )}
        {state.diagnostics.length > 0 && (
          <Panel variant="open" title="Diagnostics">
            <ul className="grid gap-3">
              {state.diagnostics.map((diagnostic, index) => (
                <li key={`${diagnostic.code}:${index}`} className="text-sm">
                  <p>
                    {diagnostic.phase} · {diagnostic.code}
                  </p>
                  <p>{diagnostic.message}</p>
                  {diagnostic.yaml_path && (
                    <code className="break-all">{diagnostic.yaml_path}</code>
                  )}
                  {diagnostic.source && (
                    <Button
                      variant="link"
                      className="font-mono text-xs break-all"
                      onClick={() =>
                        selectSource({
                          revision: diagnostic.source!.revision_id,
                          line: diagnostic.source!.line ?? null,
                        })
                      }
                    >
                      {diagnostic.source.file}
                      {diagnostic.source.line
                        ? `:${diagnostic.source.line}`
                        : ""}
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          </Panel>
        )}
      </div>
    </>
  )
}
