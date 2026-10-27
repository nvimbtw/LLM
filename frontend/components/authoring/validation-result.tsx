"use client"

import * as React from "react"
import Link from "next/link"
import { useValidationJob } from "@/hooks/use-validation-job"
import type {
  ValidationReport,
  ValidationRun,
} from "@/lib/architecture/contracts.generated"
import { SavedArchitectureGraph } from "@/components/architecture-graph/saved-graph"
import {
  RetainedSourceDialog,
  type RetainedSourceSelection,
} from "@/components/architecture-graph/retained-source"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"

export interface ValidationResultProps {
  run: ValidationRun
  stale: boolean
  onReport?: (report: ValidationReport) => void
  actions?: (report: ValidationReport) => React.ReactNode
}

/** Observes one durable saved snapshot, displaying verified verdict/diagnostics and shared retained graphs; browser disposal never owns worker cancellation. */
export function ValidationResult({
  run,
  stale,
  onReport,
  actions,
}: ValidationResultProps) {
  const [source, selectSource] = React.useState<RetainedSourceSelection | null>(
    null
  )
  const { observer, state } = useValidationJob(run.run_id)
  const report = state?.report
  React.useEffect(() => {
    if (report) onReport?.(report)
  }, [report, onReport])
  return (
    <div className="grid gap-3 border-t border-border pt-4">
      <RetainedSourceDialog
        kind="validations"
        id={run.run_id}
        selection={source}
        onClose={() => selectSource(null)}
      />
      <Link
        className="text-sm underline"
        href={`/arch/validations/${run.run_id}`}
      >
        Open validation job
      </Link>
      <p className="text-sm" role="status">
        {report?.verdict ?? state?.run?.status ?? run.status}
        {stale && run.config.mode !== "inspect"
          ? " · Stale configuration: revalidate before use"
          : ""}
      </p>
      {state?.error && (
        <LoadError
          title="Validation observation failed."
          hint={state.error}
          onRetry={() => void observer?.refresh()}
        />
      )}
      {report?.diagnostics.map((item, index) => (
        <p key={index} className="text-sm">
          {item.phase}: {item.message}
          {item.source && (
            <Button
              variant="link"
              className="font-mono text-xs break-all"
              onClick={() =>
                selectSource({
                  revision: item.source!.revision_id,
                  line: item.source!.line ?? null,
                })
              }
            >
              {item.source.file}
              {item.source.line ? `:${item.source.line}` : ""}
            </Button>
          )}
        </p>
      ))}
      {report && (
        <p className="text-sm">
          Unique parameters:{" "}
          {report.unique_parameter_count == null
            ? "Unavailable"
            : report.unique_parameter_count.toLocaleString()}
        </p>
      )}
      {report && actions?.(report)}
      {report?.receipt?.graph_manifest_hash && (
        <SavedArchitectureGraph
          kind="validations"
          id={run.run_id}
          receipt={report.receipt}
          architectureView={report.declaration == null}
          stale={stale}
          onSource={(revision, line) => selectSource({ revision, line })}
        />
      )}
    </div>
  )
}
