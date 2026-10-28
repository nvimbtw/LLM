"use client"

import * as React from "react"
import Link from "next/link"
import { useArchitectureCatalog } from "@/hooks/use-architecture-catalog"
import { useArchitectureHistory } from "@/hooks/use-architecture-history"
import { useSelectedTrainingArchitecture } from "@/hooks/use-selected-training-architecture"
import { useTrainingArchitecturePreview } from "@/hooks/use-training-architecture-preview"
import { useValidationSubmission } from "@/hooks/use-validation-submission"
import {
  trainingParameterErrors,
  trainingSelectionKey,
  withoutTrainingReceipt,
  type TrainingArchitectureCheck,
} from "@/lib/training-selection"
import { identityDigest } from "@/lib/architecture/decoder"
import type { TrainConfig } from "@/lib/training"
import type {
  ArchitectureRevisionView,
  RevisionRef,
  ValidationReport,
} from "@/lib/architecture/contracts.generated"
import type { ParameterValue } from "@/lib/architecture/parameters"
import { ParameterFields } from "@/components/authoring/parameter-fields"
import { ValidationResult } from "@/components/authoring/validation-result"
import { Button } from "@/components/ui/button"
import { SearchField } from "@/components/ui/search-field"
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

export interface ArchitectureChoiceProps {
  config: TrainConfig
  disabled: boolean
  onChange: (config: TrainConfig) => void
  onValidity: (value: TrainingArchitectureCheck) => void
}

/** Selects only explicit accepted recipe revisions, edits declared parameters and reads an engine-owned prospective profile before queueing; numerical validation remains an explicit durable action. */
export function ArchitectureChoice({
  config,
  disabled,
  onChange,
  onValidity,
}: ArchitectureChoiceProps) {
  const [choosing, choose] = React.useState(false)
  const [actionError, setActionError] = React.useState<string | null>(null)
  const selected = useSelectedTrainingArchitecture(
    config.architecture?.revision_id ?? null
  )
  const preview = useTrainingArchitecturePreview(config)
  const { owner: submission, state: submitted } = useValidationSubmission(
    "architecture",
    selected.value?.accepted.reference.resource_id ?? ""
  )
  const key = trainingSelectionKey(config)
  const parameters =
    config.architecture && selected.value
      ? trainingParameterErrors(config.architecture, selected.value.recipe)
      : {}
  const errorList = !config.architecture
    ? []
    : selected.error
      ? [selected.error]
      : !selected.value
        ? ["Read the selected recipe before starting."]
        : selected.value.recipe.interface !== "causal_lm"
          ? ["Training requires a causal language-model recipe."]
          : Object.entries(parameters).map(
              ([name, message]) => `${name}: ${message}`
            )
  if (config.architecture && !errorList.length && preview.stale)
    errorList.push(
      preview.error ?? "Read the current training profile before starting."
    )
  const errorKey = errorList.join("\n")
  React.useEffect(() => {
    onValidity({ key, errors: errorKey ? errorKey.split("\n") : [] })
  }, [key, errorKey, onValidity])
  const commandKey = `${key}:${preview.value ? identityDigest(preview.value) : "unread"}`
  const stale = preview.stale || submitted?.key !== commandKey
  const busy = disabled || preview.loading || Boolean(submitted?.pending)

  /** Deliberately adopts exactly the clicked revision, resetting incompatible overrides and previous receipt claims. */
  function select(reference: RevisionRef) {
    onChange({
      ...withoutTrainingReceipt(config),
      architecture: { revision_id: reference.revision_id, parameters: {} },
    })
    choose(false)
    setActionError(null)
  }

  /** Changes one declared constructor value while retaining the immutable selected revision and invalidating old evidence. */
  function setParameter(name: string, value: ParameterValue | undefined) {
    if (!config.architecture) return
    const parameters = { ...config.architecture.parameters }
    if (value === undefined) delete parameters[name]
    else parameters[name] = value
    onChange({
      ...withoutTrainingReceipt(config),
      architecture: {
        ...config.architecture,
        parameters,
        validation_run_id: null,
      },
    })
  }

  /** Explicitly re-reads prospective engine bindings after edits and drops carried receipt authorization before that read. */
  function refreshProfile() {
    onChange(withoutTrainingReceipt(config))
    setActionError(null)
    void preview.load()
  }

  /** Queues exact accepted source and current effective profile only after a real final tokenizer exists; pending tokenization belongs to the future training job. */
  async function validate() {
    const value = preview.value
    if (
      busy ||
      preview.stale ||
      !value?.tokenizer ||
      !config.architecture ||
      !submission
    )
      return
    setActionError(null)
    try {
      await submission.submit(commandKey, {
        schema_version: 1,
        mode: "full",
        subject: { kind: "revision", reference: value.architecture },
        parameters: config.architecture.parameters,
        profile: value.profile,
        tokenizer: value.tokenizer,
        context_window: value.context_window,
      })
    } catch (cause) {
      setActionError(
        cause instanceof Error
          ? cause.message.slice(0, 2000)
          : "Validation did not queue."
      )
    }
  }

  /** Retains an explicitly selected verified report only for unchanged form/profile inputs; actual enqueue still requires final admitted numerical preflight. */
  function retainReceipt(report: ValidationReport) {
    if (
      busy ||
      stale ||
      !config.architecture ||
      report.verdict !== "valid" ||
      !report.receipt ||
      report.run_id !== submitted?.run?.run_id
    )
      return
    onChange({
      ...config,
      architecture: {
        ...config.architecture,
        validation_run_id: report.run_id,
      },
      architecture_validation_key: trainingSelectionKey(config),
    })
  }

  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm">
          {config.architecture
            ? "Pinned accepted recipe"
            : "Standard model settings"}
        </span>
        <Button
          type="button"
          variant="text"
          size="sm"
          disabled={busy}
          onClick={() => choose((current) => !current)}
        >
          {choosing ? "Close recipe selection" : "Choose a recipe"}
        </Button>
        {config.architecture && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={busy}
            onClick={() =>
              onChange({
                ...withoutTrainingReceipt(config),
                architecture: undefined,
              })
            }
          >
            Use standard settings
          </Button>
        )}
      </div>
      {choosing && <TrainingRecipeCatalog disabled={busy} onSelect={select} />}
      {config.architecture && (
        <>
          <p className="font-mono text-xs break-all">
            Revision {config.architecture.revision_id}
          </p>
          {selected.loading && <Skeleton className="h-36 w-full" />}
          {selected.error && (
            <LoadError
              title="Selected recipe did not load."
              hint={selected.error}
              onRetry={selected.retry}
            />
          )}
          {selected.value && (
            <>
              <PinnedRecipeHistory
                key={selected.value.accepted.reference.resource_id}
                value={selected.value}
                disabled={busy}
                onSelect={select}
              />
              <ParameterFields
                variant="inline"
                declarations={selected.value.recipe.parameters}
                values={Object.fromEntries(
                  Object.entries(selected.value.recipe.parameters).map(
                    ([name, declaration]) => [
                      name,
                      config.architecture!.parameters[name] ??
                        declaration.default ??
                        undefined,
                    ]
                  )
                )}
                errors={parameters}
                disabled={busy}
                onChange={setParameter}
              />
              <Button
                type="button"
                className="w-fit"
                variant="text"
                disabled={
                  busy ||
                  Object.keys(parameters).length > 0 ||
                  selected.value.recipe.interface !== "causal_lm"
                }
                onClick={refreshProfile}
              >
                {preview.loading
                  ? "Reading profile…"
                  : "Read current training profile"}
              </Button>
            </>
          )}
          {preview.error && <FormError>{preview.error}</FormError>}
          {actionError && <FormError>{actionError}</FormError>}
          {submitted?.error && <FormError>{submitted.error}</FormError>}
          {preview.value && (
            <section
              aria-label="Prospective training profile"
              className="grid gap-3 pt-4 text-sm"
            >
              {preview.stale && (
                <p role="status">
                  Configuration changed. Read its current profile before use.
                </p>
              )}
              <p>
                {preview.value.profile.device} ·{" "}
                {preview.value.profile.precision} · context{" "}
                {preview.value.context_window.toLocaleString()} tokens · batch
                1–{preview.value.profile.domains[0].batch_max}
              </p>
              <p className="text-muted-foreground">
                {preview.value.tokenizer
                  ? "Final vocabulary facts are available. The training job still performs its final numerical preflight."
                  : "Vocabulary is pending. The training job will build it, bind real token IDs, then perform full numerical preflight before optimization."}
              </p>
              <details>
                <summary>Exact prospective bindings and profile</summary>
                <pre className="max-h-80 overflow-auto p-3 text-xs">
                  {JSON.stringify(preview.value, null, 2)}
                </pre>
              </details>
              <Button
                type="button"
                className="w-fit"
                disabled={
                  busy ||
                  preview.stale ||
                  !preview.value.tokenizer ||
                  !submission
                }
                onClick={() => void validate()}
              >
                Queue profile validation
              </Button>
            </section>
          )}
          {config.architecture.validation_run_id && (
            <p role="status" className="text-sm">
              {config.architecture_validation_key === key
                ? "Recorded validation selected for unchanged inputs."
                : "Recorded validation is stale and will not be sent."}
            </p>
          )}
          {submitted?.run && (
            <ValidationResult
              key={submitted.run.run_id}
              run={submitted.run}
              stale={stale}
              actions={(report) =>
                report.verdict === "valid" ? (
                  <Button
                    type="button"
                    className="w-fit"
                    disabled={busy || stale}
                    onClick={() => retainReceipt(report)}
                  >
                    Use recorded validation
                  </Button>
                ) : null
              }
            />
          )}
        </>
      )}
    </div>
  )
}

export interface TrainingRecipeCatalogProps {
  disabled: boolean
  onSelect: (reference: RevisionRef) => void
}

/** Offers a bounded unarchived recipe catalog with explicit search/paging; choosing a displayed head is deliberate and never a background latest update. */
function TrainingRecipeCatalog({
  disabled,
  onSelect,
}: TrainingRecipeCatalogProps) {
  const catalog = useArchitectureCatalog("architecture")
  return (
    <div className="grid gap-3 py-3">
      <div className="flex items-end gap-2">
        <label className="grid gap-2 text-sm">
          Search accepted recipes
          <SearchField
            value={catalog.search}
            maxLength={200}
            disabled={disabled}
            onChange={(event) => catalog.setSearch(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault()
                catalog.filter({ search: catalog.search })
              }
            }}
          />
        </label>
        <Button
          type="button"
          variant="text"
          disabled={disabled}
          onClick={() => catalog.filter({ search: catalog.search })}
        >
          Search
        </Button>
      </div>
      {catalog.loading && <Skeleton className="h-24 w-full" />}
      {catalog.error && (
        <LoadError
          title="Recipe catalog did not load."
          hint={catalog.error}
          onRetry={catalog.refresh}
        />
      )}
      <ResourceList density="compact" className="max-h-72 overflow-auto">
        {catalog.page?.items
          .filter((item) => !item.archived && item.accepted_revision)
          .map((item) => (
            <ResourceRow density="compact" key={item.resource_id}>
              <Button
                type="button"
                variant="ghost"
                disabled={disabled}
                className="h-auto w-full justify-start py-0 text-left whitespace-normal"
                onClick={() => onSelect(item.accepted_revision!)}
              >
                {item.name} ·{" "}
                {item.accepted_revision!.content_hash.slice(0, 12)}
              </Button>
            </ResourceRow>
          ))}
      </ResourceList>
      {catalog.page && (
        <div className="flex gap-2">
          <Button
            type="button"
            size="sm"
            variant="text"
            disabled={disabled || !catalog.query.cursor}
            onClick={() => catalog.filter({})}
          >
            First page
          </Button>
          <Button
            type="button"
            size="sm"
            variant="text"
            disabled={disabled || !catalog.page.next_cursor}
            onClick={catalog.next}
          >
            Next page
          </Button>
        </div>
      )}
    </div>
  )
}

export interface PinnedRecipeHistoryProps {
  value: ArchitectureRevisionView
  disabled: boolean
  onSelect: (reference: RevisionRef) => void
}

/** Opens the selected immutable recipe and offers deliberate older/newer revision adoption through bounded history; changes reset parameter overrides. */
function PinnedRecipeHistory({
  value,
  disabled,
  onSelect,
}: PinnedRecipeHistoryProps) {
  const reference = value.accepted.reference
  const history = useArchitectureHistory(
    "architecture",
    reference.resource_id,
    reference.revision_id
  )
  return (
    <details className="text-sm">
      <summary>Selected recipe and other accepted revisions</summary>
      <div className="grid gap-3 pt-3">
        <Link
          className="underline"
          href={`/arch/${reference.resource_id}?revision=${reference.revision_id}`}
        >
          Open immutable recipe
        </Link>
        <p className="text-xs text-muted-foreground">
          Choosing another revision resets parameter overrides. Existing pins do
          not adopt new revisions automatically.
        </p>
        {history.loading && <Skeleton className="h-16 w-full" />}
        {history.error && (
          <LoadError
            title="Recipe history did not load."
            hint={history.error}
            onRetry={history.retry}
          />
        )}
        <div className="flex flex-wrap gap-2">
          {history.value?.items.map((item) => (
            <Button
              key={item.reference.revision_id}
              type="button"
              size="sm"
              variant="text"
              disabled={
                disabled || item.reference.revision_id === reference.revision_id
              }
              onClick={() => onSelect(item.reference)}
            >
              Use revision {item.revision_number}
            </Button>
          ))}
        </div>
        {history.value && (
          <div className="flex gap-2">
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={!history.cursor || disabled}
              onClick={history.first}
            >
              First page
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={!history.value.next_cursor || disabled}
              onClick={history.next}
            >
              Next page
            </Button>
          </div>
        )}
      </div>
    </details>
  )
}
