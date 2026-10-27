"use client"

import * as React from "react"
import {
  useDraftValidation,
  type DraftValidationInputs,
} from "@/hooks/use-draft-validation"
import type { ValidationReport } from "@/lib/architecture/contracts.generated"
import { MAX_TENSOR_SHAPE_CHARACTERS } from "@/lib/architecture/validation-form"
import { ParameterFields } from "./parameter-fields"
import { RuntimeProfileFields } from "./runtime-profile-fields"
import { BuiltVocabularyPicker } from "./built-vocabulary-picker"
import { ValidationResult } from "./validation-result"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import { FormError } from "@/components/ui/error-state"

export type DraftValidationProps = DraftValidationInputs

/** Presents explicit inspection, declared form inputs and stale-aware validation results for one saved draft. */
export function DraftValidation(props: DraftValidationProps) {
  const {
    metadata,
    declaration,
    profile,
    parameters,
    tokenizer,
    error,
    busy,
    accepted,
    submitted,
    submission,
    stale,
    unavailable,
    inspect,
    validate,
    accept,
    inputShape,
    setDeclaredInputs,
    setParameters,
    setProfile,
    setShapes,
    setTokenizer,
    retryMetadata,
  } = useDraftValidation(props)
  const { readOnly } = props
  /** Applies only verified declaration-only reports to this saved-source form. */
  const observedReport = React.useCallback(
    (report: ValidationReport) => {
      if (report.verdict === "inspection_only" && report.declaration)
        setDeclaredInputs(report.declaration)
    },
    [setDeclaredInputs]
  )

  return (
    <Panel variant="open" title="Inspect and validate">
      <div className="grid gap-5">
        <p className="max-w-prose text-sm text-muted-foreground">
          Save your draft, inspect its declarations, then validate an explicit
          configuration. Acceptance retains an immutable revision after full
          validation.
        </p>
        {unavailable && (
          <p className="text-sm text-muted-foreground">
            {readOnly
              ? "Copy or unarchive this resource to validate a new draft."
              : "Save the current draft before inspecting, validating or accepting it."}
          </p>
        )}
        {error && <FormError>{error}</FormError>}
        {metadata?.tokenizerError && (
          <FormError>{metadata.tokenizerError}</FormError>
        )}
        {((!metadata && error) || metadata?.tokenizerError) && (
          <Button
            variant="outline"
            className="w-fit"
            disabled={busy}
            onClick={retryMetadata}
          >
            Reload metadata
          </Button>
        )}
        {submitted?.error && <FormError>{submitted.error}</FormError>}
        {accepted && (
          <p role="status" className="text-sm break-all">
            Accepted revision {accepted}
          </p>
        )}
        <Button
          className="w-fit"
          variant="outline"
          disabled={
            unavailable ||
            busy ||
            submitted?.pending ||
            !metadata ||
            !submission ||
            Boolean(declaration)
          }
          onClick={() => void inspect()}
        >
          {busy
            ? "Preparing…"
            : declaration
              ? "Declarations inspected"
              : "Inspect saved declarations"}
        </Button>
        {declaration && profile && metadata && (
          <>
            <ParameterFields
              declarations={declaration.parameters}
              values={parameters}
              disabled={unavailable || busy}
              onChange={(name, value) =>
                setParameters((current) => ({ ...current, [name]: value }))
              }
            />
            <RuntimeProfileFields
              profile={profile}
              devices={metadata.schema.available_devices ?? []}
              disabled={unavailable || busy}
              onChange={setProfile}
            />
            {!(
              "interface" in declaration &&
              declaration.interface === "causal_lm"
            ) && (
              <fieldset disabled={unavailable || busy} className="grid gap-3">
                <legend className="mb-3 text-sm font-medium">
                  Synthetic tensor inputs
                </legend>
                {Object.entries(declaration.inputs).map(([name, port]) => (
                  <label key={name} className="grid gap-2 text-sm">
                    {name} · {port.dtype} · [{port.shape.join(", ")}]
                    <Input
                      value={inputShape(name)}
                      maxLength={MAX_TENSOR_SHAPE_CHARACTERS}
                      placeholder="Positive dimensions separated by commas"
                      onChange={(event) =>
                        setShapes((current) => ({
                          ...current,
                          [name]: event.target.value,
                        }))
                      }
                    />
                  </label>
                ))}
                <p className="text-xs text-muted-foreground">
                  Every dimension must be explicit. Ordinary blocks currently
                  require singleton profile domains and exact input shapes.
                </p>
              </fieldset>
            )}
            <BuiltVocabularyPicker
              items={metadata.tokenizers}
              value={tokenizer}
              required={
                "interface" in declaration &&
                declaration.interface === "causal_lm"
              }
              disabled={unavailable || busy}
              onChange={setTokenizer}
            />
            <Button
              className="w-fit"
              disabled={unavailable || busy || submitted?.pending}
              onClick={() => void validate()}
            >
              {submitted?.pending
                ? "Queueing…"
                : submitted?.uncertain && !stale
                  ? "Recover queued validation"
                  : "Queue full validation"}
            </Button>
          </>
        )}
        {submitted?.run && (
          <ValidationResult
            key={submitted.run.run_id}
            run={submitted.run}
            stale={stale}
            onReport={observedReport}
            actions={(report) =>
              report.verdict === "valid" ? (
                <Button
                  className="w-fit"
                  disabled={unavailable || busy || stale}
                  onClick={() => void accept(report)}
                >
                  Accept validated revision
                </Button>
              ) : null
            }
          />
        )}
      </div>
    </Panel>
  )
}
