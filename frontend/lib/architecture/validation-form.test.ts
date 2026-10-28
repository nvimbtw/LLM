import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import {
  configuredDraftValidation,
  draftAcceptance,
  suggestedInputShape,
  MAX_TENSOR_SHAPE_CHARACTERS,
} from "./validation-form"
import { decodeArchitecture, identityDigest } from "./decoder"
import type { ArchitectureContracts } from "./contracts.generated"

const cases = JSON.parse(
  readFileSync(
    new URL(
      "../../../contracts/modular-architecture/v1/api/cases.json",
      import.meta.url
    ),
    "utf8"
  )
).cases

/** Reads synthetic shared DTO fixtures for client-boundary behavior, independently of production numerical evidence. */
function fixture<K extends keyof ArchitectureContracts>(
  model: K,
  name = `${model}-positive`
): ArchitectureContracts[K] {
  return decodeArchitecture(
    model,
    structuredClone(
      cases.find((item: { name: string }) => item.name === name).value
    )
  )
}

test("synthetic tensor forms leave unknown dimensions unset, retain zero values and enforce actual element budgets", () => {
  const declaration = fixture(
    "ValidationReport",
    "inspection-report"
  ).declaration!
  const run = fixture("ValidationRun")
  const profile = run.config.profile!
  assert.equal(
    suggestedInputShape(declaration.inputs.hidden, profile),
    "1, 1, "
  )
  const form = { parameters: { gain: 0 }, profile, shapes: {} }
  assert.throws(() =>
    configuredDraftValidation(
      run.config.subject as ArchitectureContracts["DraftSubject"],
      declaration,
      form,
      null
    )
  )
  const request = configuredDraftValidation(
    run.config.subject as ArchitectureContracts["DraftSubject"],
    declaration,
    { ...form, shapes: { hidden: "1, 1, 4" } },
    null
  )
  assert.equal(request.parameters?.gain, 0)
  assert.deepEqual(request.tensor_inputs?.hidden, {
    shape: [1, 1, 4],
    dtype: "float32",
  })
  assert.equal("context_window" in request, false)
  assert.throws(
    () =>
      configuredDraftValidation(
        run.config.subject as ArchitectureContracts["DraftSubject"],
        declaration,
        { ...form, shapes: { hidden: "1000000, 1000000, 4" } },
        null
      ),
    /budget/
  )
})

test("tensor form text is bounded before parsing and still enforces the wire rank limit", () => {
  const declaration = fixture(
    "ValidationReport",
    "inspection-report"
  ).declaration!
  const run = fixture("ValidationRun")
  const subject = run.config.subject as ArchitectureContracts["DraftSubject"]
  const form = {
    parameters: { gain: 0 },
    profile: run.config.profile!,
    shapes: {},
  }
  const request = configuredDraftValidation(
    subject,
    declaration,
    {
      ...form,
      shapes: { hidden: "1,1,4".padEnd(MAX_TENSOR_SHAPE_CHARACTERS, " ") },
    },
    null
  )
  assert.deepEqual(request.tensor_inputs?.hidden.shape, [1, 1, 4])
  assert.throws(
    () =>
      configuredDraftValidation(
        subject,
        declaration,
        {
          ...form,
          shapes: { hidden: ",".repeat(MAX_TENSOR_SHAPE_CHARACTERS + 1) },
        },
        null
      ),
    /Tensor shape must be at most 256 characters/
  )
  assert.throws(() =>
    configuredDraftValidation(
      subject,
      declaration,
      { ...form, shapes: { hidden: "1,1,1,1,1,1,1,1,1" } },
      null
    )
  )
})

test("acceptance requires the current saved draft and full report; local edits or a configuration generation change immediately invalidate it", () => {
  const run = fixture("ValidationRun")
  const report = fixture("ValidationReport")
  report.receipt!.identity.input_profile_digest = identityDigest({
    tensor_inputs: run.config.tensor_inputs,
    domains: run.config.profile!.domains,
  })
  report.receipt!.identity_hash = identityDigest(report.receipt!.identity)
  const subject = run.config.subject as ArchitectureContracts["DraftSubject"]
  const draft = {
    saved: {
      resource_id: subject.resource_id,
      draft_version: subject.draft_version,
      content_hash: subject.content_hash,
    },
    savedText: "saved",
    text: "saved",
    editRevision: 0,
    saving: false,
    conflict: false,
    error: null,
  }
  const submitted = {
    key: "current",
    request: run.config,
    run,
    pending: false,
    uncertain: false,
    error: null,
  }
  assert.equal(
    draftAcceptance(draft, submitted, "current", report).validation_run_id,
    run.run_id
  )
  assert.throws(
    () =>
      draftAcceptance(
        { ...draft, text: "new invalid yaml(" },
        submitted,
        "current",
        report
      ),
    /Revalidate/
  )
  assert.throws(
    () => draftAcceptance(draft, submitted, "changed profile", report),
    /Revalidate/
  )
  assert.throws(
    () =>
      draftAcceptance(
        { ...draft, saved: { ...draft.saved, draft_version: 2 } },
        submitted,
        "current",
        report
      ),
    /another saved draft/
  )
  assert.throws(
    () =>
      draftAcceptance(
        draft,
        submitted,
        "current",
        fixture("ValidationReport", "inspection-report")
      ),
    /Revalidate/
  )
  assert.throws(
    () =>
      draftAcceptance(
        draft,
        submitted,
        "current",
        fixture("ValidationReport", "invalid-completed-report")
      ),
    /Revalidate/
  )
})
