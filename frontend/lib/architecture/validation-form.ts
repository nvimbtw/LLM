import type {
  BlockDeclaration,
  DraftSubject,
  FinalTokenizer,
  Recipe,
  RuntimeProfile,
  TensorPort,
  TensorInputSpec,
  ValidationRequest,
} from "./contracts.generated"
import { decodeArchitecture } from "./decoder"
import { parameterError, type ParameterValue } from "./parameters"
import type { DraftSnapshot } from "./draft"
import type { ValidationSubmissionState } from "./validation-submission"
import type { AcceptRevision, ValidationReport } from "./contracts.generated"
import { requireValidationReport } from "../model-validation"

export type ValidationDeclaration = BlockDeclaration | Recipe
export const MAX_TENSOR_SHAPE_CHARACTERS = 256
export interface ValidationFormValues {
  parameters: Record<string, ParameterValue | undefined>
  profile: RuntimeProfile
  shapes: Record<string, string>
}

/** Suggests literals and explicit profile batch/sequence values, leaving unknown symbolic dimensions empty instead of inventing a shape. */
export function suggestedInputShape(
  port: TensorPort,
  profile: RuntimeProfile
): string {
  return port.shape
    .map((dimension) =>
      typeof dimension === "number"
        ? dimension
        : dimension === "batch"
          ? profile.domains[0].batch_min
          : dimension === "sequence"
            ? profile.domains[0].sequence_min
            : ""
    )
    .join(", ")
}

/** Parses bounded dimension text into the wire's rank, dtype and element limits before any queue request is built. */
function configuredTensorInput(
  port: TensorPort,
  profile: RuntimeProfile,
  shapeText: string
): TensorInputSpec {
  if (shapeText.length > MAX_TENSOR_SHAPE_CHARACTERS)
    throw new Error(
      `Tensor shape must be at most ${MAX_TENSOR_SHAPE_CHARACTERS} characters.`
    )
  return decodeArchitecture("TensorInputSpec", {
    shape: shapeText
      .split(",")
      .map((part) => (part.trim() === "" ? Number.NaN : Number(part.trim()))),
    dtype:
      port.dtype === "floating"
        ? ({ fp32: "float32", fp16: "float16", bf16: "bfloat16" } as const)[
            profile.precision
          ]
        : port.dtype,
  })
}

/** Interprets one complete form snapshot into bounded queue inputs without source parsing, vocabulary guesses or recipe-context calculations. */
export function configuredDraftValidation(
  subject: DraftSubject,
  declaration: ValidationDeclaration,
  form: ValidationFormValues,
  tokenizer: FinalTokenizer | null
): Omit<ValidationRequest, "request_id"> {
  for (const [name, item] of Object.entries(declaration.parameters)) {
    const value = form.parameters[name]
    const invalid =
      value === undefined
        ? item.required
          ? "Required parameter."
          : null
        : parameterError(item, value)
    if (invalid) throw new Error(`${name}: ${invalid}`)
  }
  const profile = decodeArchitecture("RuntimeProfile", form.profile)
  const causal =
    "interface" in declaration && declaration.interface === "causal_lm"
  if (causal && !tokenizer)
    throw new Error("Choose a built vocabulary before full validation.")
  const tensor_inputs = causal
    ? {}
    : Object.fromEntries(
        Object.entries(declaration.inputs).map(([name, port]) => [
          name,
          configuredTensorInput(
            port,
            profile,
            form.shapes[name] ?? suggestedInputShape(port, profile)
          ),
        ])
      )
  return {
    schema_version: 1,
    mode: "full",
    subject,
    parameters: Object.fromEntries(
      Object.entries(form.parameters).filter(([, value]) => value !== undefined)
    ) as Record<string, ParameterValue>,
    profile,
    tensor_inputs,
    tokenizer,
  }
}

/** Builds acceptance only for unchanged saved bytes and the current completed full configuration; receipt contents remain engine-owned evidence. */
export function draftAcceptance(
  draft: DraftSnapshot,
  submitted: Readonly<ValidationSubmissionState>,
  currentKey: string,
  report: ValidationReport
): AcceptRevision {
  const run = submitted.run
  if (
    !run ||
    submitted.pending ||
    submitted.key !== currentKey ||
    draft.saving ||
    draft.conflict ||
    draft.text !== draft.savedText ||
    run.config.mode !== "full" ||
    run.config.subject.kind !== "draft" ||
    report.verdict !== "valid" ||
    !report.receipt
  )
    throw new Error(
      "Revalidate the current saved configuration before acceptance."
    )
  const subject = run.config.subject
  if (
    subject.resource_id !== draft.saved.resource_id ||
    subject.draft_version !== draft.saved.draft_version ||
    subject.content_hash !== draft.saved.content_hash
  )
    throw new Error("Validation belongs to another saved draft.")
  requireValidationReport(run, report)
  return {
    expected_draft_version: subject.draft_version,
    expected_content_hash: subject.content_hash,
    expected_dependency_digest: report.receipt.identity.dependency_digest,
    validation_run_id: run.run_id,
  }
}
