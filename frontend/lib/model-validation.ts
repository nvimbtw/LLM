import {
  architectureRequest,
  type ArchitectureRequest,
} from "./architecture/http"
import { decodeArchitecture, identityDigest } from "./architecture/decoder"
import type {
  ReceiptIdentity,
  ValidationReceipt,
  ValidationRun,
  ValidationReport,
} from "./architecture/contracts.generated"

/** Requires a report to describe the same immutable request/profile/inputs, without recomputing engine-owned resolved parameter or dependency identities. */
export function requireValidationReport(
  run: ValidationRun,
  report: ValidationReport
): void {
  const request = run.config
  if (report.run_id !== run.run_id)
    throw new Error("Validation report belongs to another execution.")
  if (
    (request.mode === "inspect" &&
      (!["inspection_only", "invalid"].includes(report.verdict) ||
        report.receipt)) ||
    (request.mode === "full" && report.verdict === "inspection_only")
  )
    throw new Error("Validation report differs from the requested operation.")
  if (!report.receipt) return
  const identity = report.receipt.identity
  const requestedProfile = request.profile!
  if (
    identityDigest(identity.subject) !== identityDigest(request.subject) ||
    identityDigest({ ...identity.profile, export_guards: [] }) !==
      identityDigest({ ...requestedProfile, export_guards: [] }) ||
    (Boolean(requestedProfile.export_guards?.length) &&
      identityDigest(identity.profile.export_guards) !==
        identityDigest(requestedProfile.export_guards)) ||
    identityDigest(identity.tokenizer ?? null) !==
      identityDigest(request.tokenizer ?? null) ||
    (request.context_window != null &&
      request.context_window !== identity.context_window) ||
    identity.input_profile_digest !==
      identityDigest({
        tensor_inputs: request.tensor_inputs ?? {},
        domains: request.profile!.domains,
      })
  )
    throw new Error(
      "Validation report belongs to a different source or input profile."
    )
}
/** Reads a durable report without changing the queued execution's lifetime. */
export function getModelValidation(id: string, request: ArchitectureRequest) {
  return architectureRequest(
    `/validations/${encodeURIComponent(id)}`,
    "ValidationReport",
    request
  )
}

/** Reads the authoritative validation queue envelope with captured session identity, independently from any published report. */
export function getModelValidationRun(
  id: string,
  request: ArchitectureRequest
) {
  return architectureRequest(
    `/runs/${encodeURIComponent(id)}`,
    "ValidationRun",
    request
  )
}

/** Explicitly requests a new attempt over the original immutable validation snapshot; observer cleanup never invokes this operation. */
export function retryModelValidation(id: string, request: ArchitectureRequest) {
  return architectureRequest(
    `/runs/${encodeURIComponent(id)}/retry`,
    "ValidationRun",
    request,
    { method: "POST" }
  )
}
/** Reads the selected build's full tokenizer fingerprint and vocabulary through Rust. */
export async function getArchitectureTokenizerBinding(
  id: string,
  request: ArchitectureRequest,
  reserved: readonly string[] = []
) {
  const query = new URLSearchParams()
  reserved.forEach((name) => query.append("reserved", name))
  const binding = await architectureRequest(
    `/tokenizers/${encodeURIComponent(id)}/architecture-binding${reserved.length ? `?${query}` : ""}`,
    "FinalTokenizer",
    request
  )
  if (
    binding.tokenizer_id !== id ||
    reserved.some((name) => !(name in (binding.reserved_token_ids ?? {})))
  )
    throw new Error(
      "Tokenizer binding differs from the selected vocabulary or reserved names."
    )
  return binding
}
/** Requires exact current receipt identity; inspection and pending bindings never authorize acceptance. */
export function requireCurrentReceipt(
  receipt: ValidationReceipt,
  expected: ReceiptIdentity
): void {
  receipt = decodeArchitecture("ValidationReceipt", receipt)
  const current = decodeArchitecture("ReceiptIdentity", expected)
  if (
    receipt.verdict !== "valid" ||
    current.tokenizer?.kind === "pending" ||
    receipt.identity_hash !== identityDigest(current)
  )
    throw new Error("Validation is stale for this configuration.")
}
