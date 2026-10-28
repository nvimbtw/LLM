import type { ArchitectureContracts } from "./contracts.generated"
import { identityDigest, type JsonRecord } from "./decoder"
import { parameterError } from "./parameters"
import { sha256 } from "@noble/hashes/sha256"
import { bytesToHex } from "@noble/hashes/utils"
/** Rejects duplicate identities at the interpreted network boundary. */
function unique(values: unknown[]): void {
  if (
    new Set(values.map((value) => JSON.stringify(value))).size !== values.length
  )
    throw new Error("Duplicate contract identity")
}
/** Checks model-level invariants that JSON Schema cannot describe, after nested normalization. */
export function checkSemantics(name: string, record: JsonRecord): void {
  switch (name) {
    case "ParameterDeclaration":
      checkParameterDeclaration(record)
      break
    case "Capability":
      if (record.supported ? !record.evidence_hash : !record.reason)
        throw new Error("Capability needs evidence or an unavailable reason")
      break
    case "ValidationRequest":
      if (record.mode === "full" && !record.profile)
        throw new Error("Full validation needs a profile")
      if (
        record.mode === "inspect" &&
        (record.profile ||
          record.tokenizer ||
          record.context_window ||
          Object.keys(record.parameters as object).length ||
          Object.keys(record.tensor_inputs as object).length)
      )
        throw new Error("Inspection cannot execute configuration")
      break
    case "TensorInputSpec":
      if (
        (record.shape as number[]).reduce(
          (size, dimension) => size * dimension,
          1
        ) > 16_777_216
      )
        throw new Error("Tensor element budget exceeded")
      break
    case "SourceLocation":
    case "RetainedSource":
    case "SourceManifestEntry": {
      const path = String(record.file ?? record.path)
      if (
        path.startsWith("/") ||
        path.includes("\\") ||
        path.split("/").some((part) => ["", ".", ".."].includes(part))
      )
        throw new Error("Source path must be package relative")
      if (
        name === "RetainedSource" &&
        bytesToHex(sha256(new TextEncoder().encode(String(record.source)))) !==
          record.content_hash
      )
        throw new Error("Retained source hash mismatch")
      break
    }
    case "ValidationReceipt":
      checkValidationReceipt(record)
      break
    case "ValidationReport":
      checkValidationReport(record)
      break
    case "GraphFragment":
      checkGraphFragment(record)
      break
    case "ValidationRun":
      checkValidationRun(record)
      break
    case "ValidationEvent":
      if (
        (record.type === "validation_stage" && !record.stage) ||
        (record.type === "done" && !record.verdict) ||
        (record.type === "error" && !record.diagnostic)
      )
        throw new Error("Missing validation event payload")
      break
    case "EnqueueEvidence":
      if (
        ((record.tokenizer as JsonRecord).kind === "pending" ||
          !record.exact_receipt_id) &&
        !record.final_preflight_required
      )
        throw new Error("Final preflight required")
      break
    case "TrainingArchitecturePreview":
      checkTrainingPreview(record)
      break
    case "BlockDeclaration":
      unique(record.capabilities as unknown[])
      break
    case "FinalTokenizer":
      checkFinalTokenizer(record)
      break
    case "ShapeDomain":
      if (
        Number(record.batch_min) > Number(record.batch_max) ||
        Number(record.sequence_min) > Number(record.sequence_max) ||
        Number(record.past_min) > Number(record.past_max)
      )
        throw new Error("Inverted shape domain")
      unique(record.signatures as unknown[])
      break
    case "RuntimeProfile":
      unique(record.modes as unknown[])
      break
    case "DefinitionPackage":
      unique(record.optimizer_parameter_order as unknown[])
      break
    case "AttentionSelection":
      if (record.point_id != null && record.layer != null)
        throw new Error("Select an attention point or a legacy layer")
      break
    case "ArchitectureSchema":
      unique(record.available_devices as unknown[])
      if (
        !(record.available_devices as string[]).includes(
          (record.default_profile as JsonRecord).device as string
        )
      )
        throw new Error("Prospective profile selects an unavailable device")
      break
    case "InspectionPointView":
      if (
        Boolean(record.validated_sequence_max) !==
          (record.evidence_hash != null) ||
        (record.evidence_hash != null && record.heads == null)
      )
        throw new Error("Attention point needs complete runtime evidence")
      break
    case "InspectionCatalog":
      checkInspectionCatalog(record)
      break
    case "AttentionMatrix":
      checkAttentionMatrix(record)
      break
  }
  for (const field of ["source", "yaml"])
    if (
      typeof record[field] === "string" &&
      new TextEncoder().encode(record[field]).length > 262144
    )
      throw new Error("Source byte budget exceeded")
}

/** Checks prospective training profile coherence without treating metadata as numerical or queue authorization. */
function checkTrainingPreview(record: JsonRecord): void {
  const item = record as ArchitectureContracts["TrainingArchitecturePreview"]
  const domains = item.profile.domains
  if (
    !item.final_preflight_required ||
    item.architecture.kind !== "architecture" ||
    domains.length !== 1 ||
    !["train", "eval"].every((mode) =>
      item.profile.modes.includes(mode as "train" | "eval")
    )
  )
    throw new Error(
      "Training preview requires one causal recipe domain and both train/eval modes"
    )
  const domain = domains[0]
  if (
    domain.batch_min !== 1 ||
    domain.sequence_min !== 1 ||
    domain.sequence_max !== item.context_window ||
    domain.signatures.length !== 1 ||
    domain.signatures[0] !== "full_prefix" ||
    domain.past_min ||
    domain.past_max ||
    item.profile.token_dtype !== "int64"
  )
    throw new Error(
      "Training preview differs from canonical full-prefix input bounds"
    )
  unique(item.reserved_token_names ?? [])
}

/** Keeps declared candidates, measured bounds and actual legacy metadata consistent with the resident model identity. */
function checkInspectionCatalog(record: JsonRecord): void {
  const item = record as ArchitectureContracts["InspectionCatalog"]
  const points = item.points ?? []
  const identity = [item.definition_id, item.profile_digest, item.runtime_id]
  unique(points.map((point) => point.id))
  if (
    (item.legacy_layers == null) !== (item.legacy_heads == null) ||
    (item.context_window == null) !== (item.max_tokens == null) ||
    (item.legacy_layers != null && item.context_window == null) ||
    (item.max_tokens != null &&
      item.context_window != null &&
      item.max_tokens > item.context_window) ||
    (identity.some((value) => value != null) &&
      identity.some((value) => value == null)) ||
    (points.length > 0 &&
      (identity.some((value) => value == null) ||
        item.context_window == null)) ||
    points.some(
      (point) => (point.validated_sequence_max ?? 0) > (item.max_tokens ?? 0)
    ) ||
    (!points.length && item.legacy_layers == null && !item.unavailable_reason)
  )
    throw new Error("Attention metadata has inconsistent identity or bounds")
}

/** Validates real matrix axes, selected head/layer, base64 byte length and complete modular observation evidence. */
function checkAttentionMatrix(record: JsonRecord): void {
  const item = record as ArchitectureContracts["AttentionMatrix"]
  const size = item.tokens.length
  const proof = [
    item.definition_id,
    item.profile_digest,
    item.runtime_id,
    item.evidence_hash,
  ]
  if (
    item.shape[0] !== size ||
    item.shape[1] !== size ||
    size > item.max_tokens ||
    item.head >= item.heads ||
    (item.layer != null &&
      (item.layers == null || item.layer >= item.layers)) ||
    item.weights.length % 4 !== 0 ||
    atob(item.weights).length !== size * size * 4
  )
    throw new Error("Attention matrix axes, selection or bytes are invalid")
  const bytes = Uint8Array.from(atob(item.weights), (value) =>
    value.charCodeAt(0)
  )
  const view = new DataView(bytes.buffer)
  for (let index = 0; index < bytes.byteLength; index += 4) {
    const value = view.getFloat32(index, true)
    if (!Number.isFinite(value) || value < 0 || value > 1.00001)
      throw new Error("Attention matrix contains invalid probabilities")
  }
  if (item.point_id != null) {
    if (
      proof.some((value) => value == null) ||
      (item.validated_sequence_max ?? 0) < size ||
      (item.validated_sequence_max ?? 0) > item.max_tokens ||
      item.heads > 256
    )
      throw new Error("Modular attention needs complete runtime evidence")
  } else if (
    proof.some((value) => value != null) ||
    item.validated_sequence_max ||
    item.layer == null ||
    item.layers == null
  )
    throw new Error("Legacy attention identity is invalid")
}

/** Validates ParameterDeclaration semantic identity and consistency after structural decoding. */
function checkParameterDeclaration(record: JsonRecord): void {
  const item = record as ArchitectureContracts["ParameterDeclaration"]
  if (
    item.minimum != null &&
    item.maximum != null &&
    item.minimum > item.maximum
  )
    throw new Error("Inverted parameter bounds")
  if (item.required && item.default != null)
    throw new Error("Required parameter has a default")
  if ((item.type === "enum") !== Boolean(item.choices?.length))
    throw new Error("Invalid enum choices")
  unique(item.choices ?? [])
  if (item.default != null && parameterError(item, item.default))
    throw new Error("Invalid parameter default")
}

/** Validates ValidationReceipt semantic identity and consistency after structural decoding. */
function checkValidationReceipt(record: JsonRecord): void {
  const item = record as ArchitectureContracts["ValidationReceipt"]
  if (item.identity_hash !== identityDigest(item.identity))
    throw new Error("Receipt identity mismatch")
  if (
    item.verdict === "valid" &&
    (!item.graph_manifest_hash ||
      item.identity.tokenizer?.kind === "pending" ||
      !["train", "eval"].every((mode) =>
        item.identity.profile.modes.includes(mode as "train" | "eval")
      ))
  )
    throw new Error("Incomplete full validation receipt")
  unique((item.capabilities ?? []).map((entry) => entry.name))
}

/** Validates ValidationReport semantic identity and consistency after structural decoding. */
function checkValidationReport(record: JsonRecord): void {
  const item = record as ArchitectureContracts["ValidationReport"]
  if (item.receipt && item.receipt.verdict !== item.verdict)
    throw new Error("Report verdict mismatch")
  if (item.verdict === "inspection_only" && !item.declaration)
    throw new Error("Inspection requires declarations")
  if (
    ["valid", "provisional"].includes(item.verdict) &&
    (!item.receipt ||
      (item.verdict === "valid" &&
        ["capture", "numerical"].some(
          (stage) =>
            !item.stages.some(
              (entry) => entry.stage === stage && entry.outcome === "passed"
            )
        )) ||
      item.stages.some((stage) =>
        ["pending", "running", "failed"].includes(stage.outcome)
      ))
  )
    throw new Error("Incomplete validation report")
  unique(item.stages.map((stage) => stage.stage))
  unique((item.capabilities ?? []).map((entry) => entry.name))
}

/** Validates GraphFragment semantic identity and consistency after structural decoding. */
function checkGraphFragment(record: JsonRecord): void {
  const item = record as ArchitectureContracts["GraphFragment"]
  unique(item.nodes.map((node) => node.id))
  if (item.nodes.length > (item.query.limit ?? 256))
    throw new Error("Graph exceeds requested page size")
}

/** Validates ValidationRun semantic identity and consistency after structural decoding. */
function checkValidationRun(record: JsonRecord): void {
  const item = record as ArchitectureContracts["ValidationRun"]
  if (
    item.output.type !== "validation" ||
    item.output.id !== item.run_id ||
    (item.ref_id && item.ref_id !== item.run_id)
  )
    throw new Error("Validation run output mismatch")
}

/** Validates FinalTokenizer semantic identity and consistency after structural decoding. */
function checkFinalTokenizer(record: JsonRecord): void {
  const ids = Object.values(record.reserved_token_ids as Record<string, number>)
  unique(ids)
  if (ids.some((id) => id >= Number(record.vocabulary_size)))
    throw new Error("Token ID outside vocabulary")
}
