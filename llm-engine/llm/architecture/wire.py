"""Versioned authoring, validation, graph and submission wire contracts.

Rust authorizes these engine-owned payloads; storage and execution adapters use
these records without introducing a second queue, catalog or graph interpreter.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, JsonValue, StrictBool, StrictInt, StringConstraints, field_validator, model_validator

from llm.architecture.spec import (
    MAX_CATALOG_PAGE, MAX_GRAPH_PAGE_NODES, MAX_PARAMETERS, MAX_SOURCE_BYTES, MAX_ATTENTION_HEADS, MAX_ATTENTION_SEQUENCE,
    MAX_YAML_BYTES, BlockDeclaration, Capability, Contract, Identifier,
    DraftSubject, NonnegativeInt, ParameterDeclaration, PositiveInt, ReceiptIdentity,
    Recipe, RevisionRef, RuntimeProfile, Scalar, Sha256, Subject, TokenizerBinding, FinalTokenizer,
    ValidationReceipt, checked_relative_path,
)

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Description = Annotated[str, Field(max_length=2000)]


def checked_text_bytes(value: str, limit: int) -> str:
    """Apply UTF-8 byte budgets to exact saved text without parsing or executing it."""
    if len(value.encode("utf-8")) > limit:
        raise ValueError("source text byte budget exceeded")
    return value


class BlockCreate(Contract):
    """Create a named resource with arbitrary draft bytes, including malformed Python."""

    name: Label
    description: Description = ""
    source: str

    @field_validator("source")
    @classmethod
    def bounded_source(cls, value: str) -> str:
        """Bound stored Python bytes without requiring valid syntax or importing code."""
        return checked_text_bytes(value, MAX_SOURCE_BYTES)


class ArchitectureCreate(Contract):
    """Create a named architecture whose initial editable YAML may be invalid."""

    name: Label
    description: Description = ""
    yaml: str

    @field_validator("yaml")
    @classmethod
    def bounded_yaml(cls, value: str) -> str:
        """Bound stored YAML bytes without claiming semantic or capture validity."""
        return checked_text_bytes(value, MAX_YAML_BYTES)


class MetadataPatch(Contract):
    """Optimistically change metadata; omission preserves and nullable description clears."""

    expected_metadata_version: PositiveInt
    name: Label | None = None
    description: Description | None = None
    archived: StrictBool | None = None


class BlockDraftSave(Contract):
    """Save exact Python bytes only if the current editable draft version matches."""

    expected_version: PositiveInt
    source: str

    @field_validator("source")
    @classmethod
    def bounded_source(cls, value: str) -> str:
        """Apply the same byte budget as creation without syntax validation."""
        return checked_text_bytes(value, MAX_SOURCE_BYTES)


class ArchitectureDraftSave(Contract):
    """Save exact YAML bytes only if the current editable draft version matches."""

    expected_version: PositiveInt
    yaml: str

    @field_validator("yaml")
    @classmethod
    def bounded_yaml(cls, value: str) -> str:
        """Apply the same byte budget as creation while preserving invalid editable text."""
        return checked_text_bytes(value, MAX_YAML_BYTES)


class DraftSaved(Contract):
    """Acknowledge saved bytes, not inspection, full validation or revision acceptance."""

    resource_id: UUID
    draft_version: PositiveInt
    content_hash: Sha256


class BlockDraftView(DraftSaved):
    """Read exact editable Python bytes independently of the last accepted revision."""

    source: Annotated[str, Field(max_length=MAX_SOURCE_BYTES)]

    @field_validator("source")
    @classmethod
    def bounded_source(cls, value: str) -> str:
        """Bound UTF-8 source while permitting syntactically invalid saved drafts."""
        return checked_text_bytes(value, MAX_SOURCE_BYTES)


class ArchitectureDraftView(DraftSaved):
    """Read exact editable YAML bytes without executing or validating its contents."""

    yaml: Annotated[str, Field(max_length=MAX_YAML_BYTES)]

    @field_validator("yaml")
    @classmethod
    def bounded_yaml(cls, value: str) -> str:
        """Bound UTF-8 YAML while retaining invalid drafts for repair."""
        return checked_text_bytes(value, MAX_YAML_BYTES)


class ArchitectureLimits(Contract):
    """Expose actual engine authoring ceilings as data; worker admission remains a separate execution boundary."""

    source_bytes: PositiveInt
    yaml_bytes: PositiveInt
    expanded_calls: PositiveInt
    nesting: PositiveInt


class ArchitectureSchema(Contract):
    """Supply the authoritative recipe schema and prospective runtime defaults without claiming validation or device admission."""

    schema_version: Literal[1]
    recipe: dict[str, JsonValue]
    limits: ArchitectureLimits
    binding_syntax: Literal["${parameter}"]
    default_profile: RuntimeProfile
    available_devices: Annotated[tuple[Literal["cpu", "cuda"], ...], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def available_default(self):
        """Require unique device names and a prospective default selecting an actually reported device."""
        if len(set(self.available_devices)) != len(self.available_devices) or self.default_profile.device not in self.available_devices:
            raise ValueError("prospective default profile requires an available device")
        return self


class RecipeDraftInspection(Contract):
    """Return safely parsed saved YAML declarations with exact draft identity, without imports, resolution, construction or evidence."""

    subject: DraftSubject
    recipe: Recipe
    verdict: Literal["inspection_only"] = "inspection_only"


class ResourceSummary(Contract):
    """List bounded catalog metadata without embedding administrative source or graphs."""

    resource_id: UUID
    kind: Literal["block", "architecture"]
    name: Label
    description: Description
    origin: Literal["builtin", "custom"]
    archived: StrictBool
    metadata_version: PositiveInt
    draft_version: PositiveInt
    accepted_revision: RevisionRef | None
    capabilities: Annotated[tuple[Capability, ...], Field(max_length=6)] = ()


class CatalogPage(Contract):
    """Page summary rows with one stable opaque cursor scoped to filters and visibility."""

    items: Annotated[tuple[ResourceSummary, ...], Field(max_length=MAX_CATALOG_PAGE)]
    next_cursor: Annotated[str, Field(max_length=2048)] | None = None


class ResourceCreated(Contract):
    """Return the new stable UUID after source bytes and first draft are durable."""

    resource_id: UUID


class ResourceDetail(Contract):
    """Expose catalog metadata and known declarations without loading editable source."""

    summary: ResourceSummary
    revision_count: NonnegativeInt
    declaration: BlockDeclaration | None = None
    parameters: Annotated[dict[Identifier, ParameterDeclaration], Field(max_length=MAX_PARAMETERS)] = Field(default_factory=dict)


class CatalogQuery(Contract):
    """Bound catalog reads independently of untrusted client-supplied paging values."""

    search: Annotated[str, Field(max_length=200)] = ""
    origin: Literal["builtin", "custom"] | None = None
    archived: StrictBool = False
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_CATALOG_PAGE)] = 50
    cursor: Annotated[str, Field(max_length=2048)] | None = None


class AcceptRevision(Contract):
    """Atomically accept exact current bytes and a matching fully bound validation receipt."""

    expected_draft_version: PositiveInt
    validation_run_id: UUID
    expected_content_hash: Sha256
    expected_dependency_digest: Sha256


class AcceptedRevision(Contract):
    """Return immutable revision/evidence identity without tying retention to job history."""

    reference: RevisionRef
    revision_number: PositiveInt
    acceptance_receipt_id: UUID
    acceptance_identity_hash: Sha256


class RevisionPage(Contract):
    """Page retained immutable revision/acceptance summaries without copying source."""

    items: Annotated[tuple[AcceptedRevision, ...], Field(max_length=MAX_CATALOG_PAGE)]
    next_cursor: Annotated[str, Field(max_length=2048)] | None = None


class BlockRevisionView(Contract):
    """Read immutable block source and its accepted constructor/port declarations."""

    accepted: AcceptedRevision
    source: Annotated[str, Field(max_length=MAX_SOURCE_BYTES)]
    declaration: BlockDeclaration
    sdk_version: Literal[1]

    @field_validator("source")
    @classmethod
    def bounded_source(cls, value: str) -> str:
        """Keep revision reads under the same exact-byte source budget as draft saves."""
        return checked_text_bytes(value, MAX_SOURCE_BYTES)


class ArchitectureRevisionView(Contract):
    """Read immutable YAML, its typed composition and exact pinned dependency manifest."""

    accepted: AcceptedRevision
    yaml: Annotated[str, Field(max_length=MAX_YAML_BYTES)]
    recipe: Recipe
    dependencies: Annotated[tuple[RevisionRef, ...], Field(max_length=128)]
    dependency_digest: Sha256

    @field_validator("yaml")
    @classmethod
    def bounded_yaml(cls, value: str) -> str:
        """Keep immutable recipe source inside the declared UTF-8 transport budget."""
        return checked_text_bytes(value, MAX_YAML_BYTES)


class TensorInputSpec(Contract):
    """Generate bounded synthetic block inputs in workers rather than upload tensor blobs."""

    dtype: Literal["int64", "int32", "bool", "float32", "float16", "bfloat16"]
    shape: Annotated[tuple[PositiveInt, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def bounded_elements(self) -> TensorInputSpec:
        """Reject excessive effective element counts before synthetic allocation."""
        elements = 1
        for dimension in self.shape:
            elements *= dimension
            if elements > 16_777_216:
                raise ValueError("synthetic tensor element budget exceeded")
        return self


class ValidationRequest(Contract):
    """Queue immutable source inspection or configuration-specific full validation."""

    schema_version: Literal[1]
    request_id: UUID
    mode: Literal["inspect", "full"]
    subject: Subject
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)] = Field(default_factory=dict)
    tensor_inputs: Annotated[dict[Identifier, TensorInputSpec], Field(max_length=32)] = Field(default_factory=dict)
    tokenizer: TokenizerBinding | None = None
    profile: RuntimeProfile | None = None
    context_window: PositiveInt | None = None

    @model_validator(mode="after")
    def require_full_profile(self) -> ValidationRequest:
        """Inspection needs no constructor/profile inputs; full checks require explicit bounds."""
        if self.mode == "full" and self.profile is None:
            raise ValueError("full validation requires runtime and input profile")
        if self.mode == "inspect" and (self.profile is not None or self.parameters or self.tensor_inputs or self.tokenizer is not None or self.context_window is not None):
            raise ValueError("inspection-only request must not imply construction or execution")
        return self


class SourceLocation(Contract):
    """Locate packaged source by immutable revision and relative file, not private host paths."""

    revision_id: UUID
    file: Annotated[str, Field(min_length=1, max_length=512)]
    line: PositiveInt | None = None
    column: PositiveInt | None = None

    @field_validator("file")
    @classmethod
    def relative_file(cls, value: str) -> str:
        """Reject filesystem traversal before emitting or resolving source locations."""
        return checked_relative_path(value)


class RetainedSource(Contract):
    """Expose exact retained text independently of mutable catalog rows or installed implementations."""

    revision_id: UUID
    definition_id: Sha256 | None = None
    content_hash: Sha256
    file: Annotated[str, Field(min_length=1, max_length=512)]
    language: Literal["python", "yaml", "json"]
    source: Annotated[str, Field(max_length=MAX_SOURCE_BYTES)]

    @model_validator(mode="after")
    def exact_bytes(self) -> RetainedSource:
        """Require normalized display paths, bounded UTF-8 and an exact raw-byte source digest."""
        checked_relative_path(self.file)
        checked_text_bytes(self.source, MAX_SOURCE_BYTES)
        if hashlib.sha256(self.source.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("retained source hash mismatch")
        return self


class Diagnostic(Contract):
    """Return bounded actionable failures without tensor values, arbitrary objects or raw logs."""

    code: Identifier
    message: Annotated[str, Field(min_length=1, max_length=2000)]
    phase: Literal["save", "inspection", "resolution", "construction", "capture", "numerical", "admission", "execution", "publication"]
    resource_id: UUID | None = None
    revision_id: UUID | None = None
    yaml_path: Annotated[str, Field(max_length=512)] | None = None
    instance_path: Annotated[str, Field(max_length=512)] | None = None
    source: SourceLocation | None = None


class ErrorResponse(Contract):
    """Forward structured engine errors with unchanged HTTP status through Rust."""

    error: Diagnostic
    current_version: PositiveInt | None = None


class StageOutcome(Contract):
    """Report finite validation stages independently of execution lifecycle status."""

    stage: Literal["inspection", "resolution", "construction", "capture", "numerical", "admission", "publication"]
    outcome: Literal["pending", "running", "passed", "failed", "skipped"]
    elapsed_ms: NonnegativeInt | None = None


class ValidationReport(Contract):
    """A completed-invalid/inspection report is distinct from a successful full receipt."""

    schema_version: Literal[1]
    run_id: UUID
    verdict: Literal["inspection_only", "invalid", "valid", "provisional"]
    declaration: BlockDeclaration | None = None
    receipt: ValidationReceipt | None = None
    stages: Annotated[tuple[StageOutcome, ...], Field(max_length=16)]
    diagnostics: Annotated[tuple[Diagnostic, ...], Field(max_length=64)]
    unique_parameter_count: NonnegativeInt | None = None
    capabilities: Annotated[tuple[Capability, ...], Field(max_length=6)] = ()

    @model_validator(mode="after")
    def matching_verdict(self) -> ValidationReport:
        """Prevent reports from labeling absent, provisional or inspection evidence as valid."""
        if self.receipt is not None and self.receipt.verdict != self.verdict:
            raise ValueError("report verdict differs from receipt")
        if self.verdict in {"valid", "provisional"} and self.receipt is None:
            raise ValueError("validated report needs exact receipt identity")
        if self.verdict == "inspection_only" and self.declaration is None:
            raise ValueError("inspection report requires declaration metadata")
        names = [stage.stage for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("duplicate validation stage")
        if self.verdict == "valid":
            outcomes = {stage.stage: stage.outcome for stage in self.stages}
            if any(outcomes.get(stage) != "passed" for stage in ("capture", "numerical")):
                raise ValueError("successful report requires capture and numerical checks")
        if self.verdict in {"valid", "provisional"}:
            if any(stage.outcome in {"pending", "running", "failed"} for stage in self.stages):
                raise ValueError("successful report contains unfinished or failed stage")
        if len({item.name for item in self.capabilities}) != len(self.capabilities):
            raise ValueError("duplicate reported capability")
        return self


class GraphQuery(Contract):
    """Read only published bounded graph scopes; a query never launches a capture."""

    identity_hash: Sha256
    view: Literal["architecture", "operations"]
    mode: Literal["train", "eval"]
    signature: Literal["full_prefix", "prefill", "decode"]
    scope: Annotated[str, Field(min_length=1, max_length=512)] = "root"
    cursor: Annotated[str, Field(max_length=2048)] | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=MAX_GRAPH_PAGE_NODES)] = MAX_GRAPH_PAGE_NODES


class GraphCursor(Contract):
    """Bind opaque cursor contents to immutable graph identity, mode, view and scope."""

    graph_hash: Sha256
    identity_hash: Sha256
    view: Literal["architecture", "operations"]
    mode: Literal["train", "eval"]
    signature: Literal["full_prefix", "prefill", "decode"]
    scope: Annotated[str, Field(min_length=1, max_length=512)]
    offset: NonnegativeInt

    def require_query(self, query: GraphQuery) -> None:
        """Refuse replaying a scope cursor against another graph profile or presentation."""
        fields = ("identity_hash", "view", "mode", "signature", "scope")
        if any(getattr(self, name) != getattr(query, name) for name in fields):
            raise ValueError("graph cursor belongs to another scope or profile")


class TensorSummary(Contract):
    """Show bounded shape/dtype metadata without embedding raw tensor contents."""

    shape: Annotated[tuple[NonnegativeInt | Identifier, ...], Field(max_length=8)]
    dtype: Annotated[str, Field(min_length=1, max_length=32)]


class GraphNode(Contract):
    """Separate computation occurrence paths from shared parameter owner identity."""

    id: Annotated[str, Field(min_length=1, max_length=512)]
    kind: Literal["input", "output", "module", "call", "loop", "operation"]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    instance_path: Annotated[str, Field(max_length=512)] | None = None
    occurrence_path: Annotated[str, Field(min_length=1, max_length=512)]
    parameter_owner_ids: Annotated[tuple[Annotated[str, Field(min_length=1, max_length=512)], ...], Field(max_length=128)] = ()
    operation_target: Annotated[str, Field(max_length=256)] | None = None
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)] = Field(default_factory=dict)
    outputs: Annotated[dict[Identifier, TensorSummary], Field(max_length=32)] = Field(default_factory=dict)
    source: SourceLocation | None = None
    child_scope: Annotated[str, Field(max_length=512)] | None = None
    iterations: PositiveInt | None = None
    unique_parameter_count: NonnegativeInt | None = None


class GraphEdge(Contract):
    """Connect explicit named ports between graph nodes with no implicit merge semantics."""

    source: Annotated[str, Field(min_length=1, max_length=512)]
    source_port: Identifier
    target: Annotated[str, Field(min_length=1, max_length=512)]
    target_port: Identifier


class GraphFragment(Contract):
    """Return bounded immutable scopes with explicit deferred children and pagination."""

    schema_version: Literal[1]
    graph_hash: Sha256
    query: GraphQuery
    nodes: Annotated[tuple[GraphNode, ...], Field(max_length=MAX_GRAPH_PAGE_NODES)]
    edges: Annotated[tuple[GraphEdge, ...], Field(max_length=1024)]
    next_cursor: Annotated[str, Field(max_length=2048)] | None = None

    @model_validator(mode="after")
    def unique_nodes(self) -> GraphFragment:
        """Reject duplicate node IDs rather than silently merging computation occurrences."""
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("duplicate graph node ID")
        if len(self.nodes) > self.query.limit:
            raise ValueError("graph response exceeds requested page size")
        return self


class ArchitectureSelection(Contract):
    """Freeze submission selection; pending tokenizer requires final in-job preflight."""

    revision_id: UUID
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)]
    validation_run_id: UUID | None = None


class TrainingArchitectureRequest(Contract):
    """Read prospective selected-recipe bindings/profile without allocating a model, queue row or future tokenizer identity."""

    schema_version: Literal[1] = 1
    selection: ArchitectureSelection
    batch_size: Annotated[PositiveInt, Field(le=9007199254740991)]
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    seed: Annotated[NonnegativeInt, Field(le=9007199254740991)] = 0
    allow_tf32: StrictBool = True
    tokenizer_id: UUID | None = None


class TrainingArchitecturePreview(Contract):
    """Return resolved data-only configuration; final tokenization and admitted numerical preflight remain mandatory in the actual training job."""

    schema_version: Literal[1] = 1
    architecture: RevisionRef
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)]
    parameter_digest: Sha256
    dependency_digest: Sha256
    context_window: PositiveInt
    profile: RuntimeProfile
    tokenizer: FinalTokenizer | None = None
    reserved_token_names: Annotated[tuple[Identifier, ...], Field(max_length=128)] = ()
    final_preflight_required: StrictBool = True

    @model_validator(mode="after")
    def training_domain(self) -> TrainingArchitecturePreview:
        """Require a causal training-shaped prospective profile, never a contradictory or partial validation claim."""
        domains = self.profile.domains
        if not self.final_preflight_required or self.architecture.kind != "architecture" or len(domains) != 1 or set(self.profile.modes) != {"train", "eval"}:
            raise ValueError("training preview requires one causal recipe domain and both train/eval modes")
        domain = domains[0]
        if (domain.batch_min != 1 or domain.sequence_min != 1 or domain.sequence_max != self.context_window
                or domain.signatures != ("full_prefix",) or domain.past_min or domain.past_max or self.profile.token_dtype != "int64"):
            raise ValueError("training preview differs from canonical full-prefix input bounds")
        if len(set(self.reserved_token_names)) != len(self.reserved_token_names):
            raise ValueError("training preview has duplicate reserved token names")
        return self


class EnqueueEvidence(Contract):
    """Pin an atomic queued/waiting step before catalog edits, archive or history pruning."""

    architecture: RevisionRef
    dependency_digest: Sha256
    parameter_digest: Sha256
    acceptance_receipt_id: UUID
    tokenizer: TokenizerBinding
    exact_receipt_id: UUID | None
    final_preflight_required: StrictBool

    @model_validator(mode="after")
    def require_final_preflight(self) -> EnqueueEvidence:
        """Pending or absent exact profile evidence cannot permit immediate optimizer work."""
        if (self.tokenizer.kind == "pending" or self.exact_receipt_id is None) and not self.final_preflight_required:
            raise ValueError("unbound configuration requires in-job final preflight")
        return self


class AuthorizedVisibility(Contract):
    """Trusted Rust-to-engine scope; browser filters cannot grant administration visibility."""

    schema_version: Literal[1]
    actor_id: UUID
    scope: Literal["public", "administration"]


class RunOutput(Contract):
    """Preserve generic output typing while validation output identity equals its run UUID."""

    type: Literal["model", "tokenizer", "benchmark", "validation"]
    id: UUID | None


class ValidationRun(Contract):
    """Define the normal queue envelope without duplicating lifecycle in validation storage."""

    run_id: UUID
    kind: Literal["model_validation"]
    status: Literal["waiting", "queued", "running", "cancelling", "done", "failed", "cancelled"]
    label: Annotated[str, Field(max_length=120)]
    rank: StrictInt | float
    progress: Annotated[StrictInt | float, Field(ge=0, le=1)]
    stage: Annotated[str, Field(max_length=120)] | None
    output: RunOutput
    ref_id: UUID | None
    depends_on: UUID | None
    config: ValidationRequest
    result: ValidationReport | None
    error: Annotated[str, Field(max_length=2000)] | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    frames: NonnegativeInt = 0

    @model_validator(mode="after")
    def validation_output(self) -> ValidationRun:
        """Keep validation output/ref identity deterministic and distinct from a produced model."""
        if self.output.type != "validation" or self.output.id != self.run_id:
            raise ValueError("validation output must identify the same run")
        if self.ref_id is not None and self.ref_id != self.run_id:
            raise ValueError("validation ref_id must identify the same run")
        return self


class ValidationEvent(Contract):
    """Extend existing job events with bounded validation stages and terminal REST reconciliation."""

    schema_version: Literal[1]
    run_id: UUID
    sequence: NonnegativeInt
    type: Literal["validation_stage", "done", "error", "cancelled"]
    stage: StageOutcome | None = None
    verdict: Literal["inspection_only", "invalid", "valid", "provisional"] | None = None
    diagnostic: Diagnostic | None = None

    @model_validator(mode="after")
    def meaningful_payload(self) -> ValidationEvent:
        """Require the payload appropriate to a stage, terminal verdict or execution failure."""
        if self.type == "validation_stage" and self.stage is None:
            raise ValueError("validation stage event requires stage outcome")
        if self.type == "done" and self.verdict is None:
            raise ValueError("completed validation requires a verdict")
        if self.type == "error" and self.diagnostic is None:
            raise ValueError("validation execution error requires a diagnostic")
        return self


class ModelDefinitionSummary(Contract):
    """Read saved definition metadata without importing source or loading weights."""

    definition_id: Sha256
    architecture: RevisionRef
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)]
    context_window: PositiveInt
    vocabulary_size: PositiveInt
    tokenizer_fingerprint: Sha256
    dependencies: Annotated[tuple[RevisionRef, ...], Field(max_length=128)]
    capabilities: Annotated[tuple[Capability, ...], Field(max_length=6)]
    parameter_count: NonnegativeInt | None
    legacy_architecture: Annotated[dict[Literal["dimensions", "heads", "layers", "ff_multiplier", "ctx_window"], PositiveInt], Field(max_length=5)] | None
    validation_receipts: Annotated[tuple[ValidationReceipt, ...], Field(max_length=64)] = ()


class AttentionSelection(Contract):
    """Select an actual declared observation or a compatible legacy layer; one request never guesses between the two."""

    text: Annotated[str, Field(min_length=1, max_length=1048576)]
    layer: NonnegativeInt | None = None
    head: NonnegativeInt = 0
    point_id: Sha256 | None = None

    @model_validator(mode="after")
    def unambiguous_selection(self):
        """Retain the default legacy layer zero while rejecting a request that supplies both selection identities."""
        if self.point_id is not None and self.layer is not None:
            raise ValueError("select an attention point or a legacy layer, not both")
        return self


class InspectionPointView(Contract):
    """Expose declared view identity separately from worker-lifetime numerical evidence and its exact tested bound."""

    id: Sha256
    title: Annotated[str, Field(min_length=1, max_length=120)]
    instance_path: Annotated[str, Field(min_length=1, max_length=1024)]
    call_path: Annotated[tuple[NonnegativeInt, ...], Field(max_length=64)]
    heads: Annotated[PositiveInt, Field(le=MAX_ATTENTION_HEADS)] | None = None
    validated_sequence_max: Annotated[NonnegativeInt, Field(le=MAX_ATTENTION_SEQUENCE)] = 0
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def complete_runtime_evidence(self):
        """A declared point cannot claim a supported runtime bound without its observed head count and exact evidence hash."""
        if bool(self.validated_sequence_max) != (self.evidence_hash is not None) or self.evidence_hash is not None and self.heads is None:
            raise ValueError("validated attention points require heads, sequence bounds and matching evidence")
        return self


class InspectionCatalog(Contract):
    """Describe bounded model-local attention candidates; reading declarations alone never certifies their measurement path."""

    model_name: Annotated[str, Field(min_length=1, max_length=255)]
    definition_id: Sha256 | None = None
    profile_digest: Sha256 | None = None
    runtime_id: UUID | None = None
    context_window: PositiveInt | None = None
    max_tokens: Annotated[PositiveInt, Field(le=MAX_ATTENTION_SEQUENCE)] | None = None
    legacy_layers: PositiveInt | None = None
    legacy_heads: PositiveInt | None = None
    points: Annotated[tuple[InspectionPointView, ...], Field(max_length=128)] = ()
    unavailable_reason: Annotated[str, Field(min_length=1, max_length=2000)] | None = None

    @model_validator(mode="after")
    def coherent_catalog(self):
        """Keep runtime identity, actual legacy dimensions and unavailable metadata explicit and refuse duplicate/stale view bounds."""
        if (self.legacy_layers is None) != (self.legacy_heads is None):
            raise ValueError("legacy attention metadata requires both layers and heads")
        if (self.context_window is None) != (self.max_tokens is None):
            raise ValueError("attention context and token ceiling must be given together")
        if self.legacy_layers is not None and self.context_window is None:
            raise ValueError("legacy attention metadata requires actual context")
        if self.max_tokens is not None and self.max_tokens > self.context_window:
            raise ValueError("attention token ceiling exceeds trained context")
        if len({point.id for point in self.points}) != len(self.points):
            raise ValueError("attention point IDs must be unique")
        identity = (self.definition_id, self.profile_digest, self.runtime_id)
        if any(value is not None for value in identity) and any(value is None for value in identity):
            raise ValueError("attention runtime identity must be complete")
        if self.points and any(value is None for value in (self.definition_id, self.profile_digest, self.runtime_id, self.context_window)):
            raise ValueError("declared attention points require exact resident definition/profile/runtime identity")
        if any(point.validated_sequence_max > (self.max_tokens or 0) for point in self.points):
            raise ValueError("attention point proof exceeds the available token ceiling")
        if not self.points and self.legacy_layers is None and self.unavailable_reason is None:
            raise ValueError("unavailable attention requires an explicit reason")
        return self


class AttentionMatrix(Contract):
    """Return one bounded real fp32 little-endian head matrix, with actual axes and optional modular measurement identity."""

    model_name: Annotated[str, Field(min_length=1, max_length=255)]
    layer: NonnegativeInt | None = None
    head: NonnegativeInt
    layers: PositiveInt | None = None
    heads: PositiveInt
    max_tokens: Annotated[PositiveInt, Field(le=MAX_ATTENTION_SEQUENCE)]
    tokens: Annotated[tuple[Annotated[str, Field(max_length=8192)], ...], Field(min_length=1, max_length=MAX_ATTENTION_SEQUENCE)]
    shape: tuple[Annotated[PositiveInt, Field(le=MAX_ATTENTION_SEQUENCE)], Annotated[PositiveInt, Field(le=MAX_ATTENTION_SEQUENCE)]]
    dtype: Literal["float32"] = "float32"
    encoding: Literal["base64"] = "base64"
    weights: Annotated[str, Field(min_length=8, max_length=((MAX_ATTENTION_SEQUENCE**2 * 4 + 2) // 3) * 4, pattern=r"^[A-Za-z0-9+/]+={0,2}$")]
    point_id: Sha256 | None = None
    definition_id: Sha256 | None = None
    profile_digest: Sha256 | None = None
    runtime_id: UUID | None = None
    validated_sequence_max: Annotated[NonnegativeInt, Field(le=MAX_ATTENTION_SEQUENCE)] = 0
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def actual_matrix_identity(self):
        """Refuse inconsistent axes, resource ranges, byte lengths or a modular result without its exact measurement proof."""
        size = len(self.tokens)
        if self.shape != (size, size) or size > self.max_tokens or self.head >= self.heads:
            raise ValueError("attention matrix axes or head are outside actual bounds")
        if self.layer is not None and (self.layers is None or self.layer >= self.layers):
            raise ValueError("attention layer is outside actual legacy bounds")
        try:
            raw = base64.b64decode(self.weights, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("attention weights are not valid base64") from error
        if len(raw) != size * size * 4:
            raise ValueError("attention matrix byte length differs from its axes")
        if any(not math.isfinite(value) or not 0 <= value <= 1.00001 for (value,) in struct.iter_unpack("<f", raw)):
            raise ValueError("attention matrix contains invalid probabilities")
        proof = (self.definition_id, self.profile_digest, self.runtime_id, self.evidence_hash)
        if self.point_id is not None:
            if any(value is None for value in proof) or not size <= self.validated_sequence_max <= self.max_tokens or self.heads > MAX_ATTENTION_HEADS:
                raise ValueError("modular attention requires complete bounded runtime evidence")
        else:
            if any(value is not None for value in proof) or self.validated_sequence_max:
                raise ValueError("legacy attention cannot claim modular runtime evidence")
            if self.layer is None or self.layers is None:
                raise ValueError("legacy attention requires actual layer identity")
        return self


CONTRACT_MODELS = {model.__name__: model for model in (
    BlockCreate, ArchitectureCreate, MetadataPatch, BlockDraftSave,
    ArchitectureDraftSave, DraftSaved, BlockDraftView, ArchitectureDraftView,
    ArchitectureLimits, ArchitectureSchema, RecipeDraftInspection,
    ResourceSummary, CatalogPage, ResourceCreated, ResourceDetail, CatalogQuery,
    AcceptRevision, AcceptedRevision, RevisionPage, BlockRevisionView,
    ArchitectureRevisionView, TensorInputSpec, ValidationRequest,
    SourceLocation, RetainedSource, Diagnostic, ErrorResponse, StageOutcome, ValidationReport,
    GraphQuery, GraphCursor, TensorSummary, GraphNode, GraphEdge, GraphFragment,
    ArchitectureSelection, TrainingArchitectureRequest, TrainingArchitecturePreview, EnqueueEvidence, AuthorizedVisibility, RunOutput,
    ValidationRun, ValidationEvent, ModelDefinitionSummary,
    AttentionSelection, InspectionPointView, InspectionCatalog, AttentionMatrix,
)}
