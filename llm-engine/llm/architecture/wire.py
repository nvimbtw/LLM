"""Versioned authoring, validation, graph and submission wire contracts.

Rust authorizes these engine-owned payloads; storage and execution adapters use
these records without introducing a second queue, catalog or graph interpreter.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictBool, StrictInt, StringConstraints, field_validator, model_validator

from llm.architecture.spec import (
    MAX_CATALOG_PAGE, MAX_GRAPH_PAGE_NODES, MAX_PARAMETERS, MAX_SOURCE_BYTES,
    MAX_YAML_BYTES, BlockDeclaration, Capability, Contract, Identifier,
    NonnegativeInt, ParameterDeclaration, PositiveInt, ReceiptIdentity,
    Recipe, RevisionRef, RuntimeProfile, Scalar, Sha256, Subject, TokenizerBinding,
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
        if self.verdict in {"valid", "provisional"}:
            outcomes = {stage.stage: stage.outcome for stage in self.stages}
            if any(outcomes.get(stage) != "passed" for stage in ("capture", "numerical")):
                raise ValueError("successful report requires capture and numerical checks")
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


CONTRACT_MODELS = {model.__name__: model for model in (
    BlockCreate, ArchitectureCreate, MetadataPatch, BlockDraftSave,
    ArchitectureDraftSave, DraftSaved, BlockDraftView, ArchitectureDraftView,
    ResourceSummary, CatalogPage, ResourceCreated, ResourceDetail, CatalogQuery,
    AcceptRevision, AcceptedRevision, RevisionPage, BlockRevisionView,
    ArchitectureRevisionView, TensorInputSpec, ValidationRequest,
    SourceLocation, Diagnostic, ErrorResponse, StageOutcome, ValidationReport,
    GraphQuery, GraphCursor, TensorSummary, GraphNode, GraphEdge, GraphFragment,
    ArchitectureSelection, EnqueueEvidence, AuthorizedVisibility, RunOutput,
    ValidationRun, ValidationEvent, ModelDefinitionSummary,
)}
