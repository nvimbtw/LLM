"""Version-one model declarations and evidence identities.

These typed boundary contracts do not parse YAML, import source, build models or
own durable lifecycle state. The resolver and storage adapters enforce catalog,
lexical scope, graph and transactional invariants against these declarations.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StrictBool, StrictFloat,
    StrictInt, StrictStr, field_validator, model_validator,
)

MAX_SOURCE_BYTES = 256 * 1024
MAX_YAML_BYTES = 256 * 1024
MAX_GRAPH_BYTES = 8 * 1024 * 1024
MAX_GRAPH_NODES = 4096
MAX_GRAPH_EDGES = 16384
MAX_EXPANDED_CALLS = 128
MAX_NESTING = 8
MAX_MODULES = 128
MAX_CATALOG_PAGE = 100
MAX_GRAPH_PAGE_NODES = 256
MAX_ATTENTION_SEQUENCE = 512
MAX_ATTENTION_HEADS = 256
MAX_DEPENDENCIES = 128
MAX_PARAMETERS = 64
MAX_PORTS = 32

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
PythonName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
Scalar = StrictBool | Annotated[StrictInt, Field(ge=-9007199254740991, le=9007199254740991)] | StrictFloat | Annotated[StrictStr, Field(max_length=4096)]
CapabilityName = Literal["cache", "introspection", "attention", "context_extension", "recompute", "token_rows"]


def checked_binding(value: Scalar) -> Scalar:
    """Require whole-value parameter/tokenizer references; reject interpolation or expressions."""
    if isinstance(value, str) and "${" in value:
        if re.fullmatch(r"\$\{(?:[a-z][a-z0-9_]*|tokenizer\.(?:vocabulary_size|[a-z][a-z0-9_]*_id))\}", value) is None:
            raise ValueError("binding must be one whole parameter or tokenizer reference")
    return value


Binding = Annotated[Scalar, AfterValidator(checked_binding)]
CountBinding = Annotated[StrictInt | StrictStr, AfterValidator(checked_binding)]


class Contract(BaseModel):
    """Reject unknown interpreted fields and freeze boundary record attributes."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("schema_version", "sdk_version", "format_version", check_fields=False, mode="before")
    @classmethod
    def reject_boolean_version(cls, value: object) -> object:
        """Keep booleans from masquerading as integer protocol versions."""
        if isinstance(value, bool):
            raise ValueError("version must be an integer")
        return value


class RevisionRef(Contract):
    """Pin immutable bytes separately from their resource's editable display identity."""

    kind: Literal["block", "architecture"]
    resource_id: UUID
    revision_id: UUID
    content_hash: Sha256


class DraftSubject(Contract):
    """Select exact saved draft bytes; later saves cannot mutate this snapshot."""

    kind: Literal["draft"]
    resource_id: UUID
    draft_version: PositiveInt
    content_hash: Sha256


class RevisionSubject(Contract):
    """Select an already accepted immutable source/recipe revision."""

    kind: Literal["revision"]
    reference: RevisionRef


Subject = Annotated[DraftSubject | RevisionSubject, Field(discriminator="kind")]


class ParameterDeclaration(Contract):
    """Declare constructor value types, defaults and editor bounds without executing code."""

    type: Literal["integer", "number", "boolean", "string", "enum"]
    required: StrictBool = False
    default: Scalar | None = None
    minimum: StrictInt | StrictFloat | None = None
    maximum: StrictInt | StrictFloat | None = None
    choices: Annotated[tuple[Scalar, ...], Field(max_length=64)] = ()
    title: Annotated[str, Field(max_length=120)] = ""
    description: Annotated[str, Field(max_length=2000)] = ""

    @model_validator(mode="after")
    def check_declaration(self) -> ParameterDeclaration:
        """Refuse inconsistent bounds, ambiguous defaults and duplicate enum alternatives."""
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")
        if self.required and self.default is not None:
            raise ValueError("required parameter cannot also declare a default")
        if self.type == "enum" and not self.choices:
            raise ValueError("enum parameter needs choices")
        if self.type != "enum" and self.choices:
            raise ValueError("choices require enum type")
        identities = [(type(value).__name__, value) for value in self.choices]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate parameter choice")
        if self.default is not None:
            self.validate_value(self.default)
        return self

    def validate_value(self, value: Scalar) -> Scalar:
        """Validate a resolved constructor value without coercing booleans or numeric strings."""
        matches = {
            "integer": type(value) is int,
            "number": type(value) in (int, float),
            "boolean": type(value) is bool,
            "string": type(value) is str,
            "enum": any(type(value) is type(choice) and value == choice for choice in self.choices),
        }
        if not matches[self.type]:
            raise ValueError(f"parameter requires {self.type}")
        if self.minimum is not None and (type(value) not in (int, float) or value < self.minimum):
            raise ValueError("parameter is below minimum")
        if self.maximum is not None and (type(value) not in (int, float) or value > self.maximum):
            raise ValueError("parameter exceeds maximum")
        return value


class TensorPort(Contract):
    """Declare tensor rank, dtype family and resolved or named dimension constraints."""

    dtype: Literal["int64", "int32", "bool", "floating", "float32", "float16", "bfloat16"]
    shape: Annotated[tuple[PositiveInt | Identifier, ...], Field(min_length=1, max_length=8)]


class BlockDeclaration(Contract):
    """Expose declaration metadata before any constructor/profile inputs are available."""

    sdk_version: Literal[1]
    name: Identifier
    title: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=2000)] = ""
    parameters: Annotated[dict[PythonName, ParameterDeclaration], Field(max_length=MAX_PARAMETERS)]
    inputs: Annotated[dict[PythonName, TensorPort], Field(min_length=1, max_length=MAX_PORTS)]
    outputs: Annotated[dict[PythonName, TensorPort], Field(min_length=1, max_length=MAX_PORTS)]
    capabilities: tuple[CapabilityName, ...] = ()

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(cls, values: tuple[CapabilityName, ...]) -> tuple[CapabilityName, ...]:
        """Require each declaration capability once instead of merging ambiguous duplicates."""
        if len(set(values)) != len(values):
            raise ValueError("duplicate capability")
        return values


class ModuleDeclaration(Contract):
    """Create exactly one instance of a pinned block/subarchitecture with typed bindings."""

    reference: RevisionRef
    parameters: Annotated[dict[PythonName, Binding], Field(max_length=MAX_PARAMETERS)] = Field(default_factory=dict)


class ParameterAlias(Contract):
    """Tie one exact compatible Parameter identity; the resolver rejects overlapping views."""

    owner: Annotated[str, Field(min_length=1, max_length=512)]
    target: Annotated[str, Field(min_length=1, max_length=512)]

    @model_validator(mode="after")
    def distinct_endpoints(self) -> ParameterAlias:
        """Refuse redundant self-aliases before resolving paths or optimizer policy."""
        if self.owner == self.target:
            raise ValueError("parameter alias must have distinct endpoints")
        return self


class CarryBinding(Contract):
    """Initialize an iteration value and export its final value into a fresh outer name."""

    initial: Identifier
    input: Identifier
    output: Identifier
    result: Identifier


class Call(Contract):
    """Call an existing instance; named outputs introduce fresh lexical values."""

    call: Identifier
    inputs: Annotated[dict[PythonName, Identifier], Field(max_length=MAX_PORTS)]
    outputs: Annotated[dict[PythonName, Identifier], Field(min_length=1, max_length=MAX_PORTS)]


class Repeat(Contract):
    """Run a fixed bounded body with shared instances and explicit lexical carry."""

    repeat: CountBinding
    carry: Annotated[dict[Identifier, CarryBinding], Field(min_length=1, max_length=MAX_PORTS)]
    body: Annotated[tuple[Call | Repeat | Stack, ...], Field(min_length=1, max_length=MAX_EXPANDED_CALLS)]

    @field_validator("repeat")
    @classmethod
    def positive_count(cls, value: int | str) -> int | str:
        """Require a bounded positive literal or whole parameter reference for fixed work."""
        return checked_count(value)


class Stack(Contract):
    """Construct independent indexed instances; carry maps template ports across them."""

    stack: CountBinding
    name: Identifier
    template: ModuleDeclaration
    inputs: Annotated[dict[PythonName, Identifier], Field(max_length=MAX_PORTS)] = Field(default_factory=dict)
    carry: Annotated[dict[Identifier, CarryBinding], Field(min_length=1, max_length=MAX_PORTS)]

    @field_validator("stack")
    @classmethod
    def positive_count(cls, value: int | str) -> int | str:
        """Apply the same fixed-count admission contract as a shared repeat."""
        return checked_count(value)


def checked_count(value: int | str) -> int | str:
    """Validate fixed-work counts before parameter resolution or model allocation."""
    if type(value) is int and 1 <= value <= MAX_EXPANDED_CALLS:
        return value
    if isinstance(value, str) and re.fullmatch(r"\$\{[a-z][a-z0-9_]*\}", value):
        return value
    raise ValueError("count must be a positive bounded integer or whole parameter reference")


class Recipe(Contract):
    """Version source composition independently of training controls and capture evidence."""

    schema_version: Literal[1]
    interface: Literal["causal_lm", "tensor"]
    parameters: Annotated[dict[PythonName, ParameterDeclaration], Field(max_length=MAX_PARAMETERS)]
    inputs: Annotated[dict[PythonName, TensorPort], Field(min_length=1, max_length=MAX_PORTS)]
    outputs: Annotated[dict[PythonName, TensorPort], Field(min_length=1, max_length=MAX_PORTS)]
    result: Annotated[dict[PythonName, Identifier], Field(min_length=1, max_length=MAX_PORTS)]
    context_window: CountBinding | None = None
    modules: Annotated[dict[Identifier, ModuleDeclaration], Field(max_length=MAX_MODULES)]
    aliases: Annotated[tuple[ParameterAlias, ...], Field(max_length=MAX_MODULES)] = ()
    flow: Annotated[tuple[Call | Repeat | Stack, ...], Field(min_length=1, max_length=MAX_EXPANDED_CALLS)]

    @model_validator(mode="after")
    def check_interface(self) -> Recipe:
        """Require declared outputs and one structural context authority for causal models."""
        if set(self.result) != set(self.outputs):
            raise ValueError("result bindings must match declared output ports")
        if self.interface == "causal_lm" and ("tokens" not in self.inputs or "logits" not in self.outputs or self.context_window is None):
            raise ValueError("causal_lm requires tokens, logits and context_window")
        if self.context_window is not None:
            if type(self.context_window) is int and self.context_window < 1:
                raise ValueError("context_window must be positive")
            if isinstance(self.context_window, str) and not re.fullmatch(r"\$\{[a-z][a-z0-9_]*\}", self.context_window):
                raise ValueError("context_window must bind one structural parameter")
        if any(name in {"vocabulary_size", "vocab_size", "tokenizer"} or name.endswith("_token_id") for name in self.parameters):
            raise ValueError("tokenizer-derived parameters are reserved engine bindings")
        return self


class FinalTokenizer(Contract):
    """Bind authoritative vocabulary facts, including compatible derived token-label identity."""

    kind: Literal["final"]
    tokenizer_id: UUID
    fingerprint: Sha256
    vocabulary_size: PositiveInt
    reserved_token_ids: Annotated[dict[Identifier, NonnegativeInt], Field(max_length=128)] = Field(default_factory=dict)
    derived_from: Sha256 | None = None

    @model_validator(mode="after")
    def check_ids(self) -> FinalTokenizer:
        """Keep reserved IDs unique and inside the vocabulary without shifting their values."""
        values = tuple(self.reserved_token_ids.values())
        if len(set(values)) != len(values) or any(value >= self.vocabulary_size for value in values):
            raise ValueError("reserved IDs must be unique and within vocabulary")
        return self


class PendingTokenizer(Contract):
    """Mark unresolved tokenizer provenance; it cannot authorize acceptance or optimization."""

    kind: Literal["pending"]
    build_request_id: UUID


TokenizerBinding = Annotated[FinalTokenizer | PendingTokenizer, Field(discriminator="kind")]


class ShapeDomain(Contract):
    """Bound actual tensor signatures; singleton specializations are explicit profiles."""

    batch_min: PositiveInt
    batch_max: PositiveInt
    sequence_min: PositiveInt
    sequence_max: PositiveInt
    signatures: Annotated[tuple[Literal["full_prefix", "prefill", "decode"], ...], Field(min_length=1, max_length=3)]
    past_min: NonnegativeInt = 0
    past_max: NonnegativeInt = 0

    @model_validator(mode="after")
    def check_ranges(self) -> ShapeDomain:
        """Refuse inverted, duplicate or inconsistent symbolic input domains."""
        if self.batch_min > self.batch_max or self.sequence_min > self.sequence_max or self.past_min > self.past_max:
            raise ValueError("invalid shape range")
        if len(set(self.signatures)) != len(self.signatures):
            raise ValueError("duplicate serving signature")
        return self


class RuntimeProfile(Contract):
    """Bind numerical/capture evidence to runtime compatibility and explicit shape guards."""

    torch_version: Annotated[str, Field(min_length=1, max_length=64)]
    sdk_version: Literal[1]
    device: Annotated[str, Field(min_length=1, max_length=64)]
    precision: Literal["fp32", "fp16", "bf16"]
    capture_policy: Literal["torch-export-strict-v1"]
    domains: Annotated[tuple[ShapeDomain, ...], Field(min_length=1, max_length=16)]
    modes: Annotated[tuple[Literal["train", "eval"], ...], Field(min_length=1, max_length=2)]
    token_dtype: Literal["int64", "int32"] = "int64"
    allow_tf32: StrictBool = False
    numerical_seed: NonnegativeInt = 0
    export_guards: Annotated[tuple[Annotated[str, Field(max_length=1024)], ...], Field(max_length=128)] = ()

    @field_validator("modes")
    @classmethod
    def unique_modes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require each trained/evaluated mode once in its evidence identity."""
        if len(set(values)) != len(values):
            raise ValueError("duplicate execution mode")
        return values


class ReceiptIdentity(Contract):
    """Identify one exact snapshot/configuration/profile, never all parameter combinations."""

    subject: Subject
    dependency_digest: Sha256
    parameter_digest: Sha256
    input_profile_digest: Sha256
    tokenizer: TokenizerBinding | None
    context_window: PositiveInt | None
    profile: RuntimeProfile


class Capability(Contract):
    """Report optional supported behavior only when required profile evidence exists."""

    name: CapabilityName
    supported: StrictBool
    reason: Annotated[str, Field(max_length=2000)] | None = None
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def require_evidence(self) -> Capability:
        """Distinguish unsupported absence from an evidence-backed optional execution path."""
        if self.supported and self.evidence_hash is None:
            raise ValueError("supported capability requires evidence")
        if not self.supported and not self.reason:
            raise ValueError("unsupported capability requires reason")
        return self


class ValidationReceipt(Contract):
    """Persist acceptance/enqueue evidence separately from a prunable execution-job row."""

    schema_version: Literal[1]
    receipt_id: UUID
    identity: ReceiptIdentity
    identity_hash: Sha256
    verdict: Literal["inspection_only", "invalid", "valid", "provisional"]
    graph_manifest_hash: Sha256 | None = None
    capabilities: Annotated[tuple[Capability, ...], Field(max_length=6)] = ()

    @model_validator(mode="after")
    def check_authorization(self) -> ValidationReceipt:
        """Refuse stale identity hashes and provisional bindings disguised as full success."""
        if self.identity_hash != canonical_digest(self.identity):
            raise ValueError("receipt identity hash mismatch")
        if self.verdict == "valid" and isinstance(self.identity.tokenizer, PendingTokenizer):
            raise ValueError("pending tokenizer requires provisional verdict")
        if self.verdict == "valid" and set(self.identity.profile.modes) != {"train", "eval"}:
            raise ValueError("full validation requires training and evaluation evidence")
        if self.verdict == "valid" and self.graph_manifest_hash is None:
            raise ValueError("full validation requires published complete-capture evidence")
        if len({item.name for item in self.capabilities}) != len(self.capabilities):
            raise ValueError("duplicate capability evidence")
        return self

    def require_current(self, expected: ReceiptIdentity) -> None:
        """Authorize only fully valid, exact, fully bound evidence; raise on stale inputs."""
        if self.verdict != "valid" or self.identity_hash != canonical_digest(expected):
            raise ValueError("receipt is not valid for the current configuration")
        if isinstance(expected.tokenizer, PendingTokenizer):
            raise ValueError("pending tokenizer cannot authorize optimizer execution")


class SourceManifestEntry(Contract):
    """Pin immutable dependency source and import identity inside a self-contained package."""

    reference: RevisionRef
    path: Annotated[str, Field(min_length=1, max_length=512)]
    bytes: Annotated[NonnegativeInt, Field(le=MAX_SOURCE_BYTES)]
    sdk_version: Literal[1]
    builtin_compatibility: Sha256 | None = None

    @field_validator("path")
    @classmethod
    def relative_source_path(cls, value: str) -> str:
        """Keep source manifest paths package-relative without traversal or platform escapes."""
        return checked_relative_path(value)


def checked_relative_path(value: str) -> str:
    """Validate a POSIX package-local path before filesystem materialization."""
    if value.startswith("/") or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must be a normalized package-relative POSIX path")
    return value


class ResolvedDefinition(Contract):
    """Freeze semantic model identity without embedding device/profile/graph evidence."""

    schema_version: Literal[1]
    architecture: RevisionRef
    recipe: Recipe
    dependencies: Annotated[tuple[SourceManifestEntry, ...], Field(max_length=MAX_DEPENDENCIES)]
    parameters: Annotated[dict[Identifier, Scalar], Field(max_length=MAX_PARAMETERS)]
    tokenizer: FinalTokenizer
    context_window: PositiveInt
    aliases: Annotated[tuple[ParameterAlias, ...], Field(max_length=MAX_MODULES)]
    sdk_version: Literal[1]

    @model_validator(mode="after")
    def consistent_bindings(self) -> ResolvedDefinition:
        """Require one context authority and exact typed structural values in saved semantics."""
        if self.architecture.kind != "architecture":
            raise ValueError("definition root must be an architecture revision")
        if set(self.parameters) != set(self.recipe.parameters):
            raise ValueError("resolved parameters must exactly match declarations")
        for name, declaration in self.recipe.parameters.items():
            declaration.validate_value(self.parameters[name])
        context = self.recipe.context_window
        if isinstance(context, str):
            context = self.parameters.get(context[2:-1])
        if context != self.context_window:
            raise ValueError("resolved context disagrees with structural context binding")
        if self.aliases != self.recipe.aliases:
            raise ValueError("resolved aliases differ from the recipe")
        revisions = [entry.reference.revision_id for entry in self.dependencies]
        paths = [entry.path for entry in self.dependencies]
        if len(set(revisions)) != len(revisions) or len(set(paths)) != len(paths):
            raise ValueError("duplicate dependency revision or source path")
        return self


class DefinitionPackage(Contract):
    """Discriminate new packages while keeping legacy version-one checkpoint loading intact."""

    format: Literal["slate.modular-model"]
    format_version: Literal[1]
    definition_id: Sha256
    definition: ResolvedDefinition
    receipt_ids: Annotated[tuple[UUID, ...], Field(min_length=1, max_length=64)]
    graph_hashes: Annotated[tuple[Sha256, ...], Field(max_length=64)]
    weights_path: Literal["model.pt"] = "model.pt"
    tokenizer_path: Literal["tokenizer.json"] = "tokenizer.json"
    metadata_path: Literal["meta.json"] = "meta.json"
    training_state_path: Literal["train_state.pt"] | None = None
    parameter_precision: Literal["fp32", "fp16", "bf16"] | None = Field(
        default=None, description="Saved floating parameter/buffer dtype; absent legacy packages use receipt precision.",
    )
    optimizer_parameter_order: Annotated[tuple[Annotated[str, Field(min_length=1, max_length=512)], ...], Field(max_length=4096)] = ()

    @model_validator(mode="after")
    def check_definition_digest(self) -> DefinitionPackage:
        """Reject tampered resolved semantics before importing packaged source or loading tensors."""
        if self.definition_id != canonical_digest(self.definition):
            raise ValueError("definition digest mismatch")
        if len(set(self.optimizer_parameter_order)) != len(self.optimizer_parameter_order):
            raise ValueError("duplicate optimizer parameter identity")
        return self


def canonical_digest(value: BaseModel | dict[str, object]) -> str:
    """Hash sorted UTF-8 JSON without runtime/profile artifacts or nonfinite values."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


Repeat.model_rebuild()
Recipe.model_rebuild()

CONTRACT_MODELS = {model.__name__: model for model in (
    RevisionRef, DraftSubject, RevisionSubject, ParameterDeclaration, TensorPort,
    BlockDeclaration, ModuleDeclaration, ParameterAlias, CarryBinding, Call,
    Repeat, Stack, Recipe, FinalTokenizer, PendingTokenizer, ShapeDomain,
    RuntimeProfile, ReceiptIdentity, Capability, ValidationReceipt,
    SourceManifestEntry, ResolvedDefinition, DefinitionPackage,
)}
