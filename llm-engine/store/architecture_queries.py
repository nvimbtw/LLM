"""Transactional catalog revisions, evidence retention and immutable enqueue snapshots.

Callers own one Session transaction and commit only after the whole operation or
chain succeeds. These functions never import source, execute models or publish
files; immutable source rows and retained evidence survive execution pruning.
"""

from __future__ import annotations

import hashlib
import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from llm.architecture import dependencies
from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import (
    MAX_DEPENDENCIES, MAX_NESTING, BlockDeclaration, DraftSubject, Recipe,
    FinalTokenizer, ResolvedDefinition, RevisionRef, RevisionSubject, ValidationReceipt,
    canonical_digest,
)
from llm.architecture.wire import (
    AcceptRevision, ArchitectureCreate, ArchitectureDraftSave, BlockCreate,
    BlockDraftSave, EnqueueEvidence, MetadataPatch,
)
from store import queries
from store.schema import (
    Architecture, ArchitectureDependency, ArchitectureRevision, BlockRevision,
    DefinitionEvidence, ModelBlock, ModelDefinition, ModelValidation, MutationRequest, Run,
    RunArchitecturePin, RunKind, RunStatus, ValidationEvidence, _now,
)

ResourceKind = Literal["block", "architecture"]
CatalogResource = ModelBlock | Architecture
TRAINING_SNAPSHOT_BYTES = 8 << 20


class CatalogError(ValueError):
    """Expose a stable domain refusal and optional current optimistic version to API adapters."""

    def __init__(self, code: str, message: str, current_version: int | None = None):
        """Retain actionable domain identity without coupling storage to HTTP status types."""
        super().__init__(message)
        self.code = code
        self.current_version = current_version


@dataclass(frozen=True)
class MutationOutcome:
    """Retain an idempotent operation's logical output without a prunable job foreign key."""

    kind: Literal["validation", "revision", "training"]
    id: uuid.UUID
    job_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True)
class RevisionSnapshot:
    """Return an immutable revision identity with its exact retained source and declarations."""

    reference: RevisionRef
    source: str
    metadata: dict
    compatibility: dict | None = None
    origin: Literal["builtin", "custom"] | None = None


@dataclass(frozen=True)
class PreparedTrainingPin:
    """Own validated source/evidence snapshots while their preparing Session retains root/evidence locks."""

    evidence: EnqueueEvidence
    parameters: dict
    snapshot: dict


def text_hash(source: str) -> str:
    """Hash exact UTF-8 source bytes without parsing, normalizing or executing them."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _resource_type(kind: ResourceKind) -> type[ModelBlock] | type[Architecture]:
    """Select the explicit catalog family; arbitrary resource implementations are unsupported."""
    return ModelBlock if kind == "block" else Architecture


def _locked_resource(db: Session, kind: ResourceKind, resource_id: uuid.UUID) -> CatalogResource:
    """Lock and refresh the mutable root before comparing versions or selection eligibility."""
    model = _resource_type(kind)
    row = db.scalar(select(model).where(model.id == resource_id).with_for_update().execution_options(populate_existing=True))
    if row is None:
        raise CatalogError("not_found", "catalog resource not found")
    return row


def _require_custom(resource: CatalogResource) -> None:
    """Refuse public mutations to supplied read-only builtin sources and metadata."""
    if resource.origin != "custom":
        raise CatalogError("read_only", "builtin catalog resources are read-only")


def create_block(db: Session, actor_id: uuid.UUID, request: BlockCreate, *, resource_id: uuid.UUID | None = None, builtin: bool = False) -> ModelBlock:
    """Persist a first Python draft without syntax checks; builtin is an internal seeding choice."""
    row = ModelBlock(id=resource_id or uuid.uuid4(), name=request.name, description=request.description,
                     draft_source=request.source, draft_hash=text_hash(request.source), creator_id=actor_id,
                     origin="builtin" if builtin else "custom")
    db.add(row)
    db.flush()
    return row


def create_architecture(db: Session, actor_id: uuid.UUID, request: ArchitectureCreate, *, resource_id: uuid.UUID | None = None, builtin: bool = False) -> Architecture:
    """Persist a first YAML draft without parsing or reference resolution."""
    row = Architecture(id=resource_id or uuid.uuid4(), name=request.name, description=request.description,
                       draft_yaml=request.yaml, draft_hash=text_hash(request.yaml), creator_id=actor_id,
                       origin="builtin" if builtin else "custom")
    db.add(row)
    db.flush()
    return row


def seed_builtin_block(db: Session, actor_id: uuid.UUID, reference: RevisionRef, request: BlockCreate) -> ModelBlock:
    """Reconcile shipped source under exclusive startup ownership; public builtin edits remain forbidden and old revisions survive."""
    if reference.kind != "block":
        raise CatalogError("identity_conflict","builtin block identity requires a block reference")
    return _seed_builtin_resource(db,"block",reference.resource_id,actor_id,request)


def seed_builtin_architecture(db: Session, actor_id: uuid.UUID, resource_id: uuid.UUID, request: ArchitectureCreate) -> Architecture:
    """Reconcile supplied recipe drafts under exclusive startup ownership, retaining all prior accepted source/pins."""
    return _seed_builtin_resource(db,"architecture",resource_id,actor_id,request)


def _seed_builtin_resource(db: Session, kind: ResourceKind, resource_id: uuid.UUID, actor_id: uuid.UUID,
                           request: BlockCreate | ArchitectureCreate) -> CatalogResource:
    """Own shared supplied-source and metadata version transitions; callers hold exclusive startup ownership."""
    row = db.get(_resource_type(kind),resource_id)
    if row is None:
        return (create_block(db,actor_id,request,resource_id=resource_id,builtin=True) if kind == "block"
                else create_architecture(db,actor_id,request,resource_id=resource_id,builtin=True))
    row = _locked_resource(db,kind,resource_id)
    if row.origin != "builtin":
        raise CatalogError("identity_conflict","installed builtin UUID belongs to a custom resource")
    source = request.source if kind == "block" else request.yaml
    previous = row.draft_source if kind == "block" else row.draft_yaml
    digest = text_hash(source)
    if previous != source or row.draft_hash != digest:
        if kind == "block":
            row.draft_source = source
        else:
            row.draft_yaml = source
        row.draft_hash = digest
        row.draft_version += 1
        row.updated_at = _now()
    if row.name != request.name or row.description != request.description:
        row.name, row.description = request.name, request.description
        row.metadata_version += 1
        row.updated_at = _now()
    db.flush()
    return row


def _save_draft(db: Session, resource: CatalogResource, expected: int, source: str) -> CatalogResource:
    """Advance exact bytes/version under a root lock; stale writers cannot overwrite another save."""
    _require_custom(resource)
    if resource.draft_version != expected:
        raise CatalogError("stale_version", "draft version changed", resource.draft_version)
    if isinstance(resource, ModelBlock):
        resource.draft_source = source
    else:
        resource.draft_yaml = source
    resource.draft_hash = text_hash(source)
    resource.draft_version += 1
    resource.updated_at = _now()
    db.flush()
    return resource


def save_block_draft(db: Session, block_id: uuid.UUID, request: BlockDraftSave) -> ModelBlock:
    """Save bounded exact Python bytes with optimistic concurrency and no source execution."""
    resource = _locked_resource(db, "block", block_id)
    return _save_draft(db, resource, request.expected_version, request.source)


def save_architecture_draft(db: Session, architecture_id: uuid.UUID, request: ArchitectureDraftSave) -> Architecture:
    """Save bounded exact YAML bytes with optimistic concurrency and no semantic validation."""
    resource = _locked_resource(db, "architecture", architecture_id)
    return _save_draft(db, resource, request.expected_version, request.yaml)


def update_metadata(db: Session, kind: ResourceKind, resource_id: uuid.UUID, request: MetadataPatch) -> CatalogResource:
    """Apply only stated metadata/archive fields after locking its independent metadata version."""
    row = _locked_resource(db, kind, resource_id)
    _require_custom(row)
    if row.metadata_version != request.expected_metadata_version:
        raise CatalogError("stale_version", "metadata version changed", row.metadata_version)
    for name in request.model_fields_set - {"expected_metadata_version"}:
        value = getattr(request, name)
        if name in {"name", "archived"} and value is None:
            raise CatalogError("invalid_metadata", f"{name} cannot be null")
        if name == "name" and not value.strip():
            raise CatalogError("invalid_metadata", "name cannot be blank")
        setattr(row, name, "" if name == "description" and value is None else value)
    row.metadata_version += 1
    row.updated_at = _now()
    db.flush()
    return row


def list_resources(db: Session, kind: ResourceKind, *, archived: bool = False, origin: str | None = None, search: str = "", after: uuid.UUID | None = None, limit: int = 100) -> list[CatalogResource]:
    """Read one bounded keyset page without source imports or loading revision bodies."""
    if not 1 <= limit <= 100 or len(search) > 200 or origin not in {None, "builtin", "custom"}:
        raise CatalogError("invalid_page", "invalid catalog filter or page budget")
    model = _resource_type(kind)
    statement = select(model).where(model.archived == archived)
    if origin is not None:
        statement = statement.where(model.origin == origin)
    if search:
        statement = statement.where(model.name.ilike(f"%{search}%"))
    if after is not None:
        statement = statement.where(model.id > after)
    return list(db.scalars(statement.order_by(model.id).limit(limit)))


def get_resource(db: Session, kind: ResourceKind, resource_id: uuid.UUID) -> CatalogResource:
    """Read one resource's saved data without importing source or acquiring a mutation lock."""
    row = db.get(_resource_type(kind),resource_id)
    if row is None:
        raise CatalogError("not_found", "catalog resource not found")
    return row


def validation_resource(db: Session, kind: ResourceKind, resource_id: uuid.UUID) -> CatalogResource:
    """Serialize validation admission with draft/archive edits under the caller's queue transaction."""
    row = _locked_resource(db,kind,resource_id)
    if row.archived:
        raise CatalogError("archived","archived resources cannot start new validation")
    return row


def revision_rows(db: Session, kind: ResourceKind, resource_id: uuid.UUID, *, after: int = 0,
                  limit: int = 50, latest: bool = False) -> list[BlockRevision | ArchitectureRevision]:
    """Read a bounded ordered revision page or latest accepted row without source execution."""
    if not 1 <= limit <= 100 or after < 0:
        raise CatalogError("invalid_page", "invalid revision page budget")
    model = BlockRevision if kind == "block" else ArchitectureRevision
    owner = model.block_id if kind == "block" else model.architecture_id
    statement = select(model).where(owner == resource_id,model.revision_number > after)
    order = model.revision_number.desc() if latest else model.revision_number.asc()
    return list(db.scalars(statement.order_by(order).limit(limit)))


def revision_count(db: Session, kind: ResourceKind, resource_id: uuid.UUID) -> int:
    """Count immutable accepted revisions without loading any source bodies."""
    model = BlockRevision if kind == "block" else ArchitectureRevision
    owner = model.block_id if kind == "block" else model.architecture_id
    return db.scalar(select(func.count()).select_from(model).where(owner == resource_id)) or 0


def architecture_revision_for_source(db: Session, resource_id: uuid.UUID, content_hash: str,
                                     dependency_digest: str) -> ArchitectureRevision | None:
    """Find an already accepted exact source/pin identity, including an older revision restored by a shipped-code rollback."""
    return db.scalar(select(ArchitectureRevision).where(ArchitectureRevision.architecture_id == resource_id,
        ArchitectureRevision.content_hash == content_hash,ArchitectureRevision.dependency_digest == dependency_digest).limit(1))


def get_catalog_revision(db: Session, kind: ResourceKind, resource_id: uuid.UUID,
                         revision_id: uuid.UUID) -> BlockRevision | ArchitectureRevision:
    """Read immutable source only under its actual family/resource identity, refusing cross-resource replay."""
    model = BlockRevision if kind == "block" else ArchitectureRevision
    row = db.get(model,revision_id)
    if row is None or (row.block_id if kind == "block" else row.architecture_id) != resource_id:
        raise CatalogError("not_found", "catalog revision not found")
    return row


def evidence_for_job(db: Session, run_id: uuid.UUID) -> ValidationEvidence | None:
    """Find independently retained evidence without requiring the prunable source validation job."""
    return db.scalar(select(ValidationEvidence).where(ValidationEvidence.source_run_id == run_id))


def bound_training_publication(db: Session) -> None:
    """Keep parent publication locks/statements finite within its caller-owned PostgreSQL transaction."""
    if db.get_bind().dialect.name == "postgresql":
        db.execute(text("SET LOCAL lock_timeout = '5s'"))
        db.execute(text("SET LOCAL statement_timeout = '30s'"))


def record_evidence(db: Session, source_run_id: uuid.UUID, receipt: ValidationReceipt) -> ValidationEvidence:
    """Retain exact immutable evidence independently of execution/history deletion."""
    existing = db.get(ValidationEvidence, receipt.receipt_id)
    payload = receipt.model_dump(mode="json")
    if existing is not None:
        if existing.receipt != payload or existing.source_run_id != source_run_id:
            raise CatalogError("identity_conflict", "receipt UUID already identifies different evidence")
        return existing
    run = db.get(Run, source_run_id)
    if run is None or run.kind != RunKind.model_validation or run.status != RunStatus.done:
        raise CatalogError("validation_not_complete", "new retained evidence requires a completed validation job")
    row = ValidationEvidence(receipt_id=receipt.receipt_id, identity_hash=receipt.identity_hash,
                             receipt=payload, source_run_id=source_run_id)
    db.add(row)
    db.flush()
    if run.kind == RunKind.model_validation:
        if run.model_validation.evidence_id is not None and run.model_validation.evidence_id != receipt.receipt_id:
            raise CatalogError("identity_conflict", "validation job already owns different receipt evidence")
        run.model_validation.evidence_id = receipt.receipt_id
        db.flush()
    return row


def _receipt(db: Session, receipt_id: uuid.UUID) -> ValidationReceipt:
    """Read fully typed retained evidence rather than relying on mutable validation history."""
    row = db.scalar(select(ValidationEvidence).where(ValidationEvidence.receipt_id == receipt_id).with_for_update(read=True, key_share=True))
    if row is None:
        raise CatalogError("missing_evidence", "retained validation evidence not found")
    return ValidationReceipt.model_validate(row.receipt)


def _require_training_definition(receipt: ValidationReceipt, definition: ResolvedDefinition) -> None:
    """Require valid final revision evidence to name the exact saved root, parameters, tokenizer and context."""
    identity = receipt.identity
    receipt.require_current(identity)
    if (not isinstance(identity.subject, RevisionSubject) or identity.subject.reference != definition.architecture
            or identity.parameter_digest != canonical_digest(definition.parameters)
            or identity.tokenizer != definition.tokenizer or identity.context_window != definition.context_window):
        raise CatalogError("definition_mismatch", "training evidence differs from the saved definition")


def record_training_evidence(db: Session, source_run_id: uuid.UUID, receipt: ValidationReceipt,
                             definition: ResolvedDefinition, *, prepared_tokenizer: FinalTokenizer | None = None) -> ValidationEvidence:
    """Retain published final-preflight evidence for an admitted immutable training pin.

    The parent verifies the complete package before this call. An interrupted or
    finished job may retain its published evidence during startup reconciliation;
    queued/waiting jobs cannot. Numerical validity is independent of the later
    training outcome. Existing identical evidence remains readable after pruning.
    Validation-job evidence still uses the separate done-only record_evidence.
    An SFT caller may finalize a pending chain tokenizer or derive labels only
    after checking registered saved inputs and durable parent preparation against
    actual output bytes. Other training retains its enqueue binding rules;
    immutable pins are never rewritten.
    """
    _require_training_definition(receipt, definition)
    existing = db.get(ValidationEvidence, receipt.receipt_id)
    if existing is not None:
        if existing.receipt != receipt.model_dump(mode="json") or existing.source_run_id != source_run_id:
            raise CatalogError("identity_conflict", "receipt UUID already identifies different evidence")
        return existing
    run = queries.lock_run(db, source_run_id)
    pin = db.get(RunArchitecturePin, source_run_id)
    if (run is None or run.kind not in (RunKind.train, RunKind.rl) or run.training is None or pin is None
            or run.status in (RunStatus.queued, RunStatus.waiting)):
        raise CatalogError("training_not_admitted", "new training evidence requires an admitted pinned job")
    evidence = EnqueueEvidence.model_validate(pin.evidence)
    if prepared_tokenizer is not None and run.training.training_type != "sft":
        raise CatalogError("definition_mismatch", "only parent-prepared SFT can finalize tokenizer labels")
    expected_tokenizer = prepared_tokenizer or evidence.tokenizer
    identity = receipt.identity
    expected_sources = {item["reference"]["revision_id"]: item["reference"] for item in pin.snapshot["dependencies"]}
    actual_sources = {str(item.reference.revision_id): item.reference.model_dump(mode="json") for item in definition.dependencies}
    if (identity.subject.reference != evidence.architecture
            or identity.dependency_digest != evidence.dependency_digest
            or identity.parameter_digest != canonical_digest(pin.parameters)
            or definition.architecture != evidence.architecture or definition.parameters != pin.parameters
            or Recipe.model_validate(pin.snapshot["recipe"]) != definition.recipe
            or actual_sources != expected_sources
            or identity.tokenizer != definition.tokenizer or identity.context_window != definition.context_window
            or (expected_tokenizer.kind == "final" and identity.tokenizer != expected_tokenizer)):
        raise CatalogError("definition_mismatch", "training evidence differs from the queued immutable bindings")
    previous = evidence_for_job(db, source_run_id)
    if previous is not None:
        raise CatalogError("identity_conflict", "training job already owns different final-preflight evidence")
    row = ValidationEvidence(receipt_id=receipt.receipt_id, identity_hash=receipt.identity_hash,
                             receipt=receipt.model_dump(mode="json"), source_run_id=source_run_id)
    db.add(row)
    db.flush()
    return row


def _acceptance_evidence(db: Session, resource: CatalogResource, request: AcceptRevision) -> ValidationReceipt:
    """Verify current draft/source/dependency identity and successful fully bound example evidence."""
    if resource.archived:
        raise CatalogError("archived", "archived resource cannot accept a new revision")
    if resource.draft_version != request.expected_draft_version or resource.draft_hash != request.expected_content_hash:
        raise CatalogError("stale_version", "draft changed before acceptance", resource.draft_version)
    evidence = db.scalar(select(ValidationEvidence).where(ValidationEvidence.source_run_id == request.validation_run_id,
                                                        ValidationEvidence.identity_hash.is_not(None)).order_by(ValidationEvidence.created_at.desc()).limit(1).with_for_update(read=True, key_share=True))
    if evidence is None:
        raise CatalogError("missing_evidence", "matching retained validation evidence not found")
    receipt = ValidationReceipt.model_validate(evidence.receipt)
    subject = receipt.identity.subject
    if not isinstance(subject, DraftSubject) or subject.resource_id != resource.id or subject.draft_version != resource.draft_version or subject.content_hash != resource.draft_hash:
        raise CatalogError("stale_evidence", "validation does not match the exact current draft")
    if receipt.identity.dependency_digest != request.expected_dependency_digest:
        raise CatalogError("stale_evidence", "validation dependency snapshot changed")
    try:
        receipt.require_current(receipt.identity)
    except ValueError as error:
        raise CatalogError("invalid_evidence", str(error)) from error
    return receipt


def accept_block(db: Session, block_id: uuid.UUID, request: AcceptRevision, declaration: BlockDeclaration, *, compatibility: dict | None = None, builtin_reference: RevisionRef | None = None) -> BlockRevision:
    """Accept validated source under the root lock; internal builtin pins identify their complete implementation manifest."""
    row = _locked_resource(db, "block", block_id)
    if builtin_reference is None:
        _require_custom(row)
    elif row.origin != "builtin" or builtin_reference.kind != "block" or builtin_reference.resource_id != row.id:
        raise CatalogError("identity_conflict","builtin acceptance requires its actual read-only resource")
    receipt = _acceptance_evidence(db, row, request)
    content_hash = builtin_reference.content_hash if builtin_reference is not None else row.draft_hash
    if builtin_reference is not None and receipt.identity.dependency_digest != dependency_digest({"builtin":builtin_reference}):
        raise CatalogError("stale_evidence","builtin evidence differs from its complete implementation manifest")
    existing = db.scalar(select(BlockRevision).where(BlockRevision.block_id == block_id, BlockRevision.content_hash == content_hash))
    if existing is not None:
        if builtin_reference is not None and revision_reference(existing) != builtin_reference:
            raise CatalogError("identity_conflict","builtin revision UUID differs from its implementation manifest")
        return existing
    number = (db.scalar(select(func.max(BlockRevision.revision_number)).where(BlockRevision.block_id == block_id)) or 0) + 1
    revision = BlockRevision(id=builtin_reference.revision_id if builtin_reference is not None else uuid.uuid4(),
                             block_id=block_id, revision_number=number, source=row.draft_source,
                             content_hash=content_hash, declaration=declaration.model_dump(mode="json"),
                             compatibility=compatibility or {"sdk_version": 1}, acceptance_receipt_id=receipt.receipt_id)
    db.add(revision)
    db.flush()
    return revision


def revision_reference(row: BlockRevision | ArchitectureRevision) -> RevisionRef:
    """Project the immutable identity/hash from a stored block or architecture revision."""
    block = isinstance(row, BlockRevision)
    return RevisionRef(kind="block" if block else "architecture", resource_id=row.block_id if block else row.architecture_id,
                       revision_id=row.id, content_hash=row.content_hash)


def require_revision(db: Session, reference: RevisionRef) -> BlockRevision | ArchitectureRevision:
    """Require exact type/resource/hash without consulting display names, archive or latest defaults."""
    model = BlockRevision if reference.kind == "block" else ArchitectureRevision
    row = db.get(model, reference.revision_id)
    if row is None or revision_reference(row) != reference:
        raise CatalogError("revision_snapshot_mismatch", "pinned dependency revision identity does not match")
    return row


def recipe_dependencies(recipe: Recipe) -> dict[str, RevisionRef]:
    """Collect declared direct pins, including nested independent stack templates, without expansion."""
    try:
        return dependencies.recipe_dependencies(recipe)
    except ArchitectureError as error:
        raise CatalogError(error.code, str(error)) from error


def dependency_digest(pins: dict[str, RevisionRef]) -> str:
    """Hash normalized path-to-revision pins independently of mutable archive/display metadata."""
    return dependencies.dependency_digest(pins)


def accept_architecture(db: Session, architecture_id: uuid.UUID, request: AcceptRevision, recipe: Recipe, *, internal_builtin: bool = False) -> ArchitectureRevision:
    """Accept exact recipe/pins atomically; immutable children remain usable even when archived."""
    row = _locked_resource(db, "architecture", architecture_id)
    if not internal_builtin:
        _require_custom(row)
    receipt = _acceptance_evidence(db, row, request)
    pins = recipe_dependencies(recipe)
    digest = dependency_digest(pins)
    if digest != request.expected_dependency_digest:
        raise CatalogError("stale_evidence", "resolved dependency pins differ from validated pins")
    for reference in pins.values():
        require_revision(db, reference)
    existing = db.scalar(select(ArchitectureRevision).where(ArchitectureRevision.architecture_id == architecture_id,
                         ArchitectureRevision.content_hash == row.draft_hash, ArchitectureRevision.dependency_digest == digest))
    if existing is not None:
        return existing
    number = (db.scalar(select(func.max(ArchitectureRevision.revision_number)).where(ArchitectureRevision.architecture_id == architecture_id)) or 0) + 1
    revision = ArchitectureRevision(architecture_id=architecture_id, revision_number=number, yaml=row.draft_yaml,
                                    content_hash=row.draft_hash, recipe=recipe.model_dump(mode="json"),
                                    dependency_digest=digest, acceptance_receipt_id=receipt.receipt_id)
    db.add(revision)
    db.flush()
    for path, reference in pins.items():
        db.add(ArchitectureDependency(architecture_revision_id=revision.id, instance_path=path,
               block_revision_id=reference.revision_id if reference.kind == "block" else None,
               child_architecture_revision_id=reference.revision_id if reference.kind == "architecture" else None))
    db.flush()
    dependency_closure(db, revision.id)
    return revision


def dependency_closure(db: Session, revision_id: uuid.UUID) -> tuple[RevisionSnapshot, ...]:
    """Read a bounded immutable transitive source closure, rejecting recursive/corrupt pins."""
    found: dict[uuid.UUID, RevisionSnapshot] = {}
    visited: set[uuid.UUID] = set()
    pending = [(revision_id, frozenset(), 0)]
    while pending:
        current, ancestry, depth = pending.pop()
        if current in ancestry or depth > MAX_NESTING:
            raise CatalogError("recursive_dependency", "recursive or overly nested architecture dependency")
        if current in visited:
            continue
        visited.add(current)
        rows = db.scalars(select(ArchitectureDependency).where(ArchitectureDependency.architecture_revision_id == current).order_by(ArchitectureDependency.instance_path))
        for dependency in rows:
            block = dependency.block_revision_id is not None
            row = db.get(BlockRevision if block else ArchitectureRevision, dependency.block_revision_id if block else dependency.child_architecture_revision_id)
            if row is None:
                raise CatalogError("missing_dependency", "retained dependency source is missing")
            if row.id not in found:
                metadata = row.declaration if block else row.recipe
                origin = get_resource(db, "block", row.block_id).origin if block else None
                found[row.id] = RevisionSnapshot(revision_reference(row), row.source if block else row.yaml, metadata,
                                                copy.deepcopy(row.compatibility) if block else None, origin)
            if not block:
                pending.append((row.id, ancestry | {current}, depth + 1))
            if len(found) > MAX_DEPENDENCIES:
                raise CatalogError("dependency_limit", "transitive dependency budget exceeded")
    return tuple(found[key] for key in sorted(found, key=str))


def prepare_training_pin(db: Session, evidence: EnqueueEvidence, parameters: dict) -> PreparedTrainingPin:
    """Validate and freeze one selection before any queue inserts, retaining locks until the caller commits."""
    revision = require_revision(db, evidence.architecture)
    if not isinstance(revision, ArchitectureRevision):
        raise CatalogError("invalid_root", "training root must be an architecture")
    resource = _locked_resource(db, "architecture", revision.architecture_id)
    if resource.archived:
        raise CatalogError("archived", "archived architecture cannot be newly selected")
    if revision.dependency_digest != evidence.dependency_digest or revision.acceptance_receipt_id != evidence.acceptance_receipt_id:
        raise CatalogError("stale_evidence", "acceptance or dependency identity changed")
    if canonical_digest(parameters) != evidence.parameter_digest:
        raise CatalogError("stale_parameters", "requested parameters differ from enqueue evidence")
    acceptance = _receipt(db, evidence.acceptance_receipt_id)
    acceptance.require_current(acceptance.identity)
    if evidence.exact_receipt_id is not None:
        exact = _receipt(db, evidence.exact_receipt_id)
        exact.require_current(exact.identity)
        _receipt_matches_revision(exact, revision)
        if exact.identity.parameter_digest != evidence.parameter_digest or exact.identity.dependency_digest != evidence.dependency_digest or exact.identity.tokenizer != evidence.tokenizer:
            raise CatalogError("stale_evidence", "exact receipt differs from submitted bindings")
    closure = dependency_closure(db, revision.id)
    return PreparedTrainingPin(evidence.model_copy(deep=True), dict(parameters), _training_snapshot(revision, closure))


def _training_snapshot(revision: ArchitectureRevision, closure: tuple[RevisionSnapshot, ...]) -> dict:
    """Freeze the same bounded immutable source/compatibility closure for fresh selections and saved-definition consumers."""
    snapshot = {"architecture": revision_reference(revision).model_dump(mode="json"), "yaml": revision.yaml,
                "recipe": revision.recipe, "dependencies": [
                    {"reference": item.reference.model_dump(mode="json"), "source": item.source, "metadata": item.metadata,
                     "compatibility": item.compatibility, "origin": item.origin}
                    for item in closure]}
    if len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) > TRAINING_SNAPSHOT_BYTES:
        raise CatalogError("snapshot_byte_limit", "training source snapshot exceeds worker artifact budget")
    return snapshot


def prepare_saved_training_pin(db: Session, definition_id: str) -> PreparedTrainingPin:
    """Inherit saved semantic/source/tokenizer authority without requiring an unpruned job or an unarchived catalog root."""
    row = db.get(ModelDefinition, definition_id)
    if row is None:
        raise CatalogError("missing_definition", "saved model definition is unavailable")
    definition = ResolvedDefinition.model_validate(row.definition)
    if canonical_digest(definition) != definition_id:
        raise CatalogError("definition_mismatch", "saved model semantic identity differs from its retained definition")
    revision = require_revision(db, definition.architecture)
    if not isinstance(revision, ArchitectureRevision) or Recipe.model_validate(revision.recipe) != definition.recipe:
        raise CatalogError("definition_mismatch", "saved definition differs from its immutable recipe")
    closure = dependency_closure(db, revision.id)
    if ({item.reference.revision_id: item.reference for item in closure}
            != {item.reference.revision_id: item.reference for item in definition.dependencies}):
        raise CatalogError("definition_mismatch", "saved definition differs from its immutable source closure")
    acceptance = _receipt(db, revision.acceptance_receipt_id)
    acceptance.require_current(acceptance.identity)
    evidence = EnqueueEvidence(architecture=definition.architecture, dependency_digest=revision.dependency_digest,
        parameter_digest=canonical_digest(definition.parameters), acceptance_receipt_id=revision.acceptance_receipt_id,
        tokenizer=definition.tokenizer, exact_receipt_id=None, final_preflight_required=True)
    return PreparedTrainingPin(evidence, dict(definition.parameters), _training_snapshot(revision, closure))


def attach_training_pin(db: Session, run: Run, prepared: PreparedTrainingPin) -> RunArchitecturePin:
    """Insert a previously prepared pin in its preparing transaction, after all chain selections validate."""
    if run.kind not in (RunKind.train, RunKind.rl):
        raise CatalogError("invalid_job_kind", "only model-producing training jobs carry architecture pins")
    evidence = prepared.evidence
    pin = RunArchitecturePin(job_id=run.id, architecture_revision_id=evidence.architecture.revision_id,
                             acceptance_receipt_id=evidence.acceptance_receipt_id, exact_receipt_id=evidence.exact_receipt_id,
                             evidence=evidence.model_dump(mode="json"), parameters=dict(prepared.parameters), snapshot=copy.deepcopy(prepared.snapshot))
    db.add(pin)
    db.flush()
    return pin


def pin_training_run(db: Session, run: Run, evidence: EnqueueEvidence, parameters: dict) -> RunArchitecturePin:
    """Validate and attach an immutable selection to an existing caller-owned training transaction."""
    if run.kind not in (RunKind.train, RunKind.rl):
        raise CatalogError("invalid_job_kind", "only model-producing training jobs carry architecture pins")
    return attach_training_pin(db, run, prepare_training_pin(db, evidence, parameters))


def enqueue_pinned_training(db: Session, *, actor_id: uuid.UUID, label: str, config: dict, evidence: EnqueueEvidence, parameters: dict, depends_on: uuid.UUID | None = None, kind: RunKind = RunKind.train) -> Run:
    """Create a queued/waiting run and its complete snapshot in one caller-owned transaction."""
    if kind not in (RunKind.train, RunKind.rl):
        raise CatalogError("invalid_job_kind", "only model-producing training jobs carry architecture pins")
    prepared = prepare_training_pin(db, evidence, parameters)
    run = queries.enqueue(db, kind=kind, label=label, config=config, owner_id=actor_id, depends_on=depends_on)
    attach_training_pin(db, run, prepared)
    return run


def record_definition(db: Session, definition: ResolvedDefinition, receipt_ids: tuple[uuid.UUID, ...]) -> ModelDefinition:
    """Deduplicate immutable semantic definitions and retain matching package evidence by foreign keys."""
    if not 1 <= len(receipt_ids) <= 64:
        raise CatalogError("missing_evidence", "definition requires bounded retained validation evidence")
    revision = require_revision(db, definition.architecture)
    if not isinstance(revision, ArchitectureRevision) or Recipe.model_validate(revision.recipe) != definition.recipe:
        raise CatalogError("definition_mismatch", "resolved recipe differs from the pinned architecture")
    closure = dependency_closure(db, revision.id)
    expected = {item.reference.revision_id: item.reference for item in closure}
    actual = {item.reference.revision_id: item.reference for item in definition.dependencies}
    if actual != expected:
        raise CatalogError("definition_mismatch", "definition does not contain the exact dependency source closure")
    digest = canonical_digest(definition)
    insert = postgres_insert if db.get_bind().dialect.name == "postgresql" else sqlite_insert
    db.execute(insert(ModelDefinition).values(id=digest, architecture_revision_id=revision.id,
                definition=definition.model_dump(mode="json"), created_at=_now()).on_conflict_do_nothing(index_elements=["id"]))
    row = db.get(ModelDefinition, digest)
    for receipt_id in receipt_ids:
        receipt = _receipt(db, receipt_id)
        receipt.require_current(receipt.identity)
        _receipt_matches_revision(receipt, revision)
        if receipt.identity.dependency_digest != revision.dependency_digest or receipt.identity.parameter_digest != canonical_digest(definition.parameters) or receipt.identity.tokenizer != definition.tokenizer or receipt.identity.context_window != definition.context_window:
            raise CatalogError("definition_mismatch", "package evidence differs from final model bindings")
        db.execute(insert(DefinitionEvidence).values(definition_id=digest, receipt_id=receipt_id)
                   .on_conflict_do_nothing(index_elements=["definition_id", "receipt_id"]))
    db.flush()
    return row


def _receipt_matches_revision(receipt: ValidationReceipt, revision: ArchitectureRevision) -> None:
    """Keep evidence for a different recipe/flow from authorizing equal-shaped model bindings."""
    subject = receipt.identity.subject
    if isinstance(subject, RevisionSubject):
        matches = subject.reference == revision_reference(revision)
    else:
        matches = subject.resource_id == revision.architecture_id and subject.content_hash == revision.content_hash
    if not matches:
        raise CatalogError("stale_evidence", "receipt source does not match the pinned architecture")


def completed_mutation(db: Session, *, actor_id: uuid.UUID, request_id: uuid.UUID,
                       operation: str, payload: dict) -> MutationOutcome | None:
    """Replay a completed operation before resolving prunable input jobs; absence never reserves a new request."""
    row = db.get(MutationRequest, (actor_id, request_id))
    if row is None:
        return None
    if row.operation != operation or row.payload_hash != canonical_digest(payload):
        raise CatalogError("request_conflict", "request UUID was reused with different arguments")
    if row.outcome is None:
        return None
    return MutationOutcome(row.outcome["kind"], uuid.UUID(row.outcome["id"]),
                           tuple(uuid.UUID(value) for value in row.outcome.get("job_ids", ())))


def idempotent_mutation(db: Session, *, actor_id: uuid.UUID, request_id: uuid.UUID, operation: str, payload: dict, execute: Callable[[], MutationOutcome]) -> MutationOutcome:
    """Serialize actor/request retries, atomically committing one matching operation/outcome."""
    dialect = db.get_bind().dialect.name
    insert = postgres_insert if dialect == "postgresql" else sqlite_insert
    digest = canonical_digest(payload)
    statement = insert(MutationRequest).values(actor_id=actor_id, request_id=request_id, operation=operation,
                                               payload_hash=digest, created_at=_now()).on_conflict_do_nothing(index_elements=["actor_id", "request_id"])
    db.execute(statement)
    row = db.scalar(select(MutationRequest).where(MutationRequest.actor_id == actor_id,
                    MutationRequest.request_id == request_id).with_for_update().execution_options(populate_existing=True))
    if row.operation != operation or row.payload_hash != digest:
        raise CatalogError("request_conflict", "request UUID was reused with different arguments")
    if row.outcome is not None:
        return MutationOutcome(row.outcome["kind"], uuid.UUID(row.outcome["id"]),
                               tuple(uuid.UUID(value) for value in row.outcome.get("job_ids", ())))
    outcome = execute()
    row.outcome = {"kind": outcome.kind, "id": str(outcome.id)}
    if outcome.job_ids:
        row.outcome = {**row.outcome, "job_ids": [str(value) for value in outcome.job_ids]}
    db.flush()
    return outcome


def prune_unreferenced_evidence(db: Session, *, limit: int = 100) -> int:
    """Delete one bounded page of unused receipts while foreign keys retain all accepted/run/package evidence."""
    if not 1 <= limit <= 100:
        raise CatalogError("invalid_page", "invalid evidence pruning budget")
    retained = select(BlockRevision.acceptance_receipt_id).union(
        select(ArchitectureRevision.acceptance_receipt_id), select(RunArchitecturePin.acceptance_receipt_id),
        select(RunArchitecturePin.exact_receipt_id).where(RunArchitecturePin.exact_receipt_id.is_not(None)),
        select(DefinitionEvidence.receipt_id), select(ModelValidation.evidence_id).where(ModelValidation.evidence_id.is_not(None)))
    candidates = list(db.scalars(select(ValidationEvidence.receipt_id).where(ValidationEvidence.receipt_id.not_in(retained)).order_by(ValidationEvidence.created_at).limit(limit).with_for_update(skip_locked=True)))
    if not candidates:
        return 0
    db.execute(delete(ValidationEvidence).where(ValidationEvidence.receipt_id.in_(candidates)))
    db.flush()
    return len(candidates)
