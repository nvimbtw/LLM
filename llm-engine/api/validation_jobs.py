"""Parent adapters freezing validation inputs and retaining results in the existing durable queue."""

from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from llm import storage
from llm.architecture.errors import ArchitectureError
from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest
from llm.architecture.parse import parse_recipe
from llm.architecture.spec import DraftSubject, FinalTokenizer, RevisionRef, TokenizerBinding, canonical_digest
from llm.architecture.source import imports_public_builtins
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES, binding_from_bytes
from llm.architecture.package_integrity import MANIFEST_BYTES, decode_json, package_file, read_bytes
from llm.architecture.validation_artifacts import PublishedValidation, read_publication
from llm.architecture.validation_records import PinnedSource, ValidationSnapshot
from llm.architecture.wire import ValidationRequest
from llm.architecture.wire import AcceptRevision
from store import architecture_queries as catalog, queries
from store.schema import ArchitectureRevision, BlockRevision, Run, RunKind, RunStatus

SNAPSHOT_BYTES = 8 << 20


def read_tokenizer_binding(tokenizer_id: UUID, reserved_names: tuple[str,...] = ()) -> FinalTokenizer:
    """Read bounded authoritative facts for a complete build, refusing missing files and inconsistent sidecars."""
    directory = storage.tokenizer_build_dir(str(tokenizer_id))
    try:
        metadata = decode_json(read_bytes(package_file(directory,"build.json"),MANIFEST_BYTES))
        raw = read_bytes(package_file(directory,"tokenizer.json"),TOKENIZER_BYTES)
        actual = binding_from_bytes(tokenizer_id,raw,reserved_names)
        if not isinstance(metadata,dict) or metadata.get("vocab_size") != actual.vocabulary_size:
            raise ValueError("tokenizer facts differ from the complete owned build")
        return actual
    except FileNotFoundError as error:
        raise catalog.CatalogError("not_found","complete tokenizer build not found") from error
    except (OSError,ValueError) as error:
        raise catalog.CatalogError("tokenizer_binding","tokenizer build bytes or metadata are inconsistent") from error


def verify_tokenizer(binding: TokenizerBinding | None) -> None:
    """Check client final facts against a complete owned build; pending and absent bindings carry no final authority."""
    if not isinstance(binding,FinalTokenizer):
        return
    try:
        actual = read_tokenizer_binding(binding.tokenizer_id,tuple(binding.reserved_token_ids))
    except catalog.CatalogError as error:
        raise catalog.CatalogError("tokenizer_binding","final tokenizer facts must match a complete owned build") from error
    if actual != binding:
        raise catalog.CatalogError("tokenizer_binding","final tokenizer facts differ from the complete owned build")


def subject_bytes(db: Session, kind: catalog.ResourceKind, resource_id: UUID, request: ValidationRequest) -> str:
    """Require an exact saved draft/revision under its resource identity before freezing source bytes."""
    resource = catalog.validation_resource(db,kind,resource_id)
    subject = request.subject
    if isinstance(subject,DraftSubject):
        if subject.resource_id != resource_id or subject.draft_version != resource.draft_version or subject.content_hash != resource.draft_hash:
            raise catalog.CatalogError("stale_version","validation subject differs from the current saved draft",resource.draft_version)
        return resource.draft_source if kind == "block" else resource.draft_yaml
    if subject.reference.kind != kind or subject.reference.resource_id != resource_id:
        raise catalog.CatalogError("not_found","validation revision belongs to another catalog resource")
    revision = catalog.require_revision(db,subject.reference)
    return revision.source if kind == "block" else revision.yaml


def frozen_dependencies(db: Session, source: str) -> tuple[tuple[PinnedSource,...],tuple[RevisionRef,...]]:
    """Freeze transitive immutable custom bytes and installed builtin pins without importing custom source."""
    try:
        recipe = parse_recipe(source)
    except ArchitectureError:
        return (),()
    pending = list(catalog.recipe_dependencies(recipe).values())
    sources, builtins, seen = [], [], set()
    while pending:
        reference = pending.pop()
        if reference.revision_id in seen:
            continue
        if len(seen) >= 128:
            raise catalog.CatalogError("dependency_limit","validation dependency closure exceeds budget")
        seen.add(reference.revision_id)
        row = catalog.require_revision(db,reference)
        if isinstance(row,BlockRevision) and catalog.get_resource(db,"block",row.block_id).origin == "builtin":
            builtins.append(reference)
        else:
            value = row.source if isinstance(row,BlockRevision) else row.yaml
            if isinstance(row,BlockRevision) and imports_public_builtins(value):
                if row.compatibility.get("builtin_compatibility") != canonical_digest(compatibility_manifest()):
                    raise catalog.CatalogError("stale_evidence","custom block builtin compatibility differs from installed runtime")
                builtins.extend(builtin.reference for builtin in builtin_definitions().values())
            sources.append(PinnedSource(reference=reference,source=value))
            if isinstance(row,ArchitectureRevision):
                pending.extend(catalog.recipe_dependencies(parse_recipe(row.yaml)).values())
    return tuple(sources),tuple({value.revision_id:value for value in builtins}.values())


def freeze_request(db: Session, kind: catalog.ResourceKind, resource_id: UUID,
                   request: ValidationRequest, run_id: UUID) -> ValidationSnapshot:
    """Capture source closure before queue commit; saved invalid syntax still reaches the bounded validation worker."""
    if request.mode == "inspect" and kind != "block":
        raise catalog.CatalogError("inspection_subject","declaration inspection requires a block")
    if request.profile is not None and request.profile.device not in {"cpu","cuda"}:
        raise catalog.CatalogError("runtime_profile","validation device must be cpu or cuda")
    verify_tokenizer(request.tokenizer)
    source = subject_bytes(db,kind,resource_id,request)
    resource = catalog.get_resource(db,kind,resource_id)
    builtin_root = None
    if kind == "block" and resource.origin == "builtin":
        builtin_root = next((value.reference for value in builtin_definitions().values()
                             if value.reference.resource_id == resource_id),None)
        if builtin_root is None:
            raise catalog.CatalogError("stale_evidence","builtin implementation is unavailable in this installation")
        if request.subject.kind == "revision" and request.subject.reference != builtin_root:
            raise catalog.CatalogError("stale_evidence","historical builtin implementation is unavailable for new validation")
    dependencies, builtins = frozen_dependencies(db,source) if kind == "architecture" else ((),())
    if kind == "block" and imports_public_builtins(source):
        builtins = tuple(value.reference for value in builtin_definitions().values())
    snapshot = ValidationSnapshot(run_id=run_id,request=request,source=source if kind == "block" else None,
                                  recipe=source if kind == "architecture" else None,dependencies=dependencies,builtins=builtins,
                                  builtin_root=builtin_root)
    limit = 256 << 10 if request.mode == "inspect" else SNAPSHOT_BYTES
    if len(snapshot.model_dump_json().encode()) > limit:
        raise catalog.CatalogError("snapshot_byte_limit","serialized validation snapshot exceeds worker artifact budget")
    return snapshot


def enqueue_validation(db: Session, kind: catalog.ResourceKind, resource_id: UUID,
                       request: ValidationRequest, actor_id: UUID) -> Run:
    """Atomically deduplicate actor/request submission and queue exact immutable source/configuration bytes."""
    def enqueue() -> catalog.MutationOutcome:
        """Freeze the selected saved bytes and queue their single validation job in this transaction."""
        run_id = uuid4()
        snapshot = freeze_request(db,kind,resource_id,request,run_id)
        resource = catalog.get_resource(db,kind,resource_id)
        queries.enqueue(db,kind=RunKind.model_validation,label=f"Validate {resource.name}"[:120],
                        config=snapshot.model_dump(mode="json"),owner_id=actor_id,id=run_id)
        return catalog.MutationOutcome("validation",run_id)
    outcome = catalog.idempotent_mutation(db,actor_id=actor_id,request_id=request.request_id,
        operation="validation",payload={"kind":kind,"resource_id":str(resource_id),"request":request.model_dump(mode="json")},execute=enqueue)
    row = db.get(Run,outcome.id)
    if row is None:
        raise catalog.CatalogError("operation_pruned",f"original validation {outcome.id} was pruned; use a fresh request UUID")
    return row


def retry_validation(db: Session, original: Run, actor_id: UUID, request_id: UUID | None = None) -> Run:
    """Create a fresh frozen-source attempt, optionally deduplicating generic run-retry requests under their existing durable operation contract."""
    snapshot = ValidationSnapshot.model_validate(original.config)
    request = snapshot.request.model_copy(update={"request_id":request_id or uuid4()})
    snapshot = snapshot.model_copy(update={"run_id":uuid4(),"request":request})
    def enqueue() -> catalog.MutationOutcome:
        """Persist the copied immutable source closure under the new retry identity."""
        queries.enqueue(db,kind=RunKind.model_validation,label=original.label,owner_id=actor_id,
                        id=snapshot.run_id,config=snapshot.model_dump(mode="json"))
        return catalog.MutationOutcome("validation",snapshot.run_id)
    outcome = catalog.idempotent_mutation(db,actor_id=actor_id,request_id=request.request_id,
        operation="training" if request_id is not None else "validation",
        payload={"retry_of":str(original.id)} if request_id is not None else
            {"retry_of":str(original.id),"snapshot":snapshot.model_dump(mode="json")},execute=enqueue)
    result = db.get(Run, outcome.id)
    if result is None:
        raise catalog.CatalogError("operation_pruned", "original retry history was pruned; use a fresh request UUID")
    return result


def complete_validation(db: Session, run: Run, result: dict) -> PublishedValidation:
    """Verify published bytes and retain evidence in the same transaction as the completed job status."""
    publication = read_publication(run.id)
    if publication.snapshot != ValidationSnapshot.model_validate(run.config) or publication.output.report.model_dump(mode="json") != result:
        raise ValueError("published validation differs from its durable job snapshot/result")
    if run.status != RunStatus.done:
        raise ValueError("validation evidence requires the final completed job transaction")
    receipt = publication.output.report.receipt
    if receipt is not None and receipt.verdict == "valid":
        catalog.record_evidence(db,run.id,receipt)
        reference = publication.snapshot.builtin_root
        subject = publication.snapshot.request.subject
        if reference is not None and subject.kind == "draft":
            catalog.accept_block(db,reference.resource_id,AcceptRevision(expected_draft_version=subject.draft_version,
                expected_content_hash=subject.content_hash,expected_dependency_digest=receipt.identity.dependency_digest,
                validation_run_id=run.id),publication.output.report.declaration,
                compatibility={"sdk_version":1,"builtin_compatibility":canonical_digest(compatibility_manifest())},
                builtin_reference=reference)
        elif publication.snapshot.recipe is not None and subject.kind == "draft":
            resource = catalog.get_resource(db,"architecture",subject.resource_id)
            if resource.origin == "builtin":
                catalog.accept_architecture(db,resource.id,AcceptRevision(expected_draft_version=subject.draft_version,
                    expected_content_hash=subject.content_hash,expected_dependency_digest=receipt.identity.dependency_digest,
                    validation_run_id=run.id),parse_recipe(publication.snapshot.recipe),internal_builtin=True)
    return publication


def published_result(db: Session, run_id: UUID) -> PublishedValidation:
    """Require terminal completed authority or retained evidence before exposing immutable report/graph bytes."""
    run = db.get(Run,run_id)
    retained = catalog.evidence_for_job(db,run_id)
    if run is None and retained is None or run is not None and run.kind != RunKind.model_validation:
        raise catalog.CatalogError("not_found","validation not found")
    if run is not None and run.status != RunStatus.done:
        code = "validation_failed" if run.status == RunStatus.failed else "validation_incomplete"
        raise catalog.CatalogError(code,(run.error or "validation has not completed successfully")[:2000])
    try:
        publication = read_publication(run_id)
    except (OSError,ValueError) as error:
        raise catalog.CatalogError("publication_missing","complete validation artifacts are unavailable or corrupt") from error
    if retained is not None and (publication.output.report.receipt is None or retained.receipt != publication.output.report.receipt.model_dump(mode="json")):
        raise catalog.CatalogError("publication_missing","published validation differs from retained evidence")
    return publication
