"""Administrator catalog metadata/draft transport; validation and acceptance use separate execution adapters."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
import torch

import store
from api import catalog, training_jobs
from llm.architecture.builtins.catalog import copyable_source, pinned_builtins
from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import MAX_EXPANDED_CALLS, MAX_NESTING, MAX_SOURCE_BYTES, MAX_YAML_BYTES, DraftSubject, Recipe, RuntimeProfile, ShapeDomain
from llm.architecture.parse import parse_recipe
from llm.architecture.wire import (
    ArchitectureCreate, ArchitectureDraftSave, ArchitectureDraftView, ArchitectureRevisionView, BlockCreate,
    BlockDraftSave, BlockDraftView, BlockRevisionView, CatalogPage, CatalogQuery, DraftSaved, MetadataPatch,
    ArchitectureSchema, RecipeDraftInspection, ResourceCreated, ResourceDetail, RevisionPage, TrainingArchitectureRequest, TrainingArchitecturePreview,
)
from store import architecture_queries as q
from store.schema import ArchitectureRevision

router = APIRouter(tags=["architectures"],dependencies=[Depends(catalog.administrator)])


@router.get("/architectures/revisions/{revision_id}")
def selected_architecture(revision_id: UUID) -> ArchitectureRevisionView:
    """Resolve an explicitly pinned revision UUID from a preset/retry to its immutable source/metadata without a mutable latest lookup."""
    with store.session() as db:
        revision = db.get(ArchitectureRevision, revision_id)
        if revision is None:
            raise q.CatalogError("not_found", "accepted architecture revision not found")
        return catalog.revision_view(db, "architecture", revision.architecture_id, revision_id)


@router.post("/architectures/revisions/{revision_id}/training-preview")
def selected_training_preview(revision_id: UUID, body: TrainingArchitectureRequest) -> TrainingArchitecturePreview:
    """Expose only prospective data bindings/profile for this exact selection; never schedule work or import source."""
    if body.selection.revision_id != revision_id:
        raise q.CatalogError("invalid_selection", "training preview belongs to another selected revision")
    with store.session() as db:
        return training_jobs.training_preview(db, body)


def catalog_query(search: Annotated[str,Query(max_length=200)] = "",
                  origin: Literal["builtin","custom"] | None = None, archived: bool = False,
                  limit: Annotated[int,Query(ge=1,le=100)] = 50,
                  cursor: Annotated[str | None,Query(max_length=2048)] = None) -> CatalogQuery:
    """Convert HTTP query scalars once into the strict shared bounded catalog contract."""
    return CatalogQuery(search=search,origin=origin,archived=archived,limit=limit,cursor=cursor)


@router.get("/architectures/schema")
def recipe_schema() -> ArchitectureSchema:
    """Expose actual declarations/limits and prospective installed runtime defaults without constructing or validating a model."""
    profile = RuntimeProfile(torch_version=str(torch.__version__), sdk_version=1, device="cpu", precision="fp32",
        capture_policy="torch-export-strict-v1", modes=("train", "eval"),
        domains=(ShapeDomain(batch_min=1, batch_max=1, sequence_min=1, sequence_max=8, signatures=("full_prefix",)),))
    return ArchitectureSchema(schema_version=1,recipe=Recipe.model_json_schema(),limits={
        "source_bytes":MAX_SOURCE_BYTES,"yaml_bytes":MAX_YAML_BYTES,
        "expanded_calls":MAX_EXPANDED_CALLS,"nesting":MAX_NESTING},binding_syntax="${parameter}",
        default_profile=profile, available_devices=("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))


@router.post("/architectures/{resource_id}/declaration")
def recipe_declaration(resource_id: UUID, subject: DraftSubject) -> RecipeDraftInspection:
    """Parse one exact saved YAML draft on explicit request; never import dependencies or issue numerical/capture authorization."""
    if subject.resource_id != resource_id:
        raise q.CatalogError("invalid_subject", "declaration request belongs to another resource")
    with store.session() as db:
        row = q.get_resource(db, "architecture", resource_id)
        if row.draft_version != subject.draft_version or row.draft_hash != subject.content_hash:
            raise q.CatalogError("stale_version", "draft changed before declaration inspection", row.draft_version)
        source = row.draft_yaml
    try:
        recipe = parse_recipe(source)
    except ArchitectureError as error:
        raise q.CatalogError(error.code, str(error)) from error
    return RecipeDraftInspection(subject=subject, recipe=recipe)


@router.get("/blocks")
@router.get("/architectures")
def list_catalog(request: Request, query: CatalogQuery = Depends(catalog_query)) -> CatalogPage:
    """Read a bounded metadata page whose cursor cannot change catalog filters or administrative scope."""
    with store.session() as db:
        return catalog.catalog_page(db,catalog.family(request),query)


@router.post("/blocks",status_code=201)
def create_block(body: BlockCreate, actor: UUID = Depends(catalog.administrator)) -> ResourceCreated:
    """Persist arbitrary bounded Python draft bytes without importing or validating the code."""
    with store.session() as db:
        row = q.create_block(db,actor,body)
        return ResourceCreated(resource_id=row.id)


@router.post("/architectures",status_code=201)
def create_architecture(body: ArchitectureCreate, actor: UUID = Depends(catalog.administrator)) -> ResourceCreated:
    """Persist arbitrary bounded YAML draft bytes without parsing or resolving the recipe."""
    with store.session() as db:
        row = q.create_architecture(db,actor,body)
        return ResourceCreated(resource_id=row.id)


@router.get("/blocks/{resource_id}")
@router.get("/architectures/{resource_id}")
def resource_detail(request: Request, resource_id: UUID) -> ResourceDetail:
    """Return saved metadata and accepted declarations without executing custom source."""
    kind = catalog.family(request)
    with store.session() as db:
        return catalog.resource_detail(db,kind,q.get_resource(db,kind,resource_id))


@router.patch("/blocks/{resource_id}")
@router.patch("/architectures/{resource_id}")
def update_metadata(request: Request, resource_id: UUID, body: MetadataPatch) -> ResourceDetail:
    """Update optimistic metadata/archive state while refusing stale writes and read-only builtins."""
    kind = catalog.family(request)
    with store.session() as db:
        return catalog.resource_detail(db,kind,q.update_metadata(db,kind,resource_id,body))


@router.get("/blocks/{resource_id}/draft")
@router.get("/architectures/{resource_id}/draft")
def read_draft(request: Request, resource_id: UUID) -> BlockDraftView | ArchitectureDraftView:
    """Read exact saved draft bytes with their optimistic version/hash, without claiming validation."""
    kind = catalog.family(request)
    with store.session() as db:
        row = q.get_resource(db,kind,resource_id)
        identity = {"resource_id":row.id,"draft_version":row.draft_version,"content_hash":row.draft_hash}
        return BlockDraftView(**identity,source=row.draft_source) if kind == "block" else ArchitectureDraftView(**identity,yaml=row.draft_yaml)


@router.put("/blocks/{resource_id}/draft")
def save_block(resource_id: UUID, body: BlockDraftSave) -> DraftSaved:
    """Durably save exact invalid/valid Python bytes only if the expected draft version is current."""
    with store.session() as db:
        row = q.save_block_draft(db,resource_id,body)
        return DraftSaved(resource_id=row.id,draft_version=row.draft_version,content_hash=row.draft_hash)


@router.put("/architectures/{resource_id}/draft")
def save_architecture(resource_id: UUID, body: ArchitectureDraftSave) -> DraftSaved:
    """Durably save exact invalid/valid YAML bytes only if the expected draft version is current."""
    with store.session() as db:
        row = q.save_architecture_draft(db,resource_id,body)
        return DraftSaved(resource_id=row.id,draft_version=row.draft_version,content_hash=row.draft_hash)


@router.get("/blocks/{resource_id}/revisions")
@router.get("/architectures/{resource_id}/revisions")
def list_revisions(request: Request, resource_id: UUID, limit: Annotated[int,Query(ge=1,le=100)] = 50,
                   cursor: Annotated[str | None,Query(max_length=2048)] = None) -> RevisionPage:
    """Read bounded immutable acceptance history under the original resource identity."""
    with store.session() as db:
        return catalog.revision_page(db,catalog.family(request),resource_id,limit=limit,cursor=cursor)


@router.get("/blocks/{resource_id}/revisions/{revision_id}")
@router.get("/architectures/{resource_id}/revisions/{revision_id}")
def read_revision(request: Request, resource_id: UUID, revision_id: UUID) -> BlockRevisionView | ArchitectureRevisionView:
    """Read exact immutable accepted source without executing it or relying on its original validation job."""
    with store.session() as db:
        return catalog.revision_view(db,catalog.family(request),resource_id,revision_id)


@router.get("/blocks/{resource_id}/revisions/{revision_id}/copy-source")
def builtin_copy_source(resource_id: UUID, revision_id: UUID) -> BlockCreate:
    """Read an eligible custom wrapper for an exact installed builtin revision; this route creates no resource."""
    with store.session() as db:
        resource = q.get_resource(db,"block",resource_id)
        revision = q.get_catalog_revision(db,"block",resource_id,revision_id)
        if resource.origin != "builtin":
            raise q.CatalogError("not_found","copy source is available for supplied builtin revisions")
        reference = q.revision_reference(revision)
        try:
            definition = pinned_builtins((reference,))[reference.revision_id]
        except ArchitectureError as error:
            raise q.CatalogError("stale_evidence",str(error)) from error
        return BlockCreate(name=f"{definition.declaration.title} copy"[:120],description=resource.description,
                           source=copyable_source(definition.declaration.name))
