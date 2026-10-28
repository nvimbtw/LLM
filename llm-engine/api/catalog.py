"""Data-only catalog projections and scoped cursors; custom source execution belongs to workers."""

import base64
from uuid import UUID

from fastapi import Depends, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api import auth
from llm.architecture.spec import BlockDeclaration, Contract, PositiveInt, Recipe, Sha256, canonical_digest
from llm.architecture.wire import (
    AcceptedRevision, ArchitectureRevisionView, BlockRevisionView, CatalogPage, CatalogQuery,
    Diagnostic, ErrorResponse, ResourceDetail, ResourceSummary, RevisionPage,
)
from store import architecture_queries as q
from store.schema import ArchitectureRevision, BlockRevision, ValidationEvidence


class CatalogCursor(Contract):
    """Bind a UUID keyset position to its catalog family, filters and administrative visibility."""

    after: UUID
    filters: Sha256


class RevisionCursor(Contract):
    """Bind revision-number pagination to one immutable resource family and identity."""

    after: PositiveInt
    kind: q.ResourceKind
    resource_id: UUID


def administrator(actor: UUID = Depends(auth.require_owner), visible: bool = Depends(auth.administration_visible)) -> UUID:
    """Require Rust-authorized administrative actor identity before any catalog/source read or write."""
    if not visible:
        raise q.CatalogError("forbidden", "catalog authoring requires administrative visibility")
    return actor


def family(request: Request) -> q.ResourceKind:
    """Select the two registered literal catalog roots, never a client-selected database identifier."""
    return "block" if request.url.path.split("/")[1] == "blocks" else "architecture"


def filter_digest(kind: q.ResourceKind, query: CatalogQuery) -> str:
    """Identify stable catalog filters independently of page size and keyset position."""
    return canonical_digest({"kind":kind,"search":query.search,"origin":query.origin,
                             "archived":query.archived,"visibility":"administration"})


def cursor_after(kind: q.ResourceKind, query: CatalogQuery) -> UUID | None:
    """Decode a bounded cursor and refuse replay against another family/filter set."""
    if query.cursor is None:
        return None
    try:
        cursor = CatalogCursor.model_validate_json(base64.b64decode(query.cursor,altchars=b"-_",validate=True))
    except ValueError as error:
        raise q.CatalogError("invalid_cursor", "invalid catalog cursor") from error
    if cursor.filters != filter_digest(kind,query):
        raise q.CatalogError("invalid_cursor", "catalog cursor belongs to another filter or family")
    return cursor.after


def resource_summary(db: Session, kind: q.ResourceKind, row: q.CatalogResource) -> ResourceSummary:
    """Project bounded metadata and immutable latest accepted identity without parsing editable drafts."""
    latest = q.revision_rows(db,kind,row.id,limit=1,latest=True)
    return ResourceSummary(resource_id=row.id,kind=kind,name=row.name,description=row.description,
        origin=row.origin,archived=row.archived,metadata_version=row.metadata_version,draft_version=row.draft_version,
        accepted_revision=q.revision_reference(latest[0]) if latest else None)


def resource_detail(db: Session, kind: q.ResourceKind, row: q.CatalogResource) -> ResourceDetail:
    """Expose declarations only from accepted immutable data, never by importing an editable source draft."""
    latest = q.revision_rows(db,kind,row.id,limit=1,latest=True)
    declaration = BlockDeclaration.model_validate(latest[0].declaration) if latest and kind == "block" else None
    parameters = Recipe.model_validate(latest[0].recipe).parameters if latest and kind == "architecture" else {}
    return ResourceDetail(summary=resource_summary(db,kind,row),revision_count=q.revision_count(db,kind,row.id),
                          declaration=declaration,parameters=parameters)


def catalog_page(db: Session, kind: q.ResourceKind, query: CatalogQuery) -> CatalogPage:
    """Read a bounded source-free response with keyset continuation scoped to its original filters."""
    filters = {"archived":query.archived,"origin":query.origin,"search":query.search}
    rows = q.list_resources(db,kind,**filters,after=cursor_after(kind,query),limit=query.limit)
    next_cursor = None
    if rows and q.list_resources(db,kind,**filters,after=rows[-1].id,limit=1):
        cursor = CatalogCursor(after=rows[-1].id,filters=filter_digest(kind,query))
        next_cursor = base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode()
    return CatalogPage(items=tuple(resource_summary(db,kind,row) for row in rows),next_cursor=next_cursor)


def accepted_revision(db: Session, row: BlockRevision | ArchitectureRevision) -> AcceptedRevision:
    """Expose immutable acceptance identity from retained evidence even after source job history is pruned."""
    evidence = db.get(ValidationEvidence,row.acceptance_receipt_id)
    if evidence is None:
        raise q.CatalogError("missing_evidence", "accepted revision evidence is unavailable")
    return AcceptedRevision(reference=q.revision_reference(row),revision_number=row.revision_number,
        acceptance_receipt_id=row.acceptance_receipt_id,acceptance_identity_hash=evidence.identity_hash)


def revision_page(db: Session, kind: q.ResourceKind, resource_id: UUID, *, limit: int,
                  cursor: str | None) -> RevisionPage:
    """Page acceptance summaries without including immutable source bodies or consulting prunable job rows."""
    q.get_resource(db,kind,resource_id)
    after = 0
    if cursor:
        try:
            decoded = RevisionCursor.model_validate_json(base64.b64decode(cursor,altchars=b"-_",validate=True))
        except ValueError as error:
            raise q.CatalogError("invalid_cursor", "invalid revision cursor") from error
        if decoded.kind != kind or decoded.resource_id != resource_id:
            raise q.CatalogError("invalid_cursor", "revision cursor belongs to another resource")
        after = decoded.after
    rows = q.revision_rows(db,kind,resource_id,after=after,limit=limit)
    next_cursor = None
    if rows and q.revision_rows(db,kind,resource_id,after=rows[-1].revision_number,limit=1):
        value = RevisionCursor(after=rows[-1].revision_number,kind=kind,resource_id=resource_id)
        next_cursor = base64.urlsafe_b64encode(value.model_dump_json().encode()).decode()
    return RevisionPage(items=tuple(accepted_revision(db,row) for row in rows),next_cursor=next_cursor)


def revision_view(db: Session, kind: q.ResourceKind, resource_id: UUID,
                  revision_id: UUID) -> BlockRevisionView | ArchitectureRevisionView:
    """Return exact accepted source/recipe metadata without importing code or resolving mutable latest references."""
    row = q.get_catalog_revision(db,kind,resource_id,revision_id)
    accepted = accepted_revision(db,row)
    if kind == "block":
        declaration = BlockDeclaration.model_validate(row.declaration)
        return BlockRevisionView(accepted=accepted,source=row.source,declaration=declaration,sdk_version=declaration.sdk_version)
    recipe = Recipe.model_validate(row.recipe)
    dependencies = {value.revision_id:value for value in q.recipe_dependencies(recipe).values()}
    return ArchitectureRevisionView(accepted=accepted,yaml=row.yaml,recipe=recipe,
        dependencies=tuple(dependencies.values()),dependency_digest=row.dependency_digest)


def http_error(request: Request, error: q.CatalogError) -> JSONResponse:
    """Translate bounded domain refusals to the shared error envelope without exposing query/traceback internals."""
    status = {"forbidden":403,"read_only":403,"not_found":404,"missing_evidence":409,
              "stale_version":409,"stale_evidence":409,"archived":409,"identity_conflict":409,
              "request_conflict":409,"operation_pruned":404,"snapshot_byte_limit":413,
              "invalid_evidence":409,"validation_failed":409,"validation_incomplete":409,
              "publication_missing":409,"graph_identity":409}.get(error.code,422)
    phase = ("resolution" if request.url.path.endswith("/training-preview") else "inspection" if request.url.path.endswith("/declaration") else "publication" if error.code == "publication_missing" else "resolution" if error.code == "tokenizer_binding"
             else "execution" if error.code.startswith("validation_") else "save")
    body = ErrorResponse(error=Diagnostic(code=error.code,message=str(error),phase=phase),current_version=error.current_version)
    return JSONResponse(status_code=status,content=body.model_dump(mode="json",exclude_none=True))


async def request_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """Return bounded authoring diagnostics for invalid HTTP values while preserving unrelated route behavior."""
    authoring = request.url.path.split("/")[1] in {"blocks","architectures","validations"}
    if not authoring and not request.url.path.endswith("/architecture-binding"):
        return await request_validation_exception_handler(request,error)
    errors = error.errors()
    oversized = any("source text byte budget exceeded" in value["msg"] for value in errors)
    message = "source text byte budget exceeded" if oversized else errors[0]["msg"][:2000]
    phase = "resolution" if request.url.path.endswith("/training-preview") else "inspection" if request.url.path.endswith("/declaration") else "save"
    body = ErrorResponse(error=Diagnostic(code="source_byte_limit" if oversized else "invalid_request",message=message,phase=phase))
    return JSONResponse(status_code=413 if oversized else 422,content=body.model_dump(mode="json",exclude_none=True))
