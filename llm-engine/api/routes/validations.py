"""Administrator durable validation submission, published reports/graphs and exact revision acceptance."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

import store
from api import catalog, validation_jobs
from api.runner import RUNNER
from api.routes.runs import row
from llm.architecture.builtins.catalog import compatibility_manifest, pinned_builtins
from llm.architecture.errors import ArchitectureError
from llm.architecture.graphs import read_fragment
from llm.architecture.parse import parse_recipe
from llm.architecture.spec import FinalTokenizer, Identifier, canonical_digest
from llm.architecture.wire import AcceptedRevision, AcceptRevision, GraphFragment, GraphQuery, RetainedSource, ValidationReport, ValidationRequest, ValidationRun
from store import architecture_queries as q

router = APIRouter(tags=["validations"],dependencies=[Depends(catalog.administrator)])


@router.get("/tokenizers/{tokenizer_id}/architecture-binding")
def tokenizer_binding(tokenizer_id: UUID, reserved: Annotated[tuple[Identifier,...],Query(max_length=128)] = ()) -> FinalTokenizer:
    """Expose full-byte tokenizer identity/vocabulary for architecture selection without loading model weights."""
    return validation_jobs.read_tokenizer_binding(tokenizer_id,reserved)


@router.post("/blocks/{resource_id}/validations",status_code=202)
@router.post("/architectures/{resource_id}/validations",status_code=202)
def submit_validation(request: Request, resource_id: UUID, body: ValidationRequest,
                      actor: UUID = Depends(catalog.administrator)) -> ValidationRun:
    """Queue one actor-idempotent immutable source/profile snapshot before waking the existing runner."""
    with store.session() as db:
        run = validation_jobs.enqueue_validation(db,catalog.family(request),resource_id,body,actor)
        response = ValidationRun.model_validate(row(run))
    RUNNER.wake()
    return response


@router.get("/validations/{run_id}")
def validation_report(run_id: UUID) -> ValidationReport:
    """Return only a completed published report, including retained evidence after source history pruning."""
    with store.session() as db:
        return validation_jobs.published_result(db,run_id).output.report


@router.get("/validations/{run_id}/sources/{revision_id}")
def validation_source(run_id: UUID, revision_id: UUID) -> RetainedSource:
    """Read frozen published source independently of current drafts; incomplete/failed worker publications remain unavailable."""
    with store.session() as db:
        publication = validation_jobs.published_result(db, run_id)
    try:
        return publication.source(revision_id)
    except ArchitectureError as error:
        raise q.CatalogError(error.code, str(error)) from error


def graph_query(identity_hash: Annotated[str,Query(pattern=r"^[0-9a-f]{64}$")],
                view: Literal["architecture","operations"],mode: Literal["train","eval"],
                signature: Literal["full_prefix","prefill","decode"],
                scope: Annotated[str,Query(min_length=1,max_length=512)] = "root",
                cursor: Annotated[str | None,Query(max_length=2048)] = None,
                limit: Annotated[int,Query(ge=1,le=256)] = 256) -> GraphQuery:
    """Interpret bounded HTTP graph selection once; reads never launch capture or construct models."""
    return GraphQuery(identity_hash=identity_hash,view=view,mode=mode,signature=signature,scope=scope,cursor=cursor,limit=limit)


@router.get("/validations/{run_id}/graph")
def validation_graph(run_id: UUID, query: GraphQuery = Depends(graph_query)) -> GraphFragment:
    """Page exact published receipt-bound graph scopes, refusing stale identities and cursor replay."""
    with store.session() as db:
        output = validation_jobs.published_result(db,run_id).output
    receipt = output.report.receipt
    if receipt is None or receipt.verdict != "valid":
        raise q.CatalogError("invalid_evidence","validation has no complete captured graph")
    try:
        return read_fragment(receipt.graph_manifest_hash,output.graph_blobs,query)
    except ArchitectureError as error:
        raise q.CatalogError(error.code,str(error)) from error


@router.post("/blocks/{resource_id}/revisions",status_code=201)
@router.post("/architectures/{resource_id}/revisions",status_code=201)
def accept_revision(request: Request, resource_id: UUID, body: AcceptRevision) -> AcceptedRevision:
    """Accept only the exact current draft and dependency snapshot with fully valid published retained evidence."""
    kind = catalog.family(request)
    with store.session() as db:
        publication = validation_jobs.published_result(db,body.validation_run_id)
        try:
            pinned_builtins(publication.snapshot.builtins)
        except ArchitectureError as error:
            raise q.CatalogError("stale_evidence",str(error)) from error
        report = publication.output.report
        if report.verdict != "valid" or (publication.snapshot.source is not None) != (kind == "block"):
            raise q.CatalogError("invalid_evidence","only fully valid matching resource evidence can accept a revision")
        if kind == "block":
            if report.declaration is None:
                raise q.CatalogError("invalid_evidence","validated block declaration is missing")
            accepted = q.accept_block(db,resource_id,body,report.declaration,
                compatibility={"sdk_version":1,"builtin_compatibility":canonical_digest(compatibility_manifest())})
        else:
            accepted = q.accept_architecture(db,resource_id,body,parse_recipe(publication.snapshot.recipe))
        return catalog.accepted_revision(db,accepted)
