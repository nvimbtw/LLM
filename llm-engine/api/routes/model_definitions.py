"""Administrator reads of retained model definitions and immutable captured graph scopes."""

from uuid import UUID

from fastapi import APIRouter, Depends

import store
from api import catalog
from api.routes.validations import graph_query
from llm import storage
from llm.architecture.errors import ArchitectureError
from llm.architecture.package_metadata import SavedMetadata, read_saved_metadata
from llm.architecture.wire import GraphFragment, GraphQuery, ModelDefinitionSummary, RetainedSource
from store import architecture_queries as catalog_queries, queries as q

router = APIRouter(tags=["models"], dependencies=[Depends(catalog.administrator)])


def saved_metadata(model_id: UUID) -> SavedMetadata:
    """Resolve an indexed immutable model package and translate missing/corrupt metadata to bounded domain errors."""
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None:
            raise catalog_queries.CatalogError("not_found", "model is unavailable")
        definition_id = model.definition_id
    directory = storage.model_dir(str(model_id))
    if not (directory / "package.json").is_file():
        raise catalog_queries.CatalogError("not_found", "model has no retained modular definition")
    try:
        return read_saved_metadata(directory, definition_id=definition_id)
    except (OSError, ValueError) as error:
        raise catalog_queries.CatalogError("publication_missing", "saved model metadata is unavailable or invalid") from error


@router.get("/models/{model_id}/definition")
def model_definition(model_id: UUID) -> ModelDefinitionSummary:
    """Return saved semantics and receipt identities without importing code, loading tensors or consulting live catalog revisions."""
    return saved_metadata(model_id).summary()


@router.get("/models/{model_id}/sources/{revision_id}")
def model_source(model_id: UUID, revision_id: UUID) -> RetainedSource:
    """Return selected frozen source bytes without catalog lookup, custom imports or tensor loading."""
    try:
        return saved_metadata(model_id).source(revision_id)
    except ArchitectureError as error:
        raise catalog_queries.CatalogError(error.code, str(error)) from error
    except (OSError, ValueError) as error:
        raise catalog_queries.CatalogError("publication_missing", "saved model source is unavailable or invalid") from error


@router.get("/models/{model_id}/graph")
def model_graph(model_id: UUID, query: GraphQuery = Depends(graph_query)) -> GraphFragment:
    """Page only retained graph bytes for the requested model/receipt/scope; this read never launches capture."""
    try:
        return saved_metadata(model_id).graph(query)
    except ArchitectureError as error:
        raise catalog_queries.CatalogError(error.code, str(error)) from error
    except (OSError, ValueError) as error:
        raise catalog_queries.CatalogError("publication_missing", "saved model graph is unavailable or invalid") from error
