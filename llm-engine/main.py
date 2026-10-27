"""The LLM engine's HTTP surface: an app, its lifespan, and the routers it mounts.

Nothing else belongs here. A route parses a request, calls an orchestrator or a
query, and returns the result — if it grows past a few lines the logic belongs
in `llm/` or in `store/`.
"""

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError

import store
from api import auth, builtin_catalog, catalog, logs, recovery
from api.routes import ROUTERS
from api.runner import RUNNER
from api.runs import REGISTRY
from api.serving import SERVING
from store import queries as q
from store.architecture_queries import CatalogError
from llm import storage
from llm.runtime.supervisor import WorkerManager


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Reconcile worker ownership before queue admission and drain process owners during shutdown."""
    REGISTRY.bind_loop(asyncio.get_running_loop())
    store.configure()
    workers = WorkerManager(storage.runtime_root())
    await asyncio.to_thread(workers.start)
    drain = None
    try:
        RUNNER.attach_workers(workers)
        SERVING.attach_workers(workers)
        recovery.reconcile()
        with store.session() as db:
            stranded = q.fail_orphaned(db)
        if stranded:
            print(f"llm-engine: failed {stranded} run(s) stranded by a restart")
        builtin_catalog.seed_blocks()
        builtin_catalog.seed_recipes()
        if not auth.enforced():
            print("llm-engine: LLM_ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")
        drain = asyncio.create_task(RUNNER.drain_forever())
        yield
    finally:
        try:
            await RUNNER.shutdown()
        finally:
            try:
                if drain is not None:
                    drain.cancel()
                    with suppress(asyncio.CancelledError):
                        await drain
            finally:
                await asyncio.to_thread(workers.close)


# The dependency is on the app rather than on each route: a new endpoint is
# authenticated because it exists, not because someone remembered to decorate
# it. A WebSocket upgrade is an HTTP request, so the socket routes are covered
# by the same check — an unsigned handshake is refused before the socket exists.
app = FastAPI(
    title="LLM engine",
    lifespan=lifespan,
    dependencies=[Depends(auth.require_service)],
    # FastAPI serves /docs, /redoc and /openapi.json itself, and its own routes
    # bypass the dependency above — so the whole API surface was enumerable by
    # anything that could reach the port. The LLM engine has one client and it calls
    # typed methods; the schema is still reachable in-process via app.openapi()
    # for anything that needs to generate against it.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

logs.quieten()
app.add_exception_handler(CatalogError,catalog.http_error)
app.add_exception_handler(RequestValidationError,catalog.request_error)

for router in ROUTERS:
    app.include_router(router)


@app.get("/")
def health(administration: bool = Depends(auth.administration_visible)):
    """Report liveness while keeping private validation identities out of public health metadata."""
    active = REGISTRY.active()
    visible = active is not None and (administration or active.kind != "model_validation")
    return {"status": "ok", "active_run": active.id if visible else None}
