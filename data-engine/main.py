"""The data engine's HTTP surface: an app, its lifespan, and the routers it mounts.

Nothing else belongs here. A route parses a request, calls a handler or a query,
and returns the result — if it grows past a few lines the logic belongs in
`hub/`, `processors/` or `store/`.

This service talks to the Rust backend and to the Hugging Face Hub. It does not
know the LLM engine exists, and must not learn: the Python services are
siblings that share a caller, not a pipeline.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

import store
from api import auth
from api.routes import ROUTERS
from api.runner import RUNNER
from api.runs import REGISTRY
from store import queries as q


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Worker threads publish progress onto this loop; see Registry.bind_loop.
    REGISTRY.bind_loop(asyncio.get_running_loop())
    store.configure()

    # Runs still marked live belong to a process that is gone. Queued rows are
    # deliberately untouched — draining them is the point of a durable queue.
    with store.session() as db:
        stranded = q.fail_orphaned(db)
    if stranded:
        print(f"data engine: failed {stranded} run(s) stranded by a restart")

    if not auth.enforced():
        print("data engine: DATA_ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")

    drain = asyncio.create_task(RUNNER.drain_forever())
    try:
        yield
    finally:
        drain.cancel()


# The dependency is on the app rather than on each route: a new endpoint is
# authenticated because it exists, not because someone remembered to decorate it.
app = FastAPI(
    title="Data engine",
    lifespan=lifespan,
    dependencies=[Depends(auth.require_service)],
    # FastAPI's own doc routes bypass the dependency above, which would leave the
    # whole API surface enumerable by anything that could reach the port.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

for router in ROUTERS:
    app.include_router(router)


@app.get("/")
def health():
    active = REGISTRY.active()
    return {"status": "ok", "active_run": active.id if active else None}
