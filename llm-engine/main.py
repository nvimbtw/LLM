"""The LLM engine's HTTP surface: an app, its lifespan, and the routers it mounts.

Nothing else belongs here. A route parses a request, calls an orchestrator or a
query, and returns the result — if it grows past a few lines the logic belongs
in `llm/` or in `store/`.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

import store
from api import auth, logs, recovery
from api.routes import ROUTERS
from api.runner import RUNNER
from api.runs import REGISTRY
from store import queries as q


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Worker threads publish progress onto this loop; see Registry.bind_loop.
    REGISTRY.bind_loop(asyncio.get_running_loop())
    store.configure()
    recovery.reconcile()

    # Runs still marked live belong to a process that is gone. Queued rows are
    # deliberately untouched — draining them is the point of a durable queue.
    with store.session() as db:
        stranded = q.fail_orphaned(db)
    if stranded:
        print(f"llm-engine: failed {stranded} run(s) stranded by a restart")

    if not auth.enforced():
        print("llm-engine: LLM_ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")

    drain = asyncio.create_task(RUNNER.drain_forever())
    try:
        yield
    finally:
        drain.cancel()


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

for router in ROUTERS:
    app.include_router(router)


@app.get("/")
def health():
    active = REGISTRY.active()
    return {"status": "ok", "active_run": active.id if active else None}
