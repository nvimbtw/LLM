import asyncio
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
import store
from api import auth, logs
from api.routes import ROUTERS
from api.runner import Runner
from api.runs import REGISTRY
from api.handlers import HANDLERS
from store import queries as q
RUNNER = Runner(REGISTRY, HANDLERS)
# expose for routes
import api.handlers as _h
_h.RUNNER = RUNNER
import api.routes.runs as _r
# monkey patch already done via import
@asynccontextmanager
async def lifespan(_: FastAPI):
    REGISTRY.bind_loop(asyncio.get_running_loop())
    store.configure()
    with store.session() as db:
        stranded = q.fail_orphaned(db)
    if stranded:
        print(f"bench-engine: failed {stranded} run(s) stranded by a restart")
    if not auth.enforced():
        print("bench-engine: BENCH_ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")
    drain = asyncio.create_task(RUNNER.drain_forever())
    try:
        yield
    finally:
        drain.cancel()
app = FastAPI(title="Bench engine", lifespan=lifespan, dependencies=[Depends(auth.require_service)], docs_url=None, redoc_url=None, openapi_url=None)
logs.quieten()
for router in ROUTERS:
    app.include_router(router)
@app.get("/")
def health():
    active = REGISTRY.active()
    return {"status": "ok", "active_run": active.id if active else None}
