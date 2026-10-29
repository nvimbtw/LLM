import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect

import store
from api import auth
from api.routes import ROUTERS


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Alembic owns the schema, exactly as it does in the LLM engine and the data
    # engine. It used to be `create=True` here, and that is what turned a
    # database stuck on an older revision into a 500 at step time rather than a
    # failure at startup: `create_all` added the tables that were missing and
    # silently left the one that already existed under its old definition. If a
    # table is not there, the migration that should have made it did not run,
    # and that is worth hearing about.
    store.configure()
    if not auth.enforced():
        print("gyms-engine: GYMS_ENGINE_SERVICE_SECRET unset — accepting unauthenticated callers")
    yield


app = FastAPI(
    title="Gyms engine",
    lifespan=lifespan,
    dependencies=[Depends(auth.require_service)],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

for router in ROUTERS:
    app.include_router(router)


@app.websocket("/health/socket")
async def health_socket(socket: WebSocket):
    """Answer backend heartbeat frames until the authenticated connection closes."""
    await socket.accept()
    try:
        while await socket.receive_text() == "ping":
            await socket.send_text("pong")
    except WebSocketDisconnect:
        return


@app.get("/")
def health():
    return {"status": "ok"}
