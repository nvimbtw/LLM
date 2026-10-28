"""Serve real engine authorization/routes over disposable storage for the Rust seam test."""

import json
import os
from pathlib import Path
import socket
import sys
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI


def fixture_stream(*_args, **_kwargs):
    """Emit one synthetic completed reply after real publication admission without loading numerical models."""
    yield "token", {"token": "fixture answer"}
    return {"text": "fixture answer", "generated_tokens": 1}


def serve() -> None:
    """Configure isolated SQLite/artifacts, seed source-bearing rows and serve without engine lifespan."""
    root, engine = map(Path, sys.argv[1:])
    os.environ["DATA_DIR"] = str(root / "artifacts")
    sys.path.insert(0, str(engine))
    import store
    from api import auth
    from api.routes import inference, models, runs
    from api.runner import RUNNER
    from api.runs import Event, REGISTRY
    from store import queries as q
    from store.schema import RunKind, RunStatus

    store.configure(f"sqlite+pysqlite:///{root / 'store.sqlite3'}", create=True)
    RUNNER.set_paused(True)
    source = "PRIVATE_SOURCE_GRAPH_MARKER"
    ids = {}
    with store.session() as db:
        for index in range(3):
            row = q.enqueue(db, kind=RunKind.train, label="public job", config={"source": source}, status=RunStatus.done)
            row.result = {"graph": source}
            row.error = source
            ids[f"public_{index}"] = str(row.id)
        for index in range(4):
            row = q.enqueue(db, kind=RunKind.model_validation, label="private validation", config={"source": source}, status=RunStatus.done)
            ids[f"validation_{index}"] = str(row.id)
        q.enqueue(db, kind=RunKind.model_validation, label="private active", config={"source": source}, status=RunStatus.queued)
        model = q.upsert_model(db, model_id=uuid4(), name="published", published=True, context=128,
            parameters={"config": {"source": source}, "result": {"graph": source, "parameters": 123}})
        ids["model"] = str(model.id)
        checkpoint = q.upsert_model(db, model_id=uuid4(), name="checkpoint", parent_id=model.id, checkpoint_index=1,
            parameters={"source": source})
        ids["private_checkpoint"] = str(checkpoint.id)
        private = q.upsert_model(db, model_id=uuid4(), name="private model", parameters={"source": source})
        ids["private_model"] = str(private.id)
    for name, kind in [("validation_0", "model_validation"), ("public_0", "train")]:
        live = REGISTRY.adopt(ids[name], kind, {"source": source})
        REGISTRY.publish(live, Event("log", {"line": source}))
        REGISTRY.publish(live, Event("train", {"loss": 1.25, "source": source}))
        REGISTRY.finish(live, "failed", error=source)
    app = FastAPI(dependencies=[Depends(auth.require_service)])
    app.include_router(runs.router)
    app.include_router(models.router)
    app.include_router(inference.router)
    inference.chat_pipeline.validate_context_tokenizer = lambda *_: None
    inference.chat_pipeline.count_tokens = lambda *_: {"tokens": 1}
    inference.serving.run = lambda *_args, **_kwargs: {"text": "fixture answer", "generated_tokens": 1}
    inference.serving.stream = fixture_stream
    connection = socket.socket()
    connection.bind(("127.0.0.1", 0))
    print(json.dumps({"url": f"http://127.0.0.1:{connection.getsockname()[1]}", **ids}), flush=True)
    uvicorn.run(app, fd=connection.fileno(), log_level="warning")


if __name__ == "__main__":
    serve()
