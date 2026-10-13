"""Benchmark ownership and scoring, independent of model storage."""
import threading
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

import auth
import store
from bench import runner
from bench.backend import Hop

_workers = ThreadPoolExecutor(max_workers=1,thread_name_prefix='bench')
_active: dict[uuid.UUID,threading.Event] = {}
_guard = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    store.configure()
    with store.session() as db:
        for row in db.scalars(select(store.Benchmark).where(store.Benchmark.status=='running')):
            row.status,row.error,row.completed_at = 'failed','bench-engine restarted during measurement',store.now()
    if not auth.enforced():
        print('bench-engine service calls are UNAUTHENTICATED',flush=True)
    yield
    with _guard:
        for event in _active.values():
            event.set()
    _workers.shutdown(wait=False,cancel_futures=True)


app = FastAPI(lifespan=lifespan,dependencies=[Depends(auth.require_service)], docs_url=None, redoc_url=None, openapi_url=None)


@app.get('/')
@app.get('/health')
def health():
    return {'status':'ok'}


class Evaluation(BaseModel):
    model_name: uuid.UUID
    suite: str = 'language'
    max_tokens: int = Field(default=96,ge=1,le=4096)
    temperature: float = Field(default=0.8,ge=0)
    top_k: int = Field(default=40,ge=0)
    top_p: float = Field(default=0.95,gt=0,le=1)
    seed: int = 1234


@app.post('/execute/{benchmark_id}')
def execute(benchmark_id: uuid.UUID, body: Evaluation, owner: str = Depends(auth.require_service)):
    if body.suite != 'language':
        raise HTTPException(400,'only the English language suite is active; historical suites remain readable')
    with _guard, store.session() as db:
        existing = db.get(store.Benchmark,benchmark_id)
        if existing:
            return store.describe(existing,True)
        row = store.Benchmark(id=benchmark_id,model_id=body.model_name,model_name=str(body.model_name),config=body.model_dump(mode='json'))
        db.add(row)
        db.flush()
        event = threading.Event()
        _active[benchmark_id] = event
    _workers.submit(_run,benchmark_id,body.model_dump(mode='json'),owner,event)
    return {'benchmark_id':str(benchmark_id),'status':'running'}


def _run(benchmark_id,config,owner,cancel):
    try:
        report = runner.run(benchmark_id,config,owner,cancel)
        with store.session() as db:
            row = db.get(store.Benchmark,benchmark_id)
            row.report = report
            row.model_name = report['model']['name']
            row.status = 'cancelled' if report['cancelled'] else 'done'
            row.score = report['score'].get('score')
            row.suite_version = report['score'].get('version',2)
            row.eval_tokens = (report.get('corpus') or {}).get('tokens_scored')
            row.completed_at = store.now()
    except Exception as error:
        with store.session() as db:
            row = db.get(store.Benchmark,benchmark_id)
            row.status,row.error,row.completed_at = 'failed',f'{type(error).__name__}: {error}',store.now()
    finally:
        with _guard:
            _active.pop(benchmark_id,None)


@app.post('/execute/{benchmark_id}/cancel')
def cancel(benchmark_id: uuid.UUID, owner: str = Depends(auth.require_service)):
    with _guard:
        event = _active.get(benchmark_id)
        if event:
            event.set()
    if event:
        with Hop(owner) as backend:
            backend.request('POST',f'/internal/inference/measure/{benchmark_id}/cancel')
    return {'cancelled':event is not None}


@app.get('/benchmarks/suite')
def suite():
    directory = runner.corpus_dir()
    return {'active_suites':['language'],'corpus_dir':str(directory),'corpus_files':len(list(directory.glob('*.txt'))),
            'chat_dir':None,'chat_files':0}


@app.get('/benchmarks')
def history(suite: str | None = None, model_id: uuid.UUID | None = None, limit: int = Query(default=200,ge=1,le=1000)):
    with store.session() as db:
        statement = select(store.Benchmark)
        if suite:
            statement = statement.where(store.Benchmark.suite==suite)
        if model_id:
            statement = statement.where(store.Benchmark.model_id==model_id)
        rows = db.scalars(statement.order_by(store.Benchmark.created_at.desc()).limit(limit))
        return {'benchmarks':[store.describe(row) for row in rows]}


@app.get('/benchmarks/{benchmark_id}')
def detail(benchmark_id: uuid.UUID):
    with store.session() as db:
        row = db.get(store.Benchmark,benchmark_id)
        if row is None:
            raise HTTPException(404,'benchmark not found')
        return store.describe(row,True)


@app.delete('/benchmarks/{benchmark_id}')
def remove(benchmark_id: uuid.UUID):
    with store.session() as db:
        row = db.get(store.Benchmark,benchmark_id)
        if row is None:
            raise HTTPException(404,'benchmark not found')
        if row.status == 'running':
            raise HTTPException(409,'cancel active work through the execution queue')
        db.delete(row)
    return {'benchmark_id':str(benchmark_id),'deleted':True}
