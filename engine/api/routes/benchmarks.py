"""Scores, across models.

`models.py` answers "what did this model score"; this answers "what has been
scored", which is the history page's question and needs no model to ask it.
"""

import uuid

from fastapi import APIRouter, HTTPException

import store
from api.routes.models import benchmark_row
from llm import storage
from store import queries as q
from store.schema import Suite

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks")
def list_benchmarks(suite: Suite = Suite.language, limit: int = 200):
    with store.session() as db:
        return {"benchmarks": [benchmark_row(b) for b in q.benchmarks_in(db, suite, limit)]}


@router.get("/benchmarks/suite")
def suite_info():
    """Whether the held-out sets are installed.

    Surfaced so an absent score reads as "no corpus here" rather than as a bug.
    """
    corpus = storage.eval_dir() / "corpus"
    chat = storage.eval_dir() / "chat"
    return {
        "corpus_dir": str(corpus),
        "corpus_files": _count(corpus, "*.txt"),
        "chat_dir": str(chat),
        "chat_files": _count(chat, "*.jsonl"),
    }


@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: uuid.UUID):
    """One run with its whole report — the only place the report is sent."""
    with store.session() as db:
        found = db.get(store.Benchmark, benchmark_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no benchmark {benchmark_id}")
        return {**benchmark_row(found), "report": found.report}


@router.delete("/benchmarks/{benchmark_id}", status_code=200)
def delete_benchmark(benchmark_id: uuid.UUID):
    with store.session() as db:
        if not q.delete_benchmark(db, benchmark_id):
            raise HTTPException(status_code=404, detail=f"no benchmark {benchmark_id}")
    return {"benchmark_id": str(benchmark_id), "deleted": True}


def _count(directory, pattern: str) -> int:
    return len(list(directory.glob(pattern))) if directory.is_dir() else 0
