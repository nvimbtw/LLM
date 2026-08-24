import uuid
from fastapi import APIRouter, HTTPException
import store
from store import queries as q
from store.schema import Suite
router = APIRouter(tags=["benchmarks"])
def _row(b):
    raw = b.suite.value if hasattr(b.suite, "value") else str(b.suite)
    suite_out = "language" if raw in ("language", "language-v3", "language_v3") else raw
    return {"benchmark_id": str(b.id), "model_id": str(b.model_id), "model_name": b.model_name, "suite": suite_out, "suite_version": b.suite_version, "status": b.status.value, "score": b.score, "eval_tokens": b.eval_tokens, "eval_conversations": b.eval_conversations, "error": b.error, "created_at": b.created_at.isoformat() if b.created_at else None, "completed_at": b.completed_at.isoformat() if b.completed_at else None}
@router.get("/benchmarks")
def list_benchmarks(suite: str | None = None, limit: int = 200):
    with store.session() as db:
        if suite:
            if suite in ("language", "language-v3", "language_v3"):
                from sqlalchemy import select
                from store.schema import Benchmark
                rows = list(db.scalars(select(Benchmark).where(Benchmark.suite.in_(["language", "language_v3", "language-v3"])).order_by(Benchmark.created_at.desc()).limit(limit)))
                return {"benchmarks": [_row(r) for r in rows]}
            if suite in ("markup", "assistant"):
                return {"benchmarks": []}
            try:
                s = Suite(suite)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"unknown suite {suite}")
            rows = q.benchmarks_in(db, s, limit)
        else:
            from sqlalchemy import select
            from store.schema import Benchmark
            rows = list(db.scalars(select(Benchmark).order_by(Benchmark.created_at.desc()).limit(limit)))
        return {"benchmarks": [_row(r) for r in rows]}
@router.get("/benchmarks/suite")
def suite_info():
    from pathlib import Path
    eval_dir = Path("/app/eval_data/language/corpus")
    if not eval_dir.exists():
        eval_dir = Path("eval_data/language/corpus")
    corpus_count = len(list(eval_dir.glob("*.txt"))) if eval_dir.exists() else 0
    chat_dir = Path("/app/eval_data/language/chat")
    return {"corpus_dir": str(eval_dir) if eval_dir.exists() else "/app/eval_data/language/corpus", "corpus_files": corpus_count, "chat_dir": str(chat_dir), "chat_files": 0}
def _sanitize(obj):
    import math
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
@router.get("/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: uuid.UUID):
    with store.session() as db:
        b = db.get(q.Benchmark, benchmark_id)
        if b is None:
            raise HTTPException(status_code=404, detail="not found")
        row = _row(b)
        report = _sanitize(b.report) if b.report else None
        if report and isinstance(report, dict) and report.get("suite") == "language-v3":
            report["suite"] = "language"
        row["report"] = report
        return row
@router.get("/models/{model_id}/benchmarks")
def list_model_benchmarks(model_id: uuid.UUID, limit: int = 50):
    with store.session() as db:
        rows = q.benchmarks_of(db, model_id, limit)
        return {"benchmarks": [_row(r) for r in rows]}
@router.delete("/benchmarks/{benchmark_id}")
def delete_benchmark(benchmark_id: uuid.UUID):
    with store.session() as db:
        if not q.delete_benchmark(db, benchmark_id):
            raise HTTPException(status_code=404, detail="not found")
        return {"deleted": True}
