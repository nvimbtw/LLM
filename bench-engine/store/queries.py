import uuid
from datetime import datetime, timezone
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from store.schema import Benchmark, BenchmarkStatus, QueueState, Run, RunStatus, Suite
MIN_RANK_GAP = 1e-6
def _now() -> datetime:
    return datetime.now(timezone.utc)
def open_benchmark(db: Session, *, model_id: uuid.UUID, model_name: str, suite: Suite | str, **fields) -> Benchmark:
    suite_str = suite.value if isinstance(suite, Suite) else str(suite)
    run = Benchmark(model_id=model_id, model_name=model_name, suite=suite_str, **fields)
    db.add(run)
    db.flush()
    return run
def close_benchmark(db: Session, benchmark_id: uuid.UUID, **fields) -> Benchmark | None:
    run = db.get(Benchmark, benchmark_id)
    if run is None:
        return None
    for k, v in fields.items():
        setattr(run, k, v)
    run.completed_at = _now()
    return run
def benchmarks_of(db: Session, model_id: uuid.UUID, limit: int = 50) -> list[Benchmark]:
    return list(db.scalars(select(Benchmark).where(Benchmark.model_id == model_id).order_by(Benchmark.created_at.desc()).limit(limit)))
def benchmarks_in(db: Session, suite: Suite | str, limit: int = 200) -> list[Benchmark]:
    suite_str = suite.value if isinstance(suite, Suite) else str(suite)
    return list(db.scalars(select(Benchmark).where(Benchmark.suite == suite_str).order_by(Benchmark.created_at.desc()).limit(limit)))
def delete_benchmark(db: Session, benchmark_id: uuid.UUID) -> bool:
    found = db.get(Benchmark, benchmark_id)
    if found is None:
        return False
    db.delete(found)
    db.flush()
    return True
def latest_benchmark(db: Session, model_id: uuid.UUID, suite: Suite | str, version: int) -> Benchmark | None:
    suite_str = suite.value if isinstance(suite, Suite) else str(suite)
    return db.scalar(select(Benchmark).where(Benchmark.model_id==model_id, Benchmark.suite==suite_str, Benchmark.suite_version==version, Benchmark.score.is_not(None)).order_by(Benchmark.created_at.desc()).limit(1))
def enqueue(db: Session, *, kind, label: str, config: dict, owner_id=None, **fields) -> Run:
    highest = db.scalar(select(func.max(Run.rank)).where(Run.status == RunStatus.queued))
    status = fields.pop("status", None) or (RunStatus.waiting if fields.get("depends_on") else RunStatus.queued)
    run = Run(kind=kind, label=label, config=config, owner_id=owner_id, rank=(highest or 0.0)+1.0, status=status, **fields)
    db.add(run)
    db.flush()
    return run
def claim_next(db: Session) -> Run | None:
    if is_paused(db):
        return None
    run = db.scalar(select(Run).where(Run.status == RunStatus.queued).order_by(Run.rank).limit(1))
    if run is None:
        return None
    run.status = RunStatus.running
    run.started_at = _now()
    db.flush()
    return run
def active_runs(db: Session) -> list[Run]:
    live = (RunStatus.waiting, RunStatus.queued, RunStatus.running, RunStatus.cancelling)
    return list(db.scalars(select(Run).where(Run.status.in_(live)).order_by(Run.rank)))
_TERMINAL = (RunStatus.done, RunStatus.cancelled, RunStatus.failed)
def run_history(db: Session, limit: int = 50, offset: int = 0) -> list[Run]:
    return list(db.scalars(select(Run).where(Run.status.in_(_TERMINAL)).order_by(Run.created_at.desc()).limit(limit).offset(offset)))
def count_history(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Run).where(Run.status.in_(_TERMINAL))) or 0
def finish_run(db: Session, run_id: uuid.UUID, status: RunStatus, **fields) -> Run | None:
    run = db.get(Run, run_id)
    if run is None:
        return None
    run.status = status
    run.finished_at = _now()
    for k, v in fields.items():
        setattr(run, k, v)
    return run
def delete_run(db: Session, run_id: uuid.UUID) -> bool:
    run = db.get(Run, run_id)
    if run is None or not run.status.is_terminal:
        return False
    db.delete(run)
    db.flush()
    return True
def fail_orphaned(db: Session) -> int:
    stranded = list(db.scalars(select(Run).where(Run.status.in_((RunStatus.running, RunStatus.cancelling)))))
    for run in stranded:
        run.status = RunStatus.failed
        run.error = "bench-engine restarted while this run was in flight"
        run.finished_at = _now()
    return len(stranded)
def reorder(db: Session, run_id: uuid.UUID, after: uuid.UUID | None) -> bool:
    run = db.get(Run, run_id)
    if run is None or run.status != RunStatus.queued:
        return False
    queued = list(db.scalars(select(Run).where(Run.status == RunStatus.queued).order_by(Run.rank)))
    others = [r for r in queued if r.id != run_id]
    if after is None:
        lower, upper = None, others[0].rank if others else run.rank + 1.0
    else:
        index = next((i for i, r in enumerate(others) if r.id == after), None)
        if index is None:
            return False
        lower = others[index].rank
        upper = others[index+1].rank if index+1 < len(others) else lower+2.0
    run.rank = (lower+upper)/2 if lower is not None else upper-1.0
    if lower is not None and upper-lower < MIN_RANK_GAP:
        for pos, other in enumerate(sorted(queued, key=lambda r: r.rank), start=1):
            other.rank = float(pos)
    db.flush()
    return True
def waiting_on(db: Session, run_id: uuid.UUID) -> list[Run]:
    return list(db.scalars(select(Run).where(Run.depends_on==run_id, Run.status==RunStatus.waiting)))
def _queue_state(db: Session) -> QueueState:
    state = db.get(QueueState, True)
    if state is None:
        state = QueueState(only_row=True, paused=False)
        db.add(state)
        db.flush()
    return state
def is_paused(db: Session) -> bool:
    return _queue_state(db).paused
def set_paused(db: Session, paused: bool) -> None:
    state = _queue_state(db)
    state.paused = paused
    state.updated_at = _now()
