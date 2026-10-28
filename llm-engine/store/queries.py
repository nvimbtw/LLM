"""Reads and writes, one function per question.

Everything that scopes by owner takes it as an argument rather than reading it
from anywhere — the LLM engine cannot resolve an owner, so forgetting to pass one
has to be a type error rather than a silent full-table read.
"""

import itertools
import uuid
from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from store.schema import (
    Model,
    Run,
    RunKind,
    RunStatus,
    Tokenizer,
)

#: How far apart two adjacent ranks must stay before the band is renumbered.
#: Doubles run out of midpoints after ~50 inserts between one pair.
MIN_RANK_GAP = 1e-6


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── models ──────────────────────────────────────────────────────────────────


def list_models(db: Session, *, include_private: bool, checkpoints: bool = False) -> list[Model]:
    """Models, newest first. Intermediates are left out unless asked for: a run
    takes ten of them and they would otherwise outnumber the finished models in
    every list in the product."""
    stmt = select(Model).order_by(Model.created_at.desc())
    if not include_private:
        stmt = stmt.where(Model.published.is_(True))
    if not checkpoints:
        stmt = stmt.where(Model.checkpoint_index.is_(None))
    return list(db.scalars(stmt))


def checkpoints_of(db: Session, model_id: uuid.UUID) -> list[Model]:
    """One model's intermediates, in the order they were taken.

    Ordered by step and not by `created_at`, which two checkpoints of a fast run
    can share to the resolution the column stores.
    """
    return list(
        db.scalars(
            select(Model)
            .where(Model.parent_id == model_id, Model.checkpoint_index.is_not(None))
            .order_by(Model.checkpoint_step)
        )
    )


def checkpoint_ids(db: Session) -> set[uuid.UUID]:
    """Every intermediate's id, for callers filtering something keyed by it —
    the weights directories, which carry no marking of their own."""
    return set(db.scalars(select(Model.id).where(Model.checkpoint_index.is_not(None))))


def retire_recovery(db: Session, parent_id: uuid.UUID, latest_id: uuid.UUID) -> list[uuid.UUID]:
    """Keep the latest automatic save and any explicitly retained/referenced model.

    Legacy intermediate generations follow the same policy. Cross-service
    consumers pin a recovery model before storing its durable reference.
    """
    referenced = set()
    for run in db.scalars(select(Run)):
        config = run.config
        referenced.update(str(config[key]) for key in ("base_model", "resume_from", "model_name")
                          if config and config.get(key))
    for parameters in db.scalars(select(Model.parameters)):
        config = (parameters or {}).get("config") or {}
        referenced.update(str(config[key]) for key in ("base_model", "resume_from") if config.get(key))
    removed = []
    for model in checkpoints_of(db, parent_id):
        automatic = ((model.parameters or {}).get("result") or {}).get("recovery", True)
        if model.id != latest_id and automatic and not model.published and not (model.parameters or {}).get("recovery_keep") and not (model.parameters or {}).get("recovery_referenced") and str(model.id) not in referenced:
            removed.append(model.id)
            db.delete(model)
    db.flush()
    return removed


def get_model(db: Session, model_id: uuid.UUID) -> Model | None:
    return db.get(Model, model_id)


def find_model_by_name(db: Session, name: str) -> Model | None:
    return db.scalar(select(Model).where(Model.name == name))


def upsert_model(db: Session, *, model_id: uuid.UUID, name: str, **fields) -> Model:
    """Record a finished training run.

    Keyed on the id, which is also the weights directory: a run writes its own
    directory and updates its own row, whatever it is called.

    The name is *made* free rather than required to be. It is a label a person
    types, `models.name` is unique, and ids are minted per queued run — so two
    runs of the same composer are two models with one name, and queueing the same
    fine-tune twice is the ordinary way to get there. The old behaviour raised
    `UniqueViolation` from this insert, which is the worst possible moment for it:
    the run is over, the weights are on disk, and the row that would let anyone
    find them is the one thing that failed. A suffixed name costs nothing and is
    visible; a lost model costs the whole run.
    """
    model = db.get(Model, model_id)
    # Resolved before the row is added, not after: a query here would autoflush
    # the pending insert and hit the very constraint this exists to avoid.
    free = _free_model_name(db, name, model_id)
    if model is None:
        model = Model(id=model_id, name=free)
        db.add(model)
    model.name = free
    for key, value in fields.items():
        setattr(model, key, value)
    db.flush()
    return model


#: How long a model's name may be — `Model.name` is `String(120)`, and a suffix
#: has to fit inside that rather than push the row over it.
NAME_LIMIT = 120


def _free_model_name(db: Session, name: str, model_id: uuid.UUID) -> str:
    """`name`, or the first `name (n)` no other model holds.

    Only another *id* counts as a clash: a run re-registering its own row keeps
    the name it already has rather than climbing a suffix every time it is
    written.
    """
    taken = db.scalar(select(Model.id).where(Model.name == name))
    if taken is None or taken == model_id:
        return name

    stem = name[: NAME_LIMIT - 6].rstrip()
    for n in itertools.count(2):
        candidate = f"{stem} ({n})"
        holder = db.scalar(select(Model.id).where(Model.name == candidate))
        if holder is None or holder == model_id:
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover


def delete_model(db: Session, model_id: uuid.UUID) -> str | None:
    """Remove a model and its checkpoints, returning its name to confirm it
    existed.

    The intermediates go with the parent because nothing can reach them once it
    is gone: they are kept out of every listing, and the only way to one is
    `checkpoints_of` on the row being deleted. Their *directories* are the
    caller's to remove — `store/` has never heard of the disk — and
    `checkpoints_of`, read before this, is what names them.
    """
    model = db.get(Model, model_id)
    if model is None:
        return None
    name = model.name
    for checkpoint in checkpoints_of(db, model_id):
        db.delete(checkpoint)
    db.delete(model)
    db.flush()
    return name


# ── tokenizers ──────────────────────────────────────────────────────────────


def list_tokenizers(db: Session, *, include_implicit: bool = True) -> list[Tokenizer]:
    stmt = select(Tokenizer).order_by(Tokenizer.created_at.desc())
    if not include_implicit:
        stmt = stmt.where(Tokenizer.implicit.is_(False))
    return list(db.scalars(stmt))


def get_tokenizer(db: Session, tokenizer_id: uuid.UUID) -> Tokenizer | None:
    return db.get(Tokenizer, tokenizer_id)


def find_tokenizer_by_name(db: Session, name: str) -> Tokenizer | None:
    return db.scalar(select(Tokenizer).where(Tokenizer.name == name))


def upsert_tokenizer(db: Session, *, tokenizer_id: uuid.UUID, name: str, **fields) -> Tokenizer:
    """Record a built vocabulary. Keyed on the id for the reason `upsert_model`
    is: the id is the directory, so rebuilding writes the same bytes and must
    update that row rather than collide with its unique name."""
    tokenizer = db.get(Tokenizer, tokenizer_id)
    if tokenizer is None:
        tokenizer = Tokenizer(id=tokenizer_id, name=name)
        db.add(tokenizer)
    tokenizer.name = name
    for key, value in fields.items():
        setattr(tokenizer, key, value)
    db.flush()
    return tokenizer


def delete_tokenizer(db: Session, tokenizer_id: uuid.UUID) -> str | None:
    tokenizer = db.get(Tokenizer, tokenizer_id)
    if tokenizer is None:
        return None
    name = tokenizer.name
    db.delete(tokenizer)
    db.flush()
    return name


# ── chats ───────────────────────────────────────────────────────────────────


# ── runs ────────────────────────────────────────────────────────────────────


def enqueue(db: Session, *, kind, label: str, config: dict, owner_id=None, **fields) -> Run:
    """Add a run at the back of the line.

    A run naming `depends_on` starts `waiting` instead: it consumes a model that
    does not exist yet, so it is not a queue position, it is "after that one".
    """
    highest = db.scalar(select(func.max(Run.rank)).where(Run.status == RunStatus.queued))
    status = fields.pop("status", None) or (
        RunStatus.waiting if fields.get("depends_on") else RunStatus.queued
    )
    run = Run(
        kind=kind,
        label=label,
        config=config,
        owner_id=owner_id,
        rank=(highest or 0.0) + 1.0,
        status=status,
        **fields,
    )
    db.add(run)
    db.flush()
    return run


def claim_next(db: Session) -> Run | None:
    """The next run to execute, marked `running` in the same transaction.

    One process and one runner, so this needs no `SKIP LOCKED`: nothing else
    claims. The runner checks its process-local pause flag before calling.
    """
    run = db.scalar(
        select(Run).where(Run.status == RunStatus.queued).order_by(Run.rank).limit(1)
    )
    if run is None:
        return None
    run.status = RunStatus.running
    run.started_at = _now()
    db.flush()
    return run


def active_runs(db: Session, *, include_validation: bool = True) -> list[Run]:
    """Return the live band, applying trusted validation visibility in SQL."""
    live = (RunStatus.waiting, RunStatus.queued, RunStatus.running, RunStatus.cancelling)
    statement = select(Run).where(Run.status.in_(live))
    if not include_validation:
        statement = statement.where(Run.kind != RunKind.model_validation)
    return list(db.scalars(statement.order_by(Run.rank)))


_TERMINAL = (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


def run_history(db: Session, limit: int = 50, offset: int = 0, *, include_validation: bool = True) -> list[Run]:
    """Page terminal jobs after filtering validation rows for the authorized viewer."""
    statement = select(Run).where(Run.status.in_(_TERMINAL))
    if not include_validation:
        statement = statement.where(Run.kind != RunKind.model_validation)
    return list(
        db.scalars(
            statement
            .order_by(Run.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )


def count_history(db: Session, *, include_validation: bool = True) -> int:
    """Count exactly the same visible terminal domain as the history page query."""
    statement = select(func.count()).select_from(Run).where(Run.status.in_(_TERMINAL))
    if not include_validation:
        statement = statement.where(Run.kind != RunKind.model_validation)
    return db.scalar(statement) or 0


def lock_run(db: Session, run_id: uuid.UUID) -> Run | None:
    """Serialize cancellation/completion on the owning execution row and refresh its domain snapshot.

    PostgreSQL must lock only execution_jobs: eager domain relationships use
    nullable outer joins, whose nullable sides cannot be selected FOR UPDATE.
    """
    return db.scalar(select(Run).where(Run.id == run_id).with_for_update(of=Run).execution_options(populate_existing=True))


def finish_run(db: Session, run_id: uuid.UUID, status: RunStatus, **fields) -> Run | None:
    """Write one final status and its domain result within the caller-owned transaction."""
    run = db.get(Run, run_id)
    if run is None:
        return None
    run.status = status
    run.finished_at = _now()
    for key, value in fields.items():
        setattr(run, key, value)
    return run


def delete_run(db: Session, run_id: uuid.UUID) -> bool:
    """Forget a finished run. A live one is cancelled, never deleted — dropping
    the row would leave its worker writing to nothing."""
    run = db.get(Run, run_id)
    if run is None or not run.status.is_terminal:
        return False
    db.delete(run)
    db.flush()
    return True


def fail_orphaned(db: Session) -> int:
    """Close out runs still marked live from a previous process.

    Their worker died with it. Queued rows are deliberately untouched — the
    runner drains them, which is the point of the queue surviving a restart.
    """
    stranded = list(
        db.scalars(
            select(Run).where(Run.status.in_((RunStatus.running, RunStatus.cancelling)))
        )
    )
    for run in stranded:
        run.status = RunStatus.failed
        run.error = "the LLM engine restarted while this run was in flight"
        run.finished_at = _now()
    return len(stranded)


def reorder(db: Session, run_id: uuid.UUID, after: uuid.UUID | None) -> bool:
    """Move a queued run to sit directly after another, or to the front.

    Writes the midpoint of its new neighbours, so one row changes rather than
    the whole tail being renumbered. Renormalizes when doubles run out of
    midpoints between one pair.
    """
    run = db.get(Run, run_id)
    if run is None or run.status != RunStatus.queued:
        return False

    queued = list(
        db.scalars(select(Run).where(Run.status == RunStatus.queued).order_by(Run.rank))
    )
    others = [r for r in queued if r.id != run_id]

    if after is None:
        lower, upper = None, others[0].rank if others else run.rank + 1.0
    else:
        index = next((i for i, r in enumerate(others) if r.id == after), None)
        if index is None:
            return False
        lower = others[index].rank
        upper = others[index + 1].rank if index + 1 < len(others) else lower + 2.0

    run.rank = (lower + upper) / 2 if lower is not None else upper - 1.0

    if lower is not None and upper - lower < MIN_RANK_GAP:
        for position, other in enumerate(sorted(queued, key=lambda r: r.rank), start=1):
            other.rank = float(position)
    db.flush()
    return True


def waiting_on(db: Session, run_id: uuid.UUID) -> list[Run]:
    """Runs still waiting on this one. A dependent that was cancelled in the
    meantime is not among them — promoting it would resurrect it."""
    return list(
        db.scalars(
            select(Run).where(Run.depends_on == run_id, Run.status == RunStatus.waiting)
        )
    )
