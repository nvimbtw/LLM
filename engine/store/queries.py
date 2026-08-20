"""Reads and writes, one function per question.

Everything that scopes by owner takes it as an argument rather than reading it
from anywhere — the engine cannot resolve an owner, so forgetting to pass one
has to be a type error rather than a silent full-table read.
"""

import itertools
import uuid
from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from store.schema import (
    Attachment,
    Benchmark,
    Chat,
    Message,
    Model,
    QueueState,
    Role,
    Run,
    RunStatus,
    Suite,
    Tokenizer,
)

#: How far apart two adjacent ranks must stay before the band is renumbered.
#: Doubles run out of midpoints after ~50 inserts between one pair.
MIN_RANK_GAP = 1e-6


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── models ──────────────────────────────────────────────────────────────────


def list_models(db: Session, *, include_private: bool) -> list[Model]:
    stmt = select(Model).order_by(Model.created_at.desc())
    if not include_private:
        stmt = stmt.where(Model.published.is_(True))
    return list(db.scalars(stmt))


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
    """Remove a model, returning its name to confirm it existed."""
    model = db.get(Model, model_id)
    if model is None:
        return None
    name = model.name
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


def list_chats(db: Session, owner_id: uuid.UUID) -> list[tuple[Chat, int]]:
    """Every chat of one owner, newest first, with its cumulative token count."""
    stmt = (
        select(Chat, func.coalesce(func.sum(Message.tokens), 0))
        .outerjoin(Message, Message.chat_id == Chat.id)
        .where(Chat.owner_id == owner_id)
        .group_by(Chat.id)
        .order_by(Chat.created_at.desc())
    )
    return [(chat, int(tokens)) for chat, tokens in db.execute(stmt)]


def owned_chat(db: Session, chat_id: uuid.UUID, owner_id: uuid.UUID) -> Chat | None:
    """A chat, only if this owner has it. Chats are private with no admin path."""
    return db.scalar(select(Chat).where(Chat.id == chat_id, Chat.owner_id == owner_id))


def create_chat(db: Session, owner_id: uuid.UUID, title: str) -> Chat:
    chat = Chat(owner_id=owner_id, title=title)
    db.add(chat)
    db.flush()
    return chat


def rename_chat(db: Session, chat_id: uuid.UUID, owner_id: uuid.UUID, title: str) -> Chat | None:
    chat = owned_chat(db, chat_id, owner_id)
    if chat is not None:
        chat.title = title
    return chat


def delete_chat(db: Session, chat_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
    chat = owned_chat(db, chat_id, owner_id)
    if chat is None:
        return False
    db.delete(chat)
    db.flush()
    return True


def delete_chats_of(db: Session, owner_id: uuid.UUID) -> int:
    """Everything one owner has. Called when the backend deletes the account —
    idempotent, so a half-failed deletion is fixed by retrying it."""
    chats = list(db.scalars(select(Chat).where(Chat.owner_id == owner_id)))
    for chat in chats:
        db.delete(chat)
    db.flush()
    return len(chats)


def distinct_owners(db: Session) -> list[uuid.UUID]:
    """Every owner the engine believes exists, for the reconciliation sweep."""
    return list(db.scalars(select(Chat.owner_id).distinct()))


# ── messages ────────────────────────────────────────────────────────────────


def add_message(
    db: Session,
    *,
    chat_id: uuid.UUID,
    role: Role,
    content: str,
    tokens: int = 0,
    model_id: uuid.UUID | None = None,
) -> Message:
    message = Message(
        chat_id=chat_id, role=role, content=content, tokens=tokens, model_id=model_id
    )
    db.add(message)
    db.flush()
    return message


def messages_of(db: Session, chat_id: uuid.UUID) -> list[Message]:
    # Attachments eagerly, in one further statement: every caller renders them,
    # and lazily is one query per message in a conversation of any length. The
    # bytes do not come with them — `Attachment.content` is deferred.
    stmt = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .options(selectinload(Message.attachments))
    )
    return list(db.scalars(stmt))


def set_feedback(db: Session, message_id: uuid.UUID, owner_id: uuid.UUID, feedback) -> bool:
    """Rate an assistant reply. Scoped through the chat, so one owner cannot
    rate another's."""
    stmt = (
        select(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .where(Message.id == message_id, Chat.owner_id == owner_id)
    )
    message = db.scalar(stmt)
    if message is None:
        return False
    message.feedback = feedback
    return True


# ── attachments ─────────────────────────────────────────────────────────────
#
# Scoped by owner rather than through a chat, because an attachment exists
# before the message it belongs to does — and on a new conversation, before the
# chat does. Binding is what later ties it to a turn.


def create_attachment(
    db: Session,
    *,
    owner_id: uuid.UUID,
    filename: str,
    media_type: str,
    content: bytes,
) -> Attachment:
    attachment = Attachment(
        owner_id=owner_id,
        filename=filename,
        media_type=media_type,
        size=len(content),
        content=content,
    )
    db.add(attachment)
    db.flush()
    return attachment


def owned_attachment(
    db: Session, attachment_id: uuid.UUID, owner_id: uuid.UUID
) -> Attachment | None:
    stmt = select(Attachment).where(
        Attachment.id == attachment_id, Attachment.owner_id == owner_id
    )
    return db.scalar(stmt)


def bind_attachments(
    db: Session,
    *,
    message_id: uuid.UUID,
    owner_id: uuid.UUID,
    attachment_ids: list[uuid.UUID],
) -> list[Attachment] | None:
    """Tie staged uploads to the turn they were sent with.

    None when any id is not this owner's or is already bound — all or nothing,
    because a turn that silently carried three of the four files a person
    attached is worse than one that refused. Rebinding is refused rather than
    moved: a sent turn is history, and a file cannot leave one message for
    another.
    """
    if not attachment_ids:
        return []

    stmt = select(Attachment).where(
        Attachment.id.in_(attachment_ids),
        Attachment.owner_id == owner_id,
        Attachment.message_id.is_(None),
    )
    found = {a.id: a for a in db.scalars(stmt)}
    if len(found) != len(set(attachment_ids)):
        return None

    ordered = [found[a] for a in dict.fromkeys(attachment_ids)]
    for attachment in ordered:
        attachment.message_id = message_id
    db.flush()
    return ordered


def delete_attachment(db: Session, attachment_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
    """Discard a staged upload. Only an unbound one: removing a file from a turn
    that has already been sent would edit history nobody asked to edit."""
    attachment = owned_attachment(db, attachment_id, owner_id)
    if attachment is None or attachment.message_id is not None:
        return False
    db.delete(attachment)
    db.flush()
    return True


# ── usage ───────────────────────────────────────────────────────────────────
#
# Derived from the messages themselves rather than from a separate ledger: a
# user turn is what was asked, an assistant turn is what was generated, and a
# second table recording the same numbers can only ever drift from them.


def usage_by_day(db: Session, owner_id: uuid.UUID, since: datetime) -> list[tuple]:
    """(day, asked, generated, replies) per day, oldest first."""
    day = func.date(Message.created_at).label("day")
    return list(
        db.execute(
            select(day, _asked(), _generated(), _replies())
            .join(Chat, Chat.id == Message.chat_id)
            .where(Chat.owner_id == owner_id, Message.created_at >= since)
            .group_by(day)
            .order_by(day)
        )
    )


def usage_by_model(db: Session, owner_id: uuid.UUID, since: datetime) -> list[tuple]:
    """(model_id, name, tokens, replies) per model, heaviest first.

    A null model is one since deleted; its replies still count against the
    total, and the caller labels them.
    """
    total = func.coalesce(func.sum(Message.tokens), 0).label("tokens")
    return list(
        db.execute(
            select(Message.model_id, Model.name, total, _replies())
            .join(Chat, Chat.id == Message.chat_id)
            .outerjoin(Model, Model.id == Message.model_id)
            .where(
                Chat.owner_id == owner_id,
                Message.created_at >= since,
                Message.role == Role.assistant,
            )
            .group_by(Message.model_id, Model.name)
            .order_by(total.desc())
        )
    )


def _asked():
    return func.coalesce(
        func.sum(case((Message.role == Role.user, Message.tokens), else_=0)), 0
    ).label("asked")


def _generated():
    return func.coalesce(
        func.sum(case((Message.role == Role.assistant, Message.tokens), else_=0)), 0
    ).label("generated")


def _replies():
    return func.coalesce(
        func.sum(case((Message.role == Role.assistant, 1), else_=0)), 0
    ).label("replies")


# ── benchmarks ──────────────────────────────────────────────────────────────


def open_benchmark(db: Session, *, model: Model, suite: Suite, **fields) -> Benchmark:
    """Opened when the run reaches the device, not when it is queued — a row
    here means a measurement was actually attempted."""
    run = Benchmark(model_id=model.id, model_name=model.name, suite=suite, **fields)
    db.add(run)
    db.flush()
    return run


def close_benchmark(db: Session, benchmark_id: uuid.UUID, **fields) -> Benchmark | None:
    run = db.get(Benchmark, benchmark_id)
    if run is None:
        return None
    for key, value in fields.items():
        setattr(run, key, value)
    run.completed_at = _now()
    return run


def benchmarks_of(db: Session, model_id: uuid.UUID, limit: int = 50) -> list[Benchmark]:
    return list(
        db.scalars(
            select(Benchmark)
            .where(Benchmark.model_id == model_id)
            .order_by(Benchmark.created_at.desc())
            .limit(limit)
        )
    )


def benchmarks_in(db: Session, suite: Suite, limit: int = 200) -> list[Benchmark]:
    """Every model's runs of one suite, newest first. Suites are never compared,
    so a history is always of one of them."""
    return list(
        db.scalars(
            select(Benchmark)
            .where(Benchmark.suite == suite)
            .order_by(Benchmark.created_at.desc())
            .limit(limit)
        )
    )


def delete_benchmark(db: Session, benchmark_id: uuid.UUID) -> bool:
    found = db.get(Benchmark, benchmark_id)
    if found is None:
        return False
    db.delete(found)
    db.flush()
    return True


def latest_benchmark(
    db: Session, model_id: uuid.UUID, suite: Suite, version: int
) -> Benchmark | None:
    """The newest comparable score. Takes a version because a scoring change
    moves the number without the model having changed."""
    return db.scalar(
        select(Benchmark)
        .where(
            Benchmark.model_id == model_id,
            Benchmark.suite == suite,
            Benchmark.suite_version == version,
            Benchmark.score.is_not(None),
        )
        .order_by(Benchmark.created_at.desc())
        .limit(1)
    )


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
    claims. Returns nothing while the queue is paused.
    """
    if is_paused(db):
        return None
    run = db.scalar(
        select(Run).where(Run.status == RunStatus.queued).order_by(Run.rank).limit(1)
    )
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
    return list(
        db.scalars(
            select(Run)
            .where(Run.status.in_(_TERMINAL))
            .order_by(Run.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )


def count_history(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Run).where(Run.status.in_(_TERMINAL))) or 0


def finish_run(db: Session, run_id: uuid.UUID, status: RunStatus, **fields) -> Run | None:
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
        run.error = "the engine restarted while this run was in flight"
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


# ── queue state ─────────────────────────────────────────────────────────────


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
