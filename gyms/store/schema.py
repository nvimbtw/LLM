"""What the gyms database holds.

Three tables, one nesting: a **run** is a model against N generated tasks, an
**episode** is one of those tasks, and a **step** is one command inside it. The
run exists because a single episode says almost nothing — these tasks are pass
or fail, so a model's ability is only visible across a batch.

`gym_episodes.id` is also the key the sandbox files an episode's filesystem
under (`gyms.sandbox`), which is why it is generated here rather than taken from
the caller: the row and the workspace are created together or not at all.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, Uuid, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
Key = Uuid(as_uuid=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    #: Created, with its episodes seeded, but nothing has driven it yet.
    pending = "pending"
    running = "running"
    done = "done"
    #: The socket driving it went away. Whatever episodes finished are still
    #: scored; the rest never ran.
    abandoned = "abandoned"
    failed = "failed"


class EpisodeStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    #: Answered and graded — the only status with a meaningful `score`.
    graded = "graded"
    #: Closed without an answer: the round budget ran out, or the run was
    #: abandoned under it.
    closed = "closed"


class GymRun(Base):
    __tablename__ = "gym_runs"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    gym_id: Mapped[str] = mapped_column(String(60))
    model_name: Mapped[str] = mapped_column(String(200))
    #: The task types drawn from, in the order the episodes cycle through them.
    task_types: Mapped[list] = mapped_column(Json)
    #: Episode *i* is generated from `seed_base + i`, so a run is reproducible
    #: from this row alone.
    seed_base: Mapped[int] = mapped_column(Integer)
    episode_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="gym_run_status"), default=RunStatus.pending
    )
    #: Mean episode score, over the episodes that have been graded. Null until
    #: one has been.
    score: Mapped[float | None] = mapped_column(Float)
    graded: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(Key, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GymEpisode(Base):
    __tablename__ = "gym_episodes"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Key, ForeignKey("gym_runs.id", ondelete="CASCADE"), index=True
    )
    #: Position in the run, and the offset its seed was drawn at.
    index: Mapped[int] = mapped_column(Integer)
    gym_id: Mapped[str] = mapped_column(String(60))
    task_type: Mapped[str] = mapped_column(String(60))
    seed: Mapped[int] = mapped_column(Integer)
    #: The whole generated task, ground truth included. Never served while the
    #: episode is open — see `gyms.bash.tasks.Task.public`.
    task_config: Mapped[dict] = mapped_column(Json)
    status: Mapped[EpisodeStatus] = mapped_column(
        Enum(EpisodeStatus, name="gym_episode_status"), default=EpisodeStatus.pending
    )
    #: What the model finally said, as the grader extracted it.
    answer: Mapped[str | None] = mapped_column(Text)
    #: The whole last reply, markup and all, so the transcript survives a reload.
    reply: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)
    reward_total: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GymStep(Base):
    __tablename__ = "gym_steps"
    __table_args__ = (UniqueConstraint("episode_id", "step_num", name="uq_gym_steps_order"),)

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    episode_id: Mapped[uuid.UUID] = mapped_column(
        Key, ForeignKey("gym_episodes.id", ondelete="CASCADE"), index=True
    )
    #: 0-based and actually incremented. The unique constraint beside it is what
    #: stops two concurrent drivers from writing a trajectory that cannot be read
    #: back in the order it happened.
    step_num: Mapped[int] = mapped_column(Integer)
    #: The assistant text of the round that made this call — what the model was
    #: thinking when it reached for the shell.
    reply: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    stdout: Mapped[str | None] = mapped_column(Text)
    stderr: Mapped[str | None] = mapped_column(Text)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    timed_out: Mapped[bool] = mapped_column(Boolean, default=False)
    reward: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
