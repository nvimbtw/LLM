"""the gym became runs and episodes

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-29

Reconciles a database built by the *first* `0001` — the one that made
`gym_sessions` and a `gym_steps` keyed by it — with the shape the module
actually has. That revision was rewritten in place rather than superseded,
because the module had never been committed and there was believed to be no
database anywhere holding the old shape. There was one, and reusing the id is
what made it invisible: `alembic upgrade head` saw `0001` already stamped and
did nothing at all, while the service's own `create_all` quietly added the two
tables that *were* missing and skipped `gym_steps`, which already existed under
its old definition. Every step then failed to insert, as a 500 from the gyms
service and `The shell could not be reached` in the transcript.

**This migration is conditional, which no other migration here should be.** One
revision id names two different schemas, so what has to happen depends on which
of them is on the other end. Every branch is guarded on what actually exists.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

JSON = postgresql.JSONB(astext_type=sa.Text())

RUN_STATUS = postgresql.ENUM(
    "pending", "running", "done", "abandoned", "failed",
    name="gym_run_status", create_type=False,
)
EPISODE_STATUS = postgresql.ENUM(
    "pending", "running", "graded", "closed",
    name="gym_episode_status", create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())

    # The legacy table has no counterpart here: an episode is what a session
    # became, and the rows were hand-driven trajectories with no score on them.
    if "gym_sessions" in tables:
        op.drop_table("gym_sessions")
        op.execute("DROP TYPE IF EXISTS gym_status")

    # `gym_steps` is the one that cannot be left alone: same name, different
    # table. Dropped rather than altered — its rows point at sessions that no
    # longer exist, so there is nothing in it to carry forward.
    if "gym_steps" in tables and not _has_column(bind, "gym_steps", "episode_id"):
        op.drop_table("gym_steps")
        tables.discard("gym_steps")

    RUN_STATUS.create(bind, checkfirst=True)
    EPISODE_STATUS.create(bind, checkfirst=True)

    if "gym_runs" not in tables:
        _create_runs()
    if "gym_episodes" not in tables:
        _create_episodes()
    if "gym_steps" not in tables:
        _create_steps()


def downgrade() -> None:
    """Down to an empty database rather than back to the legacy shape.

    There is no way back to `gym_sessions` that means anything: the rows it held
    were dropped on the way up, and recreating the table empty would be a
    schema pretending to be a rollback.
    """
    op.drop_table("gym_steps")
    op.drop_table("gym_episodes")
    op.drop_table("gym_runs")
    op.execute("DROP TYPE IF EXISTS gym_episode_status")
    op.execute("DROP TYPE IF EXISTS gym_run_status")


def _has_column(bind, table: str, column: str) -> bool:
    return any(c["name"] == column for c in sa.inspect(bind).get_columns(table))


def _create_runs() -> None:
    op.create_table(
        "gym_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("gym_id", sa.String(length=60), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("task_types", JSON, nullable=False),
        sa.Column("seed_base", sa.Integer(), nullable=False),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("status", RUN_STATUS, nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("graded", sa.Integer(), nullable=False),
        sa.Column("correct", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_gym_runs_owner_id", "gym_runs", ["owner_id"])


def _create_episodes() -> None:
    op.create_table(
        "gym_episodes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("gym_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("gym_id", sa.String(length=60), nullable=False),
        sa.Column("task_type", sa.String(length=60), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("task_config", JSON, nullable=False),
        sa.Column("status", EPISODE_STATUS, nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("reply", sa.Text(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("reward_total", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_gym_episodes_run_id", "gym_episodes", ["run_id"])


def _create_steps() -> None:
    op.create_table(
        "gym_steps",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("episode_id", sa.Uuid(), sa.ForeignKey("gym_episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_num", sa.Integer(), nullable=False),
        sa.Column("reply", sa.Text(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("stdout", sa.Text(), nullable=True),
        sa.Column("stderr", sa.Text(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("timed_out", sa.Boolean(), nullable=False),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("episode_id", "step_num", name="uq_gym_steps_order"),
    )
    op.create_index("ix_gym_steps_episode_id", "gym_steps", ["episode_id"])
