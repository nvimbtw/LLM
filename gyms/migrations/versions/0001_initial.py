"""initial — runs, episodes and steps

Revision ID: 0001
Revises:
Create Date: 2026-08-29
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

RUN_STATUS = sa.Enum("pending", "running", "done", "abandoned", "failed", name="gym_run_status")
EPISODE_STATUS = sa.Enum("pending", "running", "graded", "closed", name="gym_episode_status")
JSON = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
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


def downgrade() -> None:
    op.drop_index("ix_gym_steps_episode_id", table_name="gym_steps")
    op.drop_table("gym_steps")
    op.drop_index("ix_gym_episodes_run_id", table_name="gym_episodes")
    op.drop_table("gym_episodes")
    op.drop_index("ix_gym_runs_owner_id", table_name="gym_runs")
    op.drop_table("gym_runs")
    op.execute("DROP TYPE IF EXISTS gym_episode_status")
    op.execute("DROP TYPE IF EXISTS gym_run_status")
