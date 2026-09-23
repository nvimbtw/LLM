"""a run can be its own GRPO batch

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-02

`gym_runs.group_size`: how many consecutive episodes are rollouts of the *same*
task. It is a column rather than a number the caller keeps because episode *i*
now draws `seed_base + i // group_size`, and a run whose row does not carry the
divisor is a run nobody can regenerate — the reproducibility `seed_base` was
added for.

Backfilled to 1, which is exactly what every existing run is: one episode per
task. Unconditional, unlike `0002`, because this revision id names one schema.
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gym_runs",
        sa.Column("group_size", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("gym_runs", "group_size")
