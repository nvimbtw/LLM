"""an rl iteration is not a benchmark

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-04

`gym_runs.kind`: `eval` or `rl`. A training run takes one gym run per optimizer
step, so without this the board a person reads to compare models is buried by
the batches of whichever run happened to be training — same three tables, same
sandbox, and nothing else in the row that could tell them apart.

Backfilled to `eval`, which is what every run made before RL existed is.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

KIND = sa.Enum("eval", "rl", name="gym_run_kind")


def upgrade() -> None:
    KIND.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "gym_runs",
        sa.Column("kind", KIND, nullable=False, server_default="eval"),
    )


def downgrade() -> None:
    op.drop_column("gym_runs", "kind")
    KIND.drop(op.get_bind(), checkfirst=True)
