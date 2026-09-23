"""Persist a run's resolved source catalogue and sampling policy."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gym_runs", sa.Column(
        "selection", sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
        nullable=True,
    ))


def downgrade() -> None:
    op.drop_column("gym_runs", "selection")
