"""Queue pause is process-local; durable jobs remain unchanged."""

from alembic import op

revision = "d8c10f6a21b4"
down_revision = "a3f5c81b6d20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("queue_state")


def downgrade() -> None:
    raise RuntimeError("Forward-only migration: queue pause no longer persists")
