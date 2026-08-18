"""a tool message role

Revision ID: c4e91a70f2bd
Revises: 9a41c7b2d3e5
Create Date: 2026-08-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c4e91a70f2bd'
down_revision: Union[str, Sequence[str], None] = '9a41c7b2d3e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # A tool's answer is stored as a message of its own, between the assistant
    # turn that called for it and the one that uses it, so the history a later
    # turn is rendered from carries what the model was told.
    #
    # Postgres holds `role` as a real ENUM and a new value has to be added to
    # the type itself; SQLite has no enum to alter, so the tests need nothing
    # here. `IF NOT EXISTS` so a database where someone added it by hand is a
    # no-op rather than a failure.
    if op.get_bind().dialect.name == 'postgresql':
        op.execute("ALTER TYPE role ADD VALUE IF NOT EXISTS 'tool'")


def downgrade() -> None:
    """Downgrade schema."""
    # The enum value stays. Postgres cannot drop one, and recreating the type
    # means rewriting every `messages` row to remove a value nothing is using.
    pass
