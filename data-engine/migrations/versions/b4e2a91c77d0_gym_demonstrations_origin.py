"""gym demonstrations: a dataset origin for them

Revision ID: b4e2a91c77d0
Revises: a7c3f0e91d5b
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b4e2a91c77d0'
down_revision: Union[str, Sequence[str], None] = 'a7c3f0e91d5b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Postgres cannot add an enum label inside the transaction alembic wraps
    # this migration in — force autocommit for just this statement, the same
    # way `a7c3f0e91d5b` added 'search'.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE dataset_origin ADD VALUE 'gym'")


def downgrade() -> None:
    """Downgrade schema."""
    # Postgres has no ALTER TYPE ... DROP VALUE. The label stays and is inert
    # until something writes it again, as with every enum addition here.
