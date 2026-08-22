"""search cache: origin, and a keyed/timestamped document

Revision ID: a7c3f0e91d5b
Revises: 28e7f4329a1f
Create Date: 2026-08-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a7c3f0e91d5b'
down_revision: Union[str, Sequence[str], None] = '28e7f4329a1f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Postgres cannot add an enum label inside the transaction alembic wraps
    # this migration in — force autocommit for just this statement.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE dataset_origin ADD VALUE 'search'")

    op.add_column('documents', sa.Column('cache_key', sa.String(length=500), nullable=True))
    op.add_column(
        'documents',
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text('now()'),
        ),
    )
    # The default above exists only to backfill rows written before this
    # column did; new rows get their timestamp from the ORM, the same as every
    # other `created_at` in this schema — so it comes back off, matching what
    # schema.py actually declares.
    op.alter_column('documents', 'created_at', server_default=None)
    op.create_index(
        'ix_documents_dataset_id_cache_key', 'documents', ['dataset_id', 'cache_key']
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_documents_dataset_id_cache_key', table_name='documents')
    op.drop_column('documents', 'created_at')
    op.drop_column('documents', 'cache_key')
    # Postgres has no ALTER TYPE ... DROP VALUE. A downgrade leaves the label
    # in place, same as every enum addition in this stack — it is inert until
    # something writes it again.
