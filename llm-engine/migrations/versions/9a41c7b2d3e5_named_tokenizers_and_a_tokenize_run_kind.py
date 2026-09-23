"""named tokenizers and a tokenize run kind

Revision ID: 9a41c7b2d3e5
Revises: 725dbdd4e839
Create Date: 2026-08-14 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9a41c7b2d3e5'
down_revision: Union[str, Sequence[str], None] = '725dbdd4e839'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'tokenizers',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('vocab_size', sa.Integer(), nullable=True),
        sa.Column('merges', sa.Integer(), nullable=True),
        sa.Column('max_vocab', sa.Integer(), nullable=True),
        sa.Column('fingerprint', sa.String(length=64), nullable=True),
        sa.Column(
            'datasets',
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
            nullable=True,
        ),
        sa.Column('implicit', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )

    # Postgres holds `kind` as a real ENUM and a new value has to be added to the
    # type itself. SQLite has no enum to alter — SQLAlchemy emits no CHECK for
    # one — so the tests' schema needs nothing here.
    #
    # `IF NOT EXISTS` so re-running against a database where someone already
    # added it by hand is a no-op rather than a failure, and last among the
    # statements touching this type: a value added in a transaction cannot be
    # *used* until it commits.
    if op.get_bind().dialect.name == 'postgresql':
        op.execute("ALTER TYPE run_kind ADD VALUE IF NOT EXISTS 'tokenize'")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('tokenizers')
    # The enum value stays. Postgres cannot drop one, and recreating the type
    # means rewriting every `runs` row to remove a value nothing is using.
