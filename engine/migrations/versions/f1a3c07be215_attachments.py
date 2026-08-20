"""attachments

Revision ID: f1a3c07be215
Revises: c4e91a70f2bd
Create Date: 2026-08-20 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f1a3c07be215'
down_revision: Union[str, Sequence[str], None] = 'c4e91a70f2bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # `message_id` is nullable because a file is uploaded before the turn it is
    # sent with exists — on a new conversation, before the chat does. `owner_id`
    # is what scopes a row in both states, which is why it is indexed and not
    # merely carried.
    op.create_table(
        'attachments',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('owner_id', sa.Uuid(), nullable=False),
        sa.Column('message_id', sa.Uuid(), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('media_type', sa.String(length=160), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('content', sa.LargeBinary(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_attachments_owner_id', 'attachments', ['owner_id'])
    op.create_index('ix_attachments_message_id', 'attachments', ['message_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_attachments_message_id', table_name='attachments')
    op.drop_index('ix_attachments_owner_id', table_name='attachments')
    op.drop_table('attachments')
