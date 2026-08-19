"""datasets, documents and the runs that write them

Revision ID: 28e7f4329a1f
Revises: 
Create Date: 2026-08-19 15:49:04.117036

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '28e7f4329a1f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('datasets',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('kind', sa.Enum('pretrain', 'sft', name='dataset_kind'), nullable=False),
    sa.Column('origin', sa.Enum('huggingface', 'derived', name='dataset_origin'), nullable=False),
    sa.Column('status', sa.Enum('pulling', 'ready', 'failed', name='dataset_status'), nullable=False),
    sa.Column('source_repo', sa.String(length=200), nullable=True),
    sa.Column('source_config', sa.String(length=120), nullable=True),
    sa.Column('source_split', sa.String(length=120), nullable=True),
    sa.Column('parent_id', sa.Uuid(), nullable=True),
    sa.Column('steps', sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('document_count', sa.Integer(), nullable=False),
    sa.Column('byte_size', sa.BigInteger(), nullable=False),
    sa.Column('owner_id', sa.Uuid(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['parent_id'], ['datasets.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_datasets_owner_id'), 'datasets', ['owner_id'], unique=False)
    op.create_index(op.f('ix_datasets_parent_id'), 'datasets', ['parent_id'], unique=False)
    op.create_index(op.f('ix_datasets_status'), 'datasets', ['status'], unique=False)
    op.create_table('runs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('kind', sa.Enum('pull', 'process', name='run_kind'), nullable=False),
    sa.Column('status', sa.Enum('queued', 'running', 'cancelling', 'done', 'cancelled', 'failed', name='run_status'), nullable=False),
    sa.Column('label', sa.String(length=200), nullable=False),
    sa.Column('config', sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('progress', sa.Float(), nullable=False),
    sa.Column('stage', sa.String(length=120), nullable=True),
    sa.Column('ref_id', sa.Uuid(), nullable=True),
    sa.Column('result', sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('owner_id', sa.Uuid(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_runs_owner_id'), 'runs', ['owner_id'], unique=False)
    op.create_index(op.f('ix_runs_status'), 'runs', ['status'], unique=False)
    op.create_table('documents',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('dataset_id', sa.Uuid(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('messages', sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('meta', sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('byte_size', sa.Integer(), nullable=False),
    sa.CheckConstraint('(content IS NULL) <> (messages IS NULL)', name='ck_documents_content_xor_messages'),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'ordinal', name='uq_documents_dataset_ordinal')
    )
    op.create_index(op.f('ix_documents_dataset_id'), 'documents', ['dataset_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_documents_dataset_id'), table_name='documents')
    op.drop_table('documents')
    op.drop_index(op.f('ix_runs_status'), table_name='runs')
    op.drop_index(op.f('ix_runs_owner_id'), table_name='runs')
    op.drop_table('runs')
    op.drop_index(op.f('ix_datasets_status'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_parent_id'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_owner_id'), table_name='datasets')
    op.drop_table('datasets')
