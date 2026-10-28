"""modular architecture catalogs and evidence

Revision ID: b8e246d913af
Revises: a4e79c1026bf
Create Date: 2026-10-20 22:05:26.407910

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b8e246d913af'
down_revision: Union[str, Sequence[str], None] = 'a4e79c1026bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Append immutable catalog/evidence tables while preserving legacy nullable definitions."""
    op.execute("ALTER TYPE run_kind ADD VALUE IF NOT EXISTS 'model_validation'")
    op.create_table('architecture_mutation_requests',
    sa.Column('actor_id', sa.Uuid(), nullable=False),
    sa.Column('request_id', sa.Uuid(), nullable=False),
    sa.Column('operation', sa.String(length=32), nullable=False),
    sa.Column('payload_hash', sa.String(length=64), nullable=False),
    sa.Column('outcome', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('actor_id', 'request_id')
    )
    op.create_table('architectures',
    sa.Column('draft_yaml', sa.Text(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('origin', sa.String(length=12), nullable=False),
    sa.Column('archived', sa.Boolean(), nullable=False),
    sa.Column('draft_version', sa.Integer(), nullable=False),
    sa.Column('draft_hash', sa.String(length=64), nullable=False),
    sa.Column('metadata_version', sa.Integer(), nullable=False),
    sa.Column('creator_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("origin IN ('builtin','custom')", name='architectures_origin_check'),
    sa.CheckConstraint('draft_version >= 1 AND metadata_version >= 1', name='architectures_versions_check'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('model_blocks',
    sa.Column('draft_source', sa.Text(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('origin', sa.String(length=12), nullable=False),
    sa.Column('archived', sa.Boolean(), nullable=False),
    sa.Column('draft_version', sa.Integer(), nullable=False),
    sa.Column('draft_hash', sa.String(length=64), nullable=False),
    sa.Column('metadata_version', sa.Integer(), nullable=False),
    sa.Column('creator_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("origin IN ('builtin','custom')", name='model_blocks_origin_check'),
    sa.CheckConstraint('draft_version >= 1 AND metadata_version >= 1', name='model_blocks_versions_check'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('validation_evidence',
    sa.Column('receipt_id', sa.Uuid(), nullable=False),
    sa.Column('identity_hash', sa.String(length=64), nullable=False),
    sa.Column('receipt', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('source_run_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('receipt_id'),
    sa.UniqueConstraint('source_run_id', name='uq_validation_evidence_source_run')
    )
    op.create_index(op.f('ix_validation_evidence_identity_hash'), 'validation_evidence', ['identity_hash'], unique=False)
    op.create_table('architecture_revisions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('architecture_id', sa.Uuid(), nullable=False),
    sa.Column('revision_number', sa.Integer(), nullable=False),
    sa.Column('yaml', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('recipe', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('dependency_digest', sa.String(length=64), nullable=False),
    sa.Column('acceptance_receipt_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('revision_number >= 1', name='architecture_revision_number_check'),
    sa.ForeignKeyConstraint(['acceptance_receipt_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['architecture_id'], ['architectures.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('architecture_id', 'content_hash', 'dependency_digest', name='uq_architecture_revision_content'),
    sa.UniqueConstraint('architecture_id', 'revision_number', name='uq_architecture_revision_number')
    )
    op.create_index(op.f('ix_architecture_revisions_architecture_id'), 'architecture_revisions', ['architecture_id'], unique=False)
    op.create_table('block_revisions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('block_id', sa.Uuid(), nullable=False),
    sa.Column('revision_number', sa.Integer(), nullable=False),
    sa.Column('source', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('declaration', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('compatibility', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('acceptance_receipt_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('revision_number >= 1', name='block_revision_number_check'),
    sa.ForeignKeyConstraint(['acceptance_receipt_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['block_id'], ['model_blocks.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('block_id', 'content_hash', name='uq_block_revision_content'),
    sa.UniqueConstraint('block_id', 'revision_number', name='uq_block_revision_number')
    )
    op.create_index(op.f('ix_block_revisions_block_id'), 'block_revisions', ['block_id'], unique=False)
    op.create_table('model_validations',
    sa.Column('job_id', sa.Uuid(), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('result', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('evidence_id', sa.Uuid(), nullable=True),
    sa.ForeignKeyConstraint(['evidence_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['job_id'], ['execution_jobs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('job_id')
    )
    op.create_table('architecture_dependencies',
    sa.Column('architecture_revision_id', sa.Uuid(), nullable=False),
    sa.Column('instance_path', sa.String(length=512), nullable=False),
    sa.Column('block_revision_id', sa.Uuid(), nullable=True),
    sa.Column('child_architecture_revision_id', sa.Uuid(), nullable=True),
    sa.CheckConstraint('(block_revision_id IS NOT NULL AND child_architecture_revision_id IS NULL) OR (block_revision_id IS NULL AND child_architecture_revision_id IS NOT NULL)', name='architecture_dependency_exactly_one_type'),
    sa.ForeignKeyConstraint(['architecture_revision_id'], ['architecture_revisions.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['block_revision_id'], ['block_revisions.id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['child_architecture_revision_id'], ['architecture_revisions.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('architecture_revision_id', 'instance_path')
    )
    op.create_index(op.f('ix_architecture_dependencies_block_revision_id'), 'architecture_dependencies', ['block_revision_id'], unique=False)
    op.create_index(op.f('ix_architecture_dependencies_child_architecture_revision_id'), 'architecture_dependencies', ['child_architecture_revision_id'], unique=False)
    op.create_table('model_definitions',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('architecture_revision_id', sa.Uuid(), nullable=False),
    sa.Column('definition', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['architecture_revision_id'], ['architecture_revisions.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_definitions_architecture_revision_id'), 'model_definitions', ['architecture_revision_id'], unique=False)
    op.create_table('run_architecture_pins',
    sa.Column('job_id', sa.Uuid(), nullable=False),
    sa.Column('architecture_revision_id', sa.Uuid(), nullable=False),
    sa.Column('acceptance_receipt_id', sa.Uuid(), nullable=False),
    sa.Column('exact_receipt_id', sa.Uuid(), nullable=True),
    sa.Column('evidence', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('parameters', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('snapshot', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['acceptance_receipt_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['architecture_revision_id'], ['architecture_revisions.id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['exact_receipt_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['job_id'], ['execution_jobs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index(op.f('ix_run_architecture_pins_architecture_revision_id'), 'run_architecture_pins', ['architecture_revision_id'], unique=False)
    op.create_table('definition_evidence',
    sa.Column('definition_id', sa.String(length=64), nullable=False),
    sa.Column('receipt_id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['definition_id'], ['model_definitions.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['receipt_id'], ['validation_evidence.receipt_id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('definition_id', 'receipt_id')
    )
    op.add_column('models', sa.Column('definition_id', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_models_definition_id'), 'models', ['definition_id'], unique=False)
    op.create_foreign_key(None, 'models', 'model_definitions', ['definition_id'], ['id'], ondelete='RESTRICT')
    op.add_column('training_runs', sa.Column('definition_id', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_training_runs_definition_id'), 'training_runs', ['definition_id'], unique=False)
    op.create_foreign_key(None, 'training_runs', 'model_definitions', ['definition_id'], ['id'], ondelete='RESTRICT')

    op.execute("""CREATE FUNCTION reject_modular_immutable_update() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'immutable modular architecture record cannot be updated'
            USING ERRCODE='23000';
        END $$""")
    for table in ("block_revisions", "architecture_revisions", "architecture_dependencies",
                  "validation_evidence", "model_definitions", "definition_evidence", "run_architecture_pins"):
        op.execute(f"CREATE TRIGGER {table}_immutable_update BEFORE UPDATE ON {table} "
                   "FOR EACH ROW EXECUTE FUNCTION reject_modular_immutable_update()")


def downgrade() -> None:
    """Refuse destructive rollback of retained sources/evidence; restore a database backup."""
    raise RuntimeError("Forward-only modular architecture migration; restore a database backup")
