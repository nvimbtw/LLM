"""checkpoints are models

Revision ID: b7d2e5f10a94
Revises: f1a3c07be215
Create Date: 2026-08-29 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7d2e5f10a94'
down_revision: Union[str, Sequence[str], None] = 'f1a3c07be215'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # All four are nullable and every existing row keeps them null, which is
    # exactly what it means to be a finished model rather than an intermediate.
    #
    # Neither uuid is a foreign key, for two different reasons. `run_id` is not
    # one because run rows are prunable and the grouping must outlive them —
    # the same reason `runs.ref_id` is a bare uuid. `parent_id` is not one
    # because the parent row does not exist yet when a run's first checkpoint
    # lands, so the constraint would refuse the insert it is meant to describe.
    op.add_column('models', sa.Column('run_id', sa.Uuid(), nullable=True))
    op.add_column('models', sa.Column('parent_id', sa.Uuid(), nullable=True))
    op.add_column('models', sa.Column('checkpoint_step', sa.Integer(), nullable=True))
    op.add_column('models', sa.Column('checkpoint_index', sa.Integer(), nullable=True))
    # The two that are queried: `run_id` groups a run's output, `parent_id`
    # answers "the checkpoints of this model". `checkpoint_index` is only ever
    # tested for null, which no index helps on a column that is null for almost
    # every row.
    op.create_index('ix_models_run_id', 'models', ['run_id'])
    op.create_index('ix_models_parent_id', 'models', ['parent_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_models_parent_id', table_name='models')
    op.drop_index('ix_models_run_id', table_name='models')
    op.drop_column('models', 'checkpoint_index')
    op.drop_column('models', 'checkpoint_step')
    op.drop_column('models', 'parent_id')
    op.drop_column('models', 'run_id')
