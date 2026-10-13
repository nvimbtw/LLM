"""Preserve legacy reports while moving scheduling to the shared LLM queue."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '004_sprint_ownership'
down_revision = '003_suite_to_varchar'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    live = connection.execute(sa.text(
        "SELECT count(*) FROM runs WHERE status NOT IN ('done','failed','cancelled')"
    )).scalar_one()
    if live:
        raise RuntimeError('Resolve legacy benchmark work before retiring its scheduler')
    op.add_column('benchmarks', sa.Column('config', JSONB(), nullable=True))
    op.add_column('benchmarks', sa.Column('train_tokens', sa.BigInteger(), nullable=True))
    op.execute('ALTER TABLE benchmarks ALTER COLUMN suite TYPE varchar(32)')
    op.execute('ALTER TABLE benchmarks ALTER COLUMN status TYPE varchar(32) USING status::text')
    op.execute('ALTER TABLE benchmarks ALTER COLUMN report TYPE jsonb USING report::jsonb')
    op.execute('DROP TYPE benchmark_status')
    op.drop_table('queue_state')
    if connection.execute(sa.text('SELECT count(*) FROM runs')).scalar_one():
        # The administrative cutover copies these terminal jobs into execution_jobs,
        # comparing every shared field before retaining this source as an archive.
        op.rename_table('runs', 'migrated_execution_jobs')
    else:
        op.drop_table('runs')
        op.execute('DROP TYPE run_kind')
        op.execute('DROP TYPE run_status')


def downgrade():
    raise RuntimeError('Forward-only ownership migration; restore the database backup')
