"""Retire benchmarks only after verified transfer, or on an empty database."""
from alembic import op
import sqlalchemy as sa
revision = 'a4e79c1026bf'
down_revision = 'f2c84e6d190a'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if 'benchmarks' not in sa.inspect(connection).get_table_names():
        return
    if connection.execute(sa.text('SELECT count(*) FROM benchmarks')).scalar_one():
        raise RuntimeError('Run scripts/migrate_benchmarks.py before retiring benchmark storage')
    op.drop_table('benchmarks')


def downgrade():
    raise RuntimeError('Forward-only ownership cutover; restore the archived tables and database backup')
