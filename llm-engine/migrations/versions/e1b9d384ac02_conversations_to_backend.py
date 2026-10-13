"""Retire conversation persistence after an offline verified copy."""
from alembic import op
import sqlalchemy as sa
revision = 'e1b9d384ac02'
down_revision = 'd8c10f6a21b4'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    tables = sa.inspect(connection).get_table_names()
    for table in ('attachments', 'messages', 'chats'):
        if table not in tables:
            continue  # The verified transfer archived it.
        if connection.execute(sa.text(f'SELECT count(*) FROM {table}')).scalar_one():
            raise RuntimeError('Run scripts/migrate_conversations.py offline before retiring conversations')
        op.drop_table(table)


def downgrade():
    raise RuntimeError('Forward-only ownership cutover; restore the documented database backups')
