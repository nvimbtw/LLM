"""Add language_v3 underscore variant to suite enum."""
revision = "002_fix_suite_enum"
down_revision = "001_initial"
branch_labels = None
depends_on = None

from alembic import op

def upgrade():
    op.execute("ALTER TYPE suite ADD VALUE IF NOT EXISTS 'language_v3'")

def downgrade():
    # Postgres cannot remove enum values; leave as is
    pass
