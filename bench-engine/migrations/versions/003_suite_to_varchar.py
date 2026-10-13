"""Convert suite enum to varchar to avoid native enum issues."""
revision = "003_suite_to_varchar"
down_revision = "002_fix_suite_enum"
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'suite') THEN
                ALTER TABLE benchmarks ALTER COLUMN suite TYPE VARCHAR(20) USING suite::text;
                DROP TYPE IF EXISTS suite;
            END IF;
        END $$;
    """)

def downgrade():
    op.execute("CREATE TYPE suite AS ENUM ('language', 'language-v3', 'language_v3')")
    op.execute("ALTER TABLE benchmarks ALTER COLUMN suite TYPE suite USING suite::suite")

