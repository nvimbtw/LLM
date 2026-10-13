revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
def upgrade():
    op.create_table("benchmarks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_id", sa.Uuid(), nullable=False, index=True),
        sa.Column("model_name", sa.String(120), nullable=False),
        sa.Column("suite", sa.Enum("language","language-v3", name="suite"), nullable=False),
        sa.Column("suite_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.Enum("running","done","failed", name="benchmark_status"), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("eval_tokens", sa.BigInteger(), nullable=True),
        sa.Column("eval_conversations", sa.BigInteger(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table("runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.Enum("benchmark", name="run_kind"), nullable=False),
        sa.Column("status", sa.Enum("waiting","queued","running","cancelling","done","cancelled","failed", name="run_status"), nullable=False, index=True),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("rank", sa.Float(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("stage", sa.String(120), nullable=True),
        sa.Column("ref_id", sa.Uuid(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("depends_on", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["depends_on"], ["runs.id"], ondelete="SET NULL"),
    )
    op.create_table("queue_state",
        sa.Column("only_row", sa.Boolean(), primary_key=True),
        sa.Column("paused", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
def downgrade():
    op.drop_table("queue_state")
    op.drop_table("runs")
    op.drop_table("benchmarks")
    op.execute("DROP TYPE IF EXISTS suite")
    op.execute("DROP TYPE IF EXISTS benchmark_status")
    op.execute("DROP TYPE IF EXISTS run_kind")
    op.execute("DROP TYPE IF EXISTS run_status")
