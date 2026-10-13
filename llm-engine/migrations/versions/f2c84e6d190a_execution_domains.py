"""Keep stable job IDs while specializing model and vocabulary production."""
from alembic import op
import sqlalchemy as sa
revision = 'f2c84e6d190a'
down_revision = 'e1b9d384ac02'
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if connection.execute(sa.text("SELECT count(*) FROM runs WHERE kind='chat'")).scalar_one():
        raise RuntimeError('Unexpected legacy chat jobs require explicit archival before migration')
    op.rename_table('runs','execution_jobs')
    op.execute('ALTER INDEX ix_runs_status RENAME TO ix_execution_jobs_status')
    op.execute('''CREATE TABLE training_runs (
        job_id uuid PRIMARY KEY REFERENCES execution_jobs(id) ON DELETE CASCADE,
        training_type varchar(12) NOT NULL CHECK(training_type IN ('pretrain','sft','rl')),
        config jsonb NOT NULL, result jsonb, target_model_id uuid,
        model_id uuid REFERENCES models(id) ON DELETE SET NULL)''')
    op.execute('''CREATE TABLE tokenizer_builds (
        job_id uuid PRIMARY KEY REFERENCES execution_jobs(id) ON DELETE CASCADE,
        config jsonb NOT NULL, result jsonb, target_tokenizer_id uuid,
        tokenizer_id uuid REFERENCES tokenizers(id) ON DELETE SET NULL)''')
    op.execute('''CREATE TABLE benchmark_dispatches (
        job_id uuid PRIMARY KEY REFERENCES execution_jobs(id) ON DELETE CASCADE,
        config jsonb NOT NULL, benchmark_id uuid)''')
    op.execute('''INSERT INTO training_runs SELECT id,
        CASE WHEN kind='rl' THEN 'rl' WHEN config->>'base_model' IS NOT NULL THEN 'sft' ELSE 'pretrain' END,
        config,result,CASE WHEN config->>'model_name' ~* '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$' THEN (config->>'model_name')::uuid END,
        CASE WHEN ref_id IN (SELECT id FROM models) THEN ref_id END
        FROM execution_jobs WHERE kind IN ('train','rl')''')
    op.execute('''INSERT INTO tokenizer_builds SELECT id,config,result,
        CASE WHEN config->>'tokenizer_id' ~* '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$' THEN (config->>'tokenizer_id')::uuid END,
        CASE WHEN ref_id IN (SELECT id FROM tokenizers) THEN ref_id END
        FROM execution_jobs WHERE kind='tokenize' ''')
    op.execute("INSERT INTO benchmark_dispatches SELECT id,config,ref_id FROM execution_jobs WHERE kind='benchmark'")
    for column in ('config','result','ref_id'):
        op.drop_column('execution_jobs',column)
    op.execute("ALTER TABLE execution_jobs ALTER COLUMN kind TYPE text USING kind::text")
    op.execute('DROP TYPE run_kind')
    op.execute("CREATE TYPE run_kind AS ENUM ('train','rl','tokenize','benchmark')")
    op.execute('ALTER TABLE execution_jobs ALTER COLUMN kind TYPE run_kind USING kind::run_kind')


def downgrade():
    raise RuntimeError('Forward-only execution ownership migration; restore a database backup')
