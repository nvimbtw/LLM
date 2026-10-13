"""Administrative ownership transfer inside the existing LLM Compose image.

Called by apply_sprint_cutover.sh after backup verification and writer shutdown.
Database names/credentials follow this repository's Compose configuration.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]


def connection_url(database):
    source = urlsplit(os.environ['LLM_ENGINE_DATABASE_URL'])
    return urlunsplit(source._replace(scheme='postgresql', path=f'/{database}'))


def command(*args, directory=ROOT):
    subprocess.run([sys.executable, *args], cwd=directory, check=True)


def migrate(module):
    command('-m', 'alembic', 'upgrade', 'head', directory=ROOT / module)


def table_exists(db, name):
    return db.execute('SELECT to_regclass(%s) IS NOT NULL', (f'public.{name}',)).fetchone()[0]


def transfer(source, table, script):
    if table_exists(source, table):
        command(str(ROOT / 'scripts' / script), '--apply')
    elif table_exists(source, f'migrated_{table}'):
        print(f'{table}: previous transfer archive exists; leaving it intact', flush=True)
    else:
        raise RuntimeError(f'Expected {table} or its transfer archive; refusing an unknown schema')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    os.environ['DATABASE_URL'] = connection_url('llm')
    os.environ['BENCH_ENGINE_DATABASE_URL'] = connection_url('slate_bench').replace('postgresql:', 'postgresql+psycopg:')

    with psycopg.connect(connection_url('llm')) as backend:
        applied = backend.execute(
            'SELECT success FROM _sqlx_migrations WHERE version=20261009000000'
        ).fetchone()
        if applied != (True,):
            raise RuntimeError('Backend conversation migration must be applied before this transfer')
    with psycopg.connect(connection_url('postgres'), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname='slate_bench'").fetchone():
            admin.execute('CREATE DATABASE slate_bench')
    with psycopg.connect(connection_url('slate_bench')) as bench:
        if table_exists(bench, 'alembic_version'):
            revisions = {row[0] for row in bench.execute('SELECT version_num FROM alembic_version')}
            supported = {'001_initial', '002_fix_suite_enum', '003_suite_to_varchar', '004_sprint_ownership'}
            if not revisions <= supported:
                raise RuntimeError(f'Unsupported benchmark schema revisions: {sorted(revisions)}')

    migrate('bench-engine')
    # The administrative connection holds no snapshot across child transactions.
    with psycopg.connect(connection_url('slate_engine'), autocommit=True) as source:
        transfer(source, 'chats', 'migrate_conversations.py')
        transfer(source, 'benchmarks', 'migrate_benchmarks.py')
    migrate('llm-engine')
    command(str(ROOT / 'scripts' / 'migrate_legacy_bench_jobs.py'), '--apply')

    with psycopg.connect(connection_url('slate_bench')) as bench:
        model_ids = [row[0] for row in bench.execute('SELECT DISTINCT model_id FROM benchmarks')]
    with psycopg.connect(connection_url('slate_engine')) as source:
        source.execute("UPDATE models SET parameters=COALESCE(parameters,'{}'::jsonb)||'{\"recovery_referenced\":true}'::jsonb WHERE id=ANY(%s::uuid[])", (model_ids,))

    for database, tables in (
        ('llm', ('chats', 'messages', 'attachments')),
        ('slate_engine', ('execution_jobs', 'training_runs', 'tokenizer_builds', 'benchmark_dispatches')),
        ('slate_bench', ('benchmarks',)),
    ):
        with psycopg.connect(connection_url(database)) as db:
            for table in tables:
                count = db.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                print(f'{database}.{table}: {count}', flush=True)
    print('Ownership transfer complete. Source archives retained. No tests executed.', flush=True)


if __name__ == '__main__':
    main()
