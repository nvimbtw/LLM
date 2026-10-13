"""Copy archived legacy benchmark jobs into the authoritative execution queue."""
import argparse
import os

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

FIELDS = ('id', 'kind', 'status', 'label', 'rank', 'progress', 'stage', 'error',
          'owner_id', 'depends_on', 'created_at', 'started_at', 'finished_at')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    def url(name):
        return os.environ[name].replace('postgresql+psycopg:', 'postgresql:')
    with psycopg.connect(url('BENCH_ENGINE_DATABASE_URL')) as source:
        if source.execute("SELECT to_regclass('public.migrated_execution_jobs')").fetchone()[0] is None:
            print('No legacy benchmark jobs to transfer')
            return
        source.execute('LOCK TABLE migrated_execution_jobs IN ACCESS EXCLUSIVE MODE')
        names = sql.SQL(',').join(map(sql.Identifier, FIELDS))
        rows = source.execute(sql.SQL('SELECT {},config,ref_id FROM migrated_execution_jobs ORDER BY created_at,id').format(names)).fetchall()
        with psycopg.connect(url('LLM_ENGINE_DATABASE_URL')) as target:
            existing = set()
            for row in rows:
                if row[2] not in ('done', 'failed', 'cancelled'):
                    raise RuntimeError('Only terminal legacy jobs can be transferred')
                found = target.execute(sql.SQL('SELECT {} FROM execution_jobs WHERE id=%s').format(names), (row[0],)).fetchone()
                if found is not None:
                    if tuple(found) != tuple(row[:len(FIELDS)]):
                        raise RuntimeError(f'Execution {row[0]} differs; target not overwritten')
                    existing.add(row[0])
                    continue
                values = list(row[:len(FIELDS)])
                values[FIELDS.index('depends_on')] = None
                target.execute(sql.SQL('INSERT INTO execution_jobs ({}) VALUES ({})').format(
                    names, sql.SQL(',').join(sql.Placeholder() for _ in FIELDS)), values)
            for row in rows:
                if row[0] not in existing:
                    target.execute('UPDATE execution_jobs SET depends_on=%s WHERE id=%s', (row[FIELDS.index('depends_on')], row[0]))
                target.execute('INSERT INTO benchmark_dispatches(job_id,config,benchmark_id) VALUES (%s,%s,%s) ON CONFLICT(job_id) DO NOTHING',
                               (row[0], Jsonb(row[-2]), row[-1]))
                found = target.execute('SELECT config,benchmark_id FROM benchmark_dispatches WHERE job_id=%s', (row[0],)).fetchone()
                if tuple(found) != tuple(row[-2:]):
                    raise RuntimeError(f'Benchmark dispatch {row[0]} differs; target not overwritten')
                found = target.execute(sql.SQL('SELECT {} FROM execution_jobs WHERE id=%s').format(names), (row[0],)).fetchone()
                if tuple(found) != tuple(row[:len(FIELDS)]):
                    raise RuntimeError(f'Execution {row[0]} failed comparison')
        # The original result JSON remains in this archive; new result lookup uses
        # benchmark_id and the unchanged report owned by bench-engine.
    print(f'{len(rows)} legacy benchmark jobs copied and compared; source archive retained')


if __name__ == '__main__':
    main()
