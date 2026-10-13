"""Offline, exact-copy benchmark ownership transfer; archives source rows."""
import argparse
import os
import shutil
from pathlib import Path
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

COLUMNS = ('id','model_id','model_name','suite','suite_version','status','score','report',
           'eval_tokens','train_tokens','eval_conversations','error','created_at','completed_at')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',required=True)
    parser.parse_args()
    def url(name):
        return os.environ[name].replace('postgresql+psycopg:','postgresql:')
    root = Path(__file__).resolve().parents[1]
    old = Path(os.environ.get('LEGACY_BENCH_CORPUS_DIR',str(root/'llm-engine/data/eval/corpus')))
    new = Path(os.environ.get('BENCH_CORPUS_DIR',str(root/'bench-engine/data/corpus')))
    if old.is_dir():
        new.mkdir(parents=True,exist_ok=True)
        for source_file in old.glob('*.txt'):
            target_file = new/source_file.name
            if target_file.exists() and target_file.read_bytes() != source_file.read_bytes():
                raise RuntimeError(f'Corpus target differs: {target_file}; source left untouched')
            shutil.copy2(source_file,target_file)
    names = sql.SQL(',').join(map(sql.Identifier,COLUMNS))
    with psycopg.connect(url('LLM_ENGINE_DATABASE_URL')) as source:
        source.execute('LOCK TABLE benchmarks IN ACCESS EXCLUSIVE MODE')
        with psycopg.connect(url('BENCH_ENGINE_DATABASE_URL')) as target:
            insert = sql.SQL('INSERT INTO benchmarks ({}) VALUES ({}) ON CONFLICT(id) DO NOTHING').format(names,sql.SQL(',').join(sql.Placeholder() for _ in COLUMNS))
            with source.cursor(name='transfer_benchmarks') as rows:
                rows.execute(sql.SQL('SELECT {} FROM benchmarks ORDER BY id').format(names))
                count = 0
                for row in rows:
                    args = list(row)
                    args[7] = Jsonb(args[7]) if args[7] is not None else None
                    target.execute(insert,args)
                    found = target.execute(sql.SQL('SELECT {} FROM benchmarks WHERE id=%s').format(names),(row[0],)).fetchone()
                    if tuple(found) != tuple(row):
                        raise RuntimeError(f'benchmark {row[0]} differs; target not overwritten')
                    count += 1
        source.execute("UPDATE models SET parameters=COALESCE(parameters,'{}'::jsonb)||'{\"recovery_referenced\":true}'::jsonb WHERE id IN (SELECT model_id FROM benchmarks)")
        source.execute('ALTER TABLE benchmarks RENAME TO migrated_benchmarks')
    print(f'{count} benchmark rows/reports compared and copied; source archived')


if __name__ == '__main__':
    main()
