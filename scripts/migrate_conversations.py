"""Offline cutover: copy and compare every row before retiring source tables.

Stop both writers, apply backend migrations, then run with --apply from the
LLM engine's virtualenv. A repeat validates existing rows instead of overwriting.
The LLM Alembic retirement revision refuses nonempty, unverified source tables.
"""
import argparse
import os

import psycopg
from psycopg import sql


def copy(source, target, table, columns, select):
    names = sql.SQL(',').join(map(sql.Identifier, columns))
    insert = sql.SQL('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT(id) DO NOTHING').format(
        sql.Identifier(table), names, sql.SQL(',').join(sql.Placeholder() for _ in columns))
    lookup = sql.SQL('SELECT {} FROM {} WHERE id=%s').format(names, sql.Identifier(table))
    count = 0
    with source.cursor(name=f'copy_{table}') as rows:
        rows.execute(select)
        for row in rows:
            target.execute(insert, row)
            stored = target.execute(lookup, (row[0],)).fetchone()
            if tuple(stored) != tuple(row):
                raise RuntimeError(f'{table} {row[0]} differs; no rows were overwritten')
            count += 1
    print(f'{table}: {count} rows copied and compared')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    source_url = os.environ['LLM_ENGINE_DATABASE_URL'].replace('postgresql+psycopg:', 'postgresql:')
    with psycopg.connect(source_url) as source:
        # Lock writers through target commit and source retirement. After a
        # crash before retirement, rerunning performs the same comparisons.
        source.execute('LOCK TABLE chats,messages,attachments IN ACCESS EXCLUSIVE MODE')
        with psycopg.connect(os.environ['DATABASE_URL']) as target:
            copy(source,target,'chats',('id','owner_id','title','created_at'),
                 'SELECT id,owner_id,title,created_at FROM chats ORDER BY id')
            copy(source,target,'messages',('id','chat_id','position','role','content','tokens','model_id','model_name','feedback','created_at'),
                 '''SELECT m.id,m.chat_id,row_number() OVER (PARTITION BY m.chat_id ORDER BY m.created_at,m.id),
                 m.role::text,m.content,m.tokens,m.model_id,models.name,m.feedback::text,m.created_at
                 FROM messages m LEFT JOIN models ON models.id=m.model_id ORDER BY m.chat_id,m.created_at,m.id''')
            copy(source,target,'attachments',('id','owner_id','message_id','filename','media_type','size','content','created_at'),
                 'SELECT id,owner_id,message_id,filename,media_type,size,content,created_at FROM attachments ORDER BY id')
        # Retain recovery generations referenced by the newly remote history.
        source.execute("UPDATE models SET parameters=COALESCE(parameters,'{}'::jsonb)||'{\"recovery_referenced\":true}'::jsonb WHERE id IN (SELECT model_id FROM messages)")
        # Rename instead of delete: an operator can restore these after a failed
        # deployment. The new application and ORM cannot write the archive.
        for table in ('attachments','messages','chats'):
            source.execute(sql.SQL('ALTER TABLE {} RENAME TO {}').format(sql.Identifier(table),sql.Identifier('migrated_'+table)))
    print('Copy committed; source tables archived. Apply LLM Alembic migrations before starting services.')


if __name__ == '__main__':
    main()
