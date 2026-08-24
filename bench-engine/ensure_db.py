"""Ensure slate_bench exists before alembic."""
import os
import sys
import time
from sqlalchemy import create_engine, text

def ensure():
    url = os.environ.get("BENCH_ENGINE_DATABASE_URL", "postgresql+psycopg://slate:slate@postgres:5432/slate_bench")
    if "slate_bench" in url:
        admin_url = url.replace("slate_bench", "postgres")
    else:
        admin_url = url
    for attempt in range(5):
        try:
            eng = create_engine(admin_url, isolation_level="AUTOCOMMIT")
            with eng.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname='slate_bench'"))
                exists = result.scalar() is not None
                if not exists:
                    print("creating database slate_bench...")
                    conn.execute(text("CREATE DATABASE slate_bench"))
                    print("created slate_bench")
                else:
                    print("slate_bench already exists")
                # Ensure suite column is varchar (fixes enum mismatch) and has needed values
                try:
                    bench_url = url
                    bench_eng = create_engine(bench_url, isolation_level="AUTOCOMMIT")
                    with bench_eng.connect() as bconn:
                        # If suite is still an enum, convert to varchar
                        bconn.execute(text("""
                            DO $$ BEGIN
                                IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'suite') THEN
                                    BEGIN
                                        ALTER TABLE benchmarks ALTER COLUMN suite TYPE VARCHAR(20) USING suite::text;
                                    EXCEPTION WHEN OTHERS THEN
                                        NULL;
                                    END;
                                    BEGIN
                                        DROP TYPE IF EXISTS suite;
                                    EXCEPTION WHEN OTHERS THEN
                                        NULL;
                                    END;
                                END IF;
                            END $$;
                        """))
                        bconn.execute(text("ALTER TYPE suite ADD VALUE IF NOT EXISTS 'language_v3'"))
                        bconn.execute(text("ALTER TYPE suite ADD VALUE IF NOT EXISTS 'language-v3'"))
                except Exception as e:
                    print(f"ensure suite enum/varchar: {e}", file=sys.stderr)
                return 0
        except Exception as e:
            print(f"ensure_db attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2)
    return 1

if __name__ == "__main__":
    sys.exit(ensure())
