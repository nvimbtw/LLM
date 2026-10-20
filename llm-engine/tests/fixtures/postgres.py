"""Uniquely owned disposable PostgreSQL databases for migration/locking contracts."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@contextmanager
def database() -> Iterator[str]:
    """Create one uniquely named database, dropping only its matching ownership marker on exit.

    Failure to reach PostgreSQL is a required-check failure rather than a silent
    skip. Template0 avoids inheriting a modified application schema/collation.
    """
    admin_url = make_url(os.environ.get("LLM_ENGINE_TEST_ADMIN_URL", "postgresql+psycopg:///postgres"))
    token = uuid.uuid4().hex
    name = f"slate_arch_{os.getpid()}_{token[:12]}"
    marker = f"modular-architecture-test:{token}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    created = False
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}" TEMPLATE template0'))
            created = True
            connection.execute(text(f"COMMENT ON DATABASE \"{name}\" IS '{marker}'"))
        yield admin_url.set(database=name).render_as_string(hide_password=False)
    finally:
        if created:
            with admin.connect() as connection:
                actual = connection.execute(text("SELECT shobj_description(oid, 'pg_database') FROM pg_database WHERE datname=:name"), {"name": name}).scalar_one_or_none()
                if actual != marker:
                    raise RuntimeError("refusing to drop a database with another ownership marker")
                connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
