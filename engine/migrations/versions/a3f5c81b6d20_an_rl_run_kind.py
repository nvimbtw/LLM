"""an rl run kind

Revision ID: a3f5c81b6d20
Revises: b7d2e5f10a94
Create Date: 2026-09-04

A run that practises against a gym rather than reading a corpus. It writes
checkpoints and produces a model exactly as a training run does, so nothing that
consumes a model changes; what makes it a kind of its own is that its frames
carry a pass rate where a training run's carry a loss, and no consumer could tell
those apart from the numbers.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a3f5c81b6d20"
down_revision: Union[str, Sequence[str], None] = "b7d2e5f10a94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres holds `kind` as a real ENUM and a new value has to be added to
    # the type itself. SQLite has no enum to alter, so the tests' schema needs
    # nothing here — the same split `9a41c7b2d3e5` documents for `tokenize`.
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TYPE run_kind ADD VALUE IF NOT EXISTS 'rl'")


def downgrade() -> None:
    # The value stays. Postgres cannot drop one, and recreating the type means
    # rewriting every `runs` row to remove a value nothing is using.
    pass
