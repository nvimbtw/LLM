"""Copy what the LLM engine now owns out of the backend's database.

Additive and idempotent: a row that is already here is skipped, and nothing is
removed from the source. Dropping the old tables is a separate, later decision —
copying can be repeated, dropping cannot be undone.

    python -m scripts.import_from_backend postgres://user@host/llm [--dry-run]

Queue rows are deliberately not copied. A `jobs` row records work the old queue
did, in a schema that no longer describes anything the LLM engine runs.
"""

import argparse
import sys

import psycopg
from psycopg.rows import dict_row

import store
from store.schema import Benchmark, BenchmarkStatus, Chat, Message, Model, Role, Suite


def main() -> int:
    args = _arguments()
    store.configure()

    with psycopg.connect(args.source, row_factory=dict_row) as source, store.session() as db:
        # Models and chats first: a message points at both.
        counts = {
            "models": _copy(db, Model, _read(source, "models"), "id_model", _model),
            "chats": _copy(db, Chat, _read(source, "chats"), "id_chat", _chat),
            "messages": _copy(db, Message, _read(source, "messages"), "id_message", _message),
            "benchmarks": _copy(
                db, Benchmark, _read(source, "benchmark_runs"), "id_run", _benchmark
            ),
        }
        if args.dry_run:
            db.rollback()

    for name, copied in counts.items():
        print(f"{name}: {copied} {'would be copied' if args.dry_run else 'copied'}")
    return 0


def _model(row) -> Model:
    return Model(
        id=row["id_model"],
        name=row["name"],
        description=row["description"],
        family=row["family"],
        parameters=row["parameters"],
        context=row["context"],
        published=row["published"],
        created_at=row["created_at"],
    )


def _chat(row) -> Chat:
    return Chat(
        id=row["id_chat"],
        owner_id=row["id_user"],
        title=row["title"],
        created_at=row["created_at"],
    )


def _message(row) -> Message:
    return Message(
        id=row["id_message"],
        chat_id=row["id_chat"],
        # The old schema had no role column: a turn crediting a model is a
        # reply, and everything else is a question.
        role=Role.assistant if row["id_model"] else Role.user,
        model_id=row["id_model"],
        content=row["content"],
        tokens=row["tokens"],
        feedback=row["feedback"],
        created_at=row["created_at"],
    )


def _benchmark(row) -> Benchmark:
    return Benchmark(
        id=row["id_run"],
        model_id=row["id_model"],
        model_name=row["model_name"],
        suite=Suite(row["suite"]),
        suite_version=row["suite_version"],
        status=BenchmarkStatus(row["status"]),
        score=row["score"],
        report=row["report"],
        eval_tokens=row["eval_tokens"],
        train_tokens=row["train_tokens"],
        eval_conversations=row["eval_conversations"],
        error=row["error"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def _read(source, table: str) -> list[dict]:
    with source.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table}")
        return cursor.fetchall()


def _copy(db, model, rows, id_column: str, build) -> int:
    copied = 0
    for row in rows:
        if db.get(model, row[id_column]) is not None:
            continue
        db.add(build(row))
        copied += 1
    db.flush()
    return copied


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="the backend's DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true", help="count without writing")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
