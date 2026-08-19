"""Pulling rows off the Hub, one shard at a time.

Lazy on purpose, exactly as `scripts/corpus_shards.py` is: a shard is hundreds of
megabytes and a capped pull has no reason to fetch the whole repo to keep a
fraction of one. The caller stops iterating when its budget is spent and the
rest are never downloaded.

Every pull is bounded. There is no "fetch it all" — a corpus arriving as
unbounded rows into a database is the failure mode this module exists to avoid.
"""

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from hub.search import HubError, PARQUET_SUFFIX
from store.schema import DatasetKind

#: Rows per arrow batch. Not the database batch size — that is the caller's.
READ_BATCH = 256

#: Column names a source is likely to put prose under, tried in order when the
#: caller names none.
TEXT_COLUMNS = ("text", "content", "raw_content", "document", "article")

#: The same for conversations.
MESSAGE_COLUMNS = ("messages", "conversations", "conversation", "turns", "dialogue")

#: Role/content key pairs seen in the wild. ShareGPT-style dumps use from/value.
_TURN_KEYS = (("role", "content"), ("from", "value"))

#: How much of a stray string column is worth keeping alongside a document.
MAX_META_CHARS = 500


def shards_under(repo: str, prefix: str | None) -> list[str]:
    """The parquet shards of a repo, sorted, optionally under one directory.

    Sorted so the order is stable: a pull capped at N rows must take the same N
    rows every time, or re-pulling a dataset silently produces a different one.
    """
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        raise HubError(str(e)) from e

    shards = sorted(
        f
        for f in files
        if f.endswith(PARQUET_SUFFIX) and (not prefix or f.startswith(prefix))
    )
    if not shards:
        where = f" under {prefix!r}" if prefix else ""
        raise HubError(f"no parquet shards found in {repo}{where}")
    return shards


def iter_rows(repo: str, shards: list[str], columns: list[str] | None) -> Iterator[dict]:
    """Row dicts, downloading one shard at a time.

    `columns` is a projection, not a requirement: a shard missing one of them is
    read whole rather than failing, because a repo whose shards disagree about
    their schema is common and is not the caller's problem to pre-empt.
    """
    for shard in shards:
        try:
            local = hf_hub_download(repo, filename=shard, repo_type="dataset")
            handle = pq.ParquetFile(local)
        except Exception as e:
            raise HubError(f"{shard}: {e}") from e

        available = set(handle.schema_arrow.names)
        wanted = [c for c in (columns or []) if c in available] or None

        for batch in handle.iter_batches(batch_size=READ_BATCH, columns=wanted):
            table = batch.to_pydict()
            names = list(table)
            for values in zip(*(table[n] for n in names)):
                yield dict(zip(names, values))


def pick_column(row: dict, named: str | None, candidates: tuple[str, ...]) -> str | None:
    """Which column the text is in: the one asked for, or the first plausible one.

    Resolved against a real row rather than the card, because the card is often
    absent and always optional, while the row is the thing being read.
    """
    if named:
        return named if named in row else None
    return next((c for c in candidates if c in row), None)


def as_document(row: dict, kind: DatasetKind, column: str) -> dict | None:
    """One Hub row as a document, or `None` if it carries nothing usable."""
    if kind == DatasetKind.pretrain:
        return _prose(row, column)
    return _conversation(row, column)


def _prose(row: dict, column: str) -> dict | None:
    text = row.get(column)
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "content": text,
        "messages": None,
        "meta": _meta(row, {column}),
        "byte_size": len(text.encode("utf-8")),
    }


def _conversation(row: dict, column: str) -> dict | None:
    raw = row.get(column)
    if not isinstance(raw, list) or not raw:
        return None

    messages = [turn for turn in map(_turn, raw) if turn is not None]
    if not messages:
        return None

    size = sum(len(m["content"].encode("utf-8")) for m in messages)
    return {
        "content": None,
        "messages": messages,
        "meta": _meta(row, {column}),
        "byte_size": size,
    }


def _turn(raw: Any) -> dict | None:
    """One turn, under whichever pair of keys the source used.

    Roles are passed through as the source wrote them — mapping `human` onto
    `user` is the `normalise-roles` processor's job, and doing it here would hide
    from the user that it happened.
    """
    if not isinstance(raw, dict):
        return None
    for role_key, content_key in _TURN_KEYS:
        if role_key in raw and content_key in raw:
            content = raw.get(content_key)
            if not isinstance(content, str):
                return None
            return {"role": str(raw.get(role_key)), "content": content}
    return None


def _meta(row: dict, exclude: set[str]) -> dict:
    """The row's other columns, minus the one that became the content.

    Only scalars, and only short ones: a nested structure or a second text column
    here would be another copy of the payload in a column nothing queries, which
    is how a metadata field doubles the size of a corpus.
    """
    return {
        key: value[:MAX_META_CHARS] if isinstance(value, str) else value
        for key, value in row.items()
        if key not in exclude and isinstance(value, (str, int, float, bool)) and value != ""
    }
