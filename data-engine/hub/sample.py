"""Looking inside a dataset before pulling it.

A search says a repo exists and `detail` says what its card claims; neither
answers the only question that decides the pull form — is this prose or is it
conversations, and which column holds it. The card is optional and often absent,
and guessing wrong writes the wrong shape into the database.

So the rows are read from the Hub's own **datasets-server**, which already holds
the first rows of every public dataset pre-extracted. It is one small HTTP call.
The alternative — `hf_hub_download` of a shard — is hundreds of megabytes to
render a page the user is only glancing at, which is the same reason
`search.detail` refuses to open a shard for its schema.

A dataset the server has not indexed (gated, private, too new) simply has no
sample. That is a missing preview, not a failed pull: `fetch` reads parquet
directly and does not care what this module could not answer.
"""

from typing import Any

import httpx

from hub.search import HubError
from store.schema import DatasetKind

#: The Hub's read-only rows service. No token: public datasets only, which is
#: everything a preview is allowed to show anyway.
SERVER = "https://datasets-server.huggingface.co"

#: Small. This is a glance at a dataset, not a pull.
MAX_ROWS = 20
DEFAULT_ROWS = 5

#: How much of one cell survives. A single FineWeb document is tens of kilobytes
#: and the point of a preview is the shape, not the payload.
MAX_CELL_CHARS = 2000

#: Splits worth defaulting to, in order. Everything else falls back to the first
#: the server lists.
PREFERRED_SPLITS = ("train", "training", "default")

#: Seconds. The server is usually instant and occasionally cold; a preview that
#: hangs is worse than one that says it could not answer.
TIMEOUT = 15.0


def sample(
    repo: str,
    config: str | None = None,
    split: str | None = None,
    rows: int = DEFAULT_ROWS,
) -> dict[str, Any]:
    """The first few rows of one config/split, plus what they imply.

    `config` and `split` are optional: unnamed, the first plausible pair is
    chosen and reported back, so the caller can offer the others without a
    second round trip.
    """
    capped = max(1, min(rows, MAX_ROWS))
    available = splits(repo)
    if not available:
        raise HubError(f"the Hub has no indexed rows for {repo}")

    chosen = _choose(available, config, split)
    body = _get("/first-rows", dataset=repo, config=chosen["config"], split=chosen["split"])

    columns = [
        f["name"] for f in body.get("features", []) if isinstance(f, dict) and "name" in f
    ]
    sampled = [_row(entry.get("row"), columns) for entry in body.get("rows", [])[:capped]]

    return {
        "repo": repo,
        "config": chosen["config"],
        "split": chosen["split"],
        "splits": available,
        "columns": columns,
        "rows": sampled,
        "truncated": bool(body.get("truncated_cells")) or _any_truncation(sampled),
        **_suggestion(sampled),
    }


def splits(repo: str) -> list[dict[str, str]]:
    """Every config/split pair the server has indexed, as `{config, split}`."""
    body = _get("/splits", dataset=repo)
    return [
        {"config": entry["config"], "split": entry["split"]}
        for entry in body.get("splits", [])
        if isinstance(entry, dict) and "config" in entry and "split" in entry
    ]


def _choose(
    available: list[dict[str, str]], config: str | None, split: str | None
) -> dict[str, str]:
    """The pair asked for, or the most likely one.

    An unknown pair falls back rather than erroring: it usually means the caller
    carried a config over from the repo it was looking at a moment ago.
    """
    if config or split:
        for entry in available:
            if (not config or entry["config"] == config) and (
                not split or entry["split"] == split
            ):
                return entry

    for name in PREFERRED_SPLITS:
        for entry in available:
            if entry["split"] == name:
                return entry
    return available[0]


def _get(path: str, **params: str) -> dict[str, Any]:
    try:
        response = httpx.get(f"{SERVER}{path}", params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        # The server states its reason in `error` — "dataset is gated", "config
        # not found" — and that sentence is the whole value of the failure.
        detail = _reason(e.response) or str(e)
        raise HubError(detail) from e
    except Exception as e:
        raise HubError(str(e)) from e


def _reason(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body.get("error") if isinstance(body, dict) else None


def _row(raw: Any, columns: list[str]) -> dict[str, Any]:
    """One row, trimmed to something a browser can render.

    Column order follows the declared features rather than the mapping's own, so
    a table's headers and its cells cannot drift apart.
    """
    if not isinstance(raw, dict):
        return {}
    ordered = columns or list(raw)
    return {name: _trim(raw.get(name)) for name in ordered if name in raw}


def _trim(value: Any) -> Any:
    """Strings capped, containers walked, anything exotic rendered as text.

    Bytes and images come back from the server as dicts with URLs in them; those
    survive as-is. Anything JSON cannot carry becomes its `repr`, because a
    preview that 500s on one odd column is no preview at all.
    """
    if isinstance(value, str):
        return value[:MAX_CELL_CHARS]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _trim(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trim(v) for v in value]
    return repr(value)[:MAX_CELL_CHARS]


def _any_truncation(rows: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(v, str) and len(v) >= MAX_CELL_CHARS for row in rows for v in row.values()
    )


def _suggestion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which kind and column the rows look like, for the pull form's defaults.

    Read off the rows rather than the card because the rows are the thing being
    pulled. A suggestion, never a decision — it fills the form, and the user is
    looking at the same rows it was derived from.
    """
    # Imported here rather than at module scope: `fetch` imports `search`, and a
    # top-level import back the other way would close the cycle.
    from hub.fetch import MESSAGE_COLUMNS, TEXT_COLUMNS

    for name in MESSAGE_COLUMNS:
        if any(_looks_like_turns(row.get(name)) for row in rows):
            return {"suggested_kind": DatasetKind.sft.value, "suggested_column": name}

    for name in TEXT_COLUMNS:
        if any(isinstance(row.get(name), str) and row[name].strip() for row in rows):
            return {"suggested_kind": DatasetKind.pretrain.value, "suggested_column": name}

    # Nothing named as expected: the longest string column is the best remaining
    # guess, and an empty answer would leave the form no better off.
    longest = max(
        (
            (sum(len(row[c]) for row in rows if isinstance(row.get(c), str)), c)
            for c in {k for row in rows for k in row}
        ),
        default=(0, None),
    )
    return {
        "suggested_kind": DatasetKind.pretrain.value if longest[1] else None,
        "suggested_column": longest[1] if longest[0] else None,
    }


def _looks_like_turns(value: Any) -> bool:
    """A list of role/content mappings, under any of the key pairs seen in the
    wild — the same test `fetch._turn` applies when it converts one."""
    from hub.fetch import _TURN_KEYS

    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and any(
        role in first and content in first for role, content in _TURN_KEYS
    )
