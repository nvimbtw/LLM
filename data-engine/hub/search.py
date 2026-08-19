"""Finding datasets on the Hugging Face Hub.

Read-only and unauthenticated by default, which is enough for public datasets.
A gated repo fails at pull time with the Hub's own message rather than being
hidden here — "you must accept the terms" is useful; a dataset silently missing
from a search is not.

The repo-root `scripts/prepare_*.py` do the same listing with `sys.exit` on
failure. That is right for a CLI and wrong for a service, so the mechanics are
shared and the error handling is not.
"""

from typing import Any

from huggingface_hub import HfApi

#: A search is a browse, not an export. Enough rows to choose from, few enough
#: that the Hub is not asked for a catalogue on every keystroke.
MAX_RESULTS = 50

PARQUET_SUFFIX = ".parquet"


class HubError(RuntimeError):
    """The Hub could not answer. Carries its message, which is usually the
    actionable half — gated terms, a rename, a rate limit."""


def _api() -> HfApi:
    return HfApi()


def search(query: str | None, limit: int = 20) -> list[dict[str, Any]]:
    """Datasets matching `query`, most-downloaded first.

    An empty query lists the most-downloaded overall, which is what an empty
    search box should show rather than nothing.
    """
    capped = max(1, min(limit, MAX_RESULTS))
    try:
        # `sort="downloads"` already descends, and `direction` was removed in
        # huggingface_hub 1.x — passing it is a TypeError, not a no-op.
        found = _api().list_datasets(search=query or None, sort="downloads", limit=capped)
        return [_summary(d) for d in found]
    except Exception as e:
        raise HubError(str(e)) from e


def _summary(dataset: Any) -> dict[str, Any]:
    return {
        "repo": dataset.id,
        "author": getattr(dataset, "author", None),
        "downloads": getattr(dataset, "downloads", None),
        "likes": getattr(dataset, "likes", None),
        "tags": list(getattr(dataset, "tags", None) or [])[:12],
        "updated_at": _isoformat(getattr(dataset, "last_modified", None)),
    }


def _isoformat(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def detail(repo: str) -> dict[str, Any]:
    """What one dataset offers: its declared configs and columns, and its shards.

    Both halves are read from metadata the Hub already has. Opening a shard to
    learn its schema would mean downloading hundreds of megabytes to answer a
    question about a page the user is only looking at — so a dataset whose card
    declares nothing reports no columns, and the pull form asks for the column
    name instead of guessing it.
    """
    try:
        info = _api().dataset_info(repo)
        files = _api().list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        raise HubError(str(e)) from e

    shards = sorted(f for f in files if f.endswith(PARQUET_SUFFIX))
    return {
        "repo": repo,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": list(getattr(info, "tags", None) or [])[:12],
        "updated_at": _isoformat(getattr(info, "last_modified", None)),
        "gated": bool(getattr(info, "gated", False)),
        "configs": _configs(getattr(info, "card_data", None)),
        "shards": shards[:200],
        "shard_count": len(shards),
        "prefixes": _prefixes(shards),
    }


def _configs(card: Any) -> list[dict[str, Any]]:
    """Configs, splits and column names as the dataset card declares them.

    `card_data.dataset_info` is either one mapping or a list of them depending on
    whether the dataset has multiple configs — normalised to a list here so the
    caller has one shape to render.
    """
    declared = getattr(card, "dataset_info", None) if card is not None else None
    if declared is None:
        return []
    entries = declared if isinstance(declared, list) else [declared]

    configs = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        configs.append(
            {
                "name": entry.get("config_name") or "default",
                "columns": [
                    f.get("name") for f in (entry.get("features") or []) if isinstance(f, dict)
                ],
                "splits": [
                    s.get("name") for s in (entry.get("splits") or []) if isinstance(s, dict)
                ],
            }
        )
    return configs


def _prefixes(shards: list[str]) -> list[dict[str, Any]]:
    """The directories shards sit in, with how many are in each.

    This is what a pull selects by: a dataset ships `train/` and `test/`, or one
    directory per config, and picking the whole repo would mix them.
    """
    counts: dict[str, int] = {}
    for shard in shards:
        prefix = shard.rsplit("/", 1)[0] if "/" in shard else ""
        counts[prefix] = counts.get(prefix, 0) + 1
    return [{"prefix": p, "shards": n} for p, n in sorted(counts.items())]
