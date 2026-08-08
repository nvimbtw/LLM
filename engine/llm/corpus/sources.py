"""Collecting source text off disk.

Each immediate subfolder of `data/input/` is one selectable dataset, matching
the backend's `uploads/` convention (uploads/TinyStories, uploads/Wikipedia, …).
"""

from dataclasses import dataclass
from pathlib import Path

from llm import storage


@dataclass(frozen=True)
class SourceFile:
    """One file, tagged with the dataset it came from so the packed corpus can
    record which dataset each document span belongs to."""

    path: Path
    dataset: str
    # Which curriculum stage this file belongs to. 0 for a flat run.
    stage: int = 0


def datasets() -> list[str]:
    """Every dataset folder under input/, sorted."""
    base = storage.input_dir()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _selection(item) -> tuple[str, int | None]:
    """`(name, cap)` from a bare dataset name, a `{name, count}` mapping, or a
    `DatasetSelection`.

    Duck-typed rather than importing the config model, so `sources` stays usable
    from a script that has a list of folder names and nothing else.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        return item["name"], item.get("count") or None
    return item.name, getattr(item, "count", None) or None


def collect_stages(stages, pattern: str = "*.txt") -> list[SourceFile]:
    """Every source file for an ordered curriculum, tagged with its stage.

    A dataset named by two stages is collected twice on purpose — that is what
    "train on it again later" means, and the document spans keep the two apart.
    """
    files: list[SourceFile] = []
    for index, stage in enumerate(stages):
        for source in collect(stage.datasets, pattern):
            files.append(SourceFile(path=source.path, dataset=source.dataset, stage=index))
    return files


def catalogue(pattern: str = "*.txt") -> list[dict]:
    """What is on disk, for a caller choosing what to train on."""
    base = storage.input_dir()
    out = []
    for name in datasets():
        paths = sorted((base / name).glob(pattern))
        out.append({
            "name": name,
            "files": len(paths),
            "bytes": sum(p.stat().st_size for p in paths),
        })
    return out


def collect(names=None, pattern: str = "*.txt") -> list[SourceFile]:
    """Every source file for the named datasets (all of them when `names` is
    empty), sorted for a deterministic corpus.

    Entries are dataset names or `{name, count}` selections; `count` caps a
    dataset to its first N files. Without the cap one folder can dominate the
    corpus entirely — FineWeb is 99% of the tokens here — and there is no other
    way to ask for part of it.

    Loose files sitting directly in input/ are picked up as a dataset named
    "root", so dropping a .txt in without making a folder still trains.
    """
    base = storage.input_dir()
    wanted = names or datasets()

    files: list[SourceFile] = []
    for item in wanted:
        name, cap = _selection(item)
        paths = sorted((base / name).glob(pattern))
        # Capped from the front of the sorted list rather than sampled: the same
        # request must select the same text every time, or a re-pack silently
        # trains on a different corpus than the run it is meant to reproduce.
        if cap:
            paths = paths[:cap]
        for path in paths:
            files.append(SourceFile(path=path, dataset=name))

    for path in sorted(base.glob(pattern)):
        files.append(SourceFile(path=path, dataset="root"))

    return files
