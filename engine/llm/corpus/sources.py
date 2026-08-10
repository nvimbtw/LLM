"""Collecting source files off disk.

Each immediate subfolder of a data root is one selectable dataset, matching the
backend's `uploads/` convention (uploads/TinyStories, uploads/Wikipedia, …).

Two roots use this, with different file types:

    input/<dataset>/*.txt     pretrain text     -> `collect`, `catalogue`
    sft/<set>/*.jsonl         chat conversations -> `llm.corpus.sft`

The selection rules — a bare name or a `{name, count}` cap, sorted for
determinism — are the same for both and are defined once, in the `*_in`
functions. The wrappers below bind them to the pretrain root; SFT binds them to
its own.
"""

from dataclasses import dataclass
from pathlib import Path

from llm import storage

PRETRAIN_PATTERN = "*.txt"

# Loose files sitting directly in a data root rather than in a dataset folder.
#
# Selectable under this name so dropping a file in without making a folder still
# trains. It is a real dataset, not an implicit extra: an explicit selection gets
# exactly what it named, or picking one SFT set would silently drag in whatever
# else happened to be lying at the root.
ROOT_DATASET = "root"


@dataclass(frozen=True)
class SourceFile:
    """One file, tagged with the dataset it came from so the packed corpus can
    record which dataset each document span belongs to."""

    path: Path
    dataset: str
    # Which curriculum stage this file belongs to. 0 for a flat run.
    stage: int = 0


def selection(item) -> tuple[str, int | None]:
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


def _folders(base: Path) -> list[str]:
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _loose(base: Path, pattern: str) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted(base.glob(pattern))


def names_in(base: Path, pattern: str) -> list[str]:
    """Every selectable dataset under `base`, sorted, plus `root` when loose
    files are present."""
    found = _folders(base)
    if _loose(base, pattern):
        found.append(ROOT_DATASET)
    return found


def paths_in(base: Path, name: str, pattern: str) -> list[Path]:
    """The files of one dataset, sorted so a capped selection is reproducible."""
    if name == ROOT_DATASET:
        return _loose(base, pattern)
    return sorted((base / name).glob(pattern))


def catalogue_in(base: Path, pattern: str) -> list[dict]:
    """What is on disk under `base`, for a caller choosing what to train on."""
    out = []
    for name in names_in(base, pattern):
        paths = paths_in(base, name, pattern)
        out.append({
            "name": name,
            "files": len(paths),
            "bytes": sum(p.stat().st_size for p in paths),
        })
    return out


def collect_in(base: Path, names, pattern: str) -> list[SourceFile]:
    """Every file for the named datasets under `base`, all of them when `names`
    is empty.

    Entries are dataset names or `{name, count}` selections; `count` caps a
    dataset to its first N files. Without the cap one folder can dominate the
    corpus entirely — FineWeb is 99% of the tokens here — and there is no other
    way to ask for part of it.

    Capped from the front of the sorted list rather than sampled: the same
    request must select the same text every time, or a re-pack silently trains
    on a different corpus than the run it is meant to reproduce.
    """
    wanted = names or names_in(base, pattern)

    files: list[SourceFile] = []
    for item in wanted:
        name, cap = selection(item)
        paths = paths_in(base, name, pattern)
        if cap:
            paths = paths[:cap]
        for path in paths:
            files.append(SourceFile(path=path, dataset=name))
    return files


def datasets() -> list[str]:
    """Every pretrain dataset under input/, sorted."""
    return names_in(storage.input_dir(), PRETRAIN_PATTERN)


def catalogue(pattern: str = PRETRAIN_PATTERN) -> list[dict]:
    return catalogue_in(storage.input_dir(), pattern)


def collect(names=None, pattern: str = PRETRAIN_PATTERN) -> list[SourceFile]:
    return collect_in(storage.input_dir(), names, pattern)


def collect_stages(stages, pattern: str = PRETRAIN_PATTERN) -> list[SourceFile]:
    """Every source file for an ordered curriculum, tagged with its stage.

    A dataset named by two stages is collected twice on purpose — that is what
    "train on it again later" means, and the document spans keep the two apart.
    """
    files: list[SourceFile] = []
    for index, stage in enumerate(stages):
        for source in collect(stage.datasets, pattern):
            files.append(SourceFile(path=source.path, dataset=source.dataset, stage=index))
    return files
