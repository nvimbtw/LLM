"""Materializing the `tiny` fixture model into a throwaway DATA_DIR.

Shared by the pytest session fixture and the standalone trace recorder, so both
see byte-identical weights at byte-identical paths.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
TINY_MODEL = "tiny"
TINY_DATASET = "Tiny"

# Repetitive enough for BPE to find merges in a corpus this small.
SOURCE_TEXT = (
    "the cat sat on the mat. the cat ate the fish.\n"
    "a dog sat on a log. the dog saw the cat.\n"
    "birds sing in the morning and the morning is bright.\n"
) * 40


def install(root: Path) -> Path:
    model_dir = root / "artifacts" / "models" / TINY_MODEL
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURES / TINY_MODEL, model_dir)

    # A real dataset on disk, so a training run can be exercised end to end
    # rather than stubbed at the pipeline boundary.
    dataset = root / "input" / TINY_DATASET
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "a.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    (dataset / "b.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    return model_dir


@contextmanager
def data_root(root: Path | None = None):
    """Set DATA_DIR to a tree holding only `tiny`, and restore it afterwards."""
    created = None
    if root is None:
        created = tempfile.TemporaryDirectory()
        root = Path(created.name)
    install(root)

    previous = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(root)
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = previous
        if created is not None:
            created.cleanup()
