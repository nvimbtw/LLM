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

HELDOUT_TEXT = (
    "a fish swam under the log at night.\n"
    "the bright bird ate on the mat in the morning.\n"
) * 30


def install(root: Path) -> Path:
    model_dir = root / "artifacts" / "models" / TINY_MODEL
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURES / TINY_MODEL, model_dir)

    # Real data on disk, so every pipeline runs end to end rather than being
    # stubbed at its boundary.
    dataset = root / "input" / TINY_DATASET
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "a.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    (dataset / "b.txt").write_text(SOURCE_TEXT, encoding="utf-8")

    sft = root / "sft" / TINY_DATASET
    sft.mkdir(parents=True, exist_ok=True)
    (sft / "convos.jsonl").write_text(_conversations(24), encoding="utf-8")

    # Held out from training on purpose: scoring a model on its own corpus
    # measures memorization. Different sentences, same vocabulary.
    corpus = root / "eval" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "held.txt").write_text(HELDOUT_TEXT, encoding="utf-8")

    chat = root / "eval" / "chat"
    chat.mkdir(parents=True, exist_ok=True)
    (chat / "heldout.jsonl").write_text(_conversations(6), encoding="utf-8")
    return model_dir


def _conversations(count: int) -> str:
    import json

    lines = []
    for i in range(count):
        lines.append(
            json.dumps(
                {
                    "turns": [
                        {"role": "user", "text": f"where did the cat sit, take {i}?"},
                        {"role": "assistant", "text": "the cat sat on the mat."},
                    ]
                }
            )
        )
    return "\n".join(lines) + "\n"


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
