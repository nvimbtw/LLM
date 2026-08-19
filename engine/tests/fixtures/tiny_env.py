"""Materializing the `tiny` fixture model into a throwaway DATA_DIR.

Shared by the pytest session fixture and the standalone trace recorder, so both
see byte-identical weights at byte-identical paths.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from tests.fixtures import fake_backend

FIXTURES = Path(__file__).resolve().parent
TINY_MODEL = "tiny"
TINY_DATASET = "Tiny"
TINY_CHAT = "TinyChat"

#: What a run names. The engine stores no corpora, so the fixture text lives in
#: a stand-in data engine and a selection is an id — see `fake_backend`.
TINY_DATASET_ID = fake_backend.TINY_DATASET_ID
TINY_CHAT_ID = fake_backend.TINY_CHAT_ID

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
        lines.append(json.dumps({"turns": _turns(i)}))
    return "\n".join(lines) + "\n"


def _turns(i: int) -> list[dict]:
    return [
        {"role": "user", "text": f"where did the cat sit, take {i}?"},
        {"role": "assistant", "text": "the cat sat on the mat."},
    ]


def datasets() -> list[fake_backend.Dataset]:
    """The fixture text as the data engine would hold it.

    Two documents of `SOURCE_TEXT` rather than one of twice the length: that is
    what the two `.txt` files were, so a corpus packed from this is
    byte-identical to one packed before the text moved.
    """
    return [
        fake_backend.Dataset(
            TINY_DATASET_ID,
            TINY_DATASET,
            "pretrain",
            fake_backend.text_documents([SOURCE_TEXT, SOURCE_TEXT]),
        ),
        fake_backend.Dataset(
            TINY_CHAT_ID,
            TINY_CHAT,
            "chat",
            fake_backend.chat_documents(
                [
                    [{"role": t["role"], "content": t["text"]} for t in _turns(i)]
                    for i in range(24)
                ]
            ),
        ),
    ]


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


def selected(
    dataset_id: str,
    cap: int | None = None,
    *,
    name: str | None = None,
    documents: int = 2,
    size: int = 1000,
    updated_at: str = "2026-08-19T00:00:00+00:00",
):
    """A resolved selection, for a test asserting on keys rather than on text.

    The resolution itself goes through the stand-in backend; this is for the
    cases where what is under test is what a key is a function of, and standing
    a server up to learn a document count would only obscure that.
    """
    from llm.corpus.remote import Selected

    return Selected(
        id=dataset_id,
        name=name if name is not None else dataset_id,
        kind="pretrain",
        cap=cap,
        documents=documents,
        bytes=size,
        updated_at=updated_at,
    )
