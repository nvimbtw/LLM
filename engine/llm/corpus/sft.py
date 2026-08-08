"""Building the supervised fine-tuning dataset.

SFT trains a *pretrained* model on chat conversations, computing loss only on
the assistant's replies. Its dataset is two aligned files:

    sft_corpus.bin   flat uint16 token ids, same format as corpus.bin
    sft_mask.bin     one byte per token, 1 = train / 0 = ignore, aligned 1:1

The mask comes straight from `chat.template.build_training_example`, the single
source of the format the model is prompted with — so what the model is trained
on and what it is served cannot drift apart.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.chat.template import Reasoning, Role, ThinkBudget, Turn, build_training_example
from llm.corpus import bpe, special
from llm.corpus.pack import CORPUS_DTYPE, DocSpan


@dataclass
class SftStats:
    conversations: int
    total_tokens: int
    trainable_tokens: int
    skipped: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_tokens / self.total_tokens if self.total_tokens else 0.0


def conversation_files() -> list[Path]:
    return sorted(storage.sft_dir().glob("*.jsonl"))


def read_conversations(files: list[Path]) -> list[dict]:
    """Every conversation across the given JSONL files.

    A malformed line is skipped rather than fatal: these files are generated in
    bulk, and one bad row should not cost the whole fine-tune.
    """
    conversations: list[dict] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return conversations


def _turns(raw: dict) -> list[Turn] | None:
    """Parse a conversation's turns, or None if it is not trainable.

    A conversation must end on an assistant reply — that reply is the entire
    training signal, and one ending on a user turn contributes a mask of all
    zeros while still costing context.
    """
    turns: list[Turn] = []
    for turn in raw.get("turns", []):
        role = turn.get("role", "").lower()
        if role not in ("user", "assistant"):
            return None
        turns.append(Turn(role=Role(role), text=turn.get("text", "")))

    if not turns or turns[-1].role is not Role.ASSISTANT:
        return None
    return turns


def build(
    tokenizer: Tokenizer,
    reasoning: Reasoning = Reasoning.KEEP,
    files: list[Path] | None = None,
) -> SftStats:
    """Render every conversation into the aligned corpus/mask pair."""
    files = files if files is not None else conversation_files()
    conversations = read_conversations(files)
    if not conversations:
        raise ValueError(
            f"No conversations found in {storage.sft_dir()} — SFT needs .jsonl "
            f"files of {{system?, turns: [{{role, text}}], budget?}}"
        )

    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)

    tokens: list[int] = []
    mask: list[int] = []
    documents: list[DocSpan] = []
    skipped = 0

    for raw in conversations:
        turns = _turns(raw)
        if turns is None:
            skipped += 1
            continue

        budget = ThinkBudget.parse(raw["budget"]) if raw.get("budget") else None
        ids, example_mask = build_training_example(
            raw.get("system"), turns, tokenizer, budget, reasoning
        )
        if not ids:
            skipped += 1
            continue

        documents.append(DocSpan(dataset="sft", start=len(tokens), length=len(ids)))
        tokens.extend(ids)
        mask.extend(example_mask)
        # The separator is scaffold between conversations, never a target: the
        # model must not learn to predict one conversation from another.
        tokens.append(eot_id)
        mask.append(0)

    storage.ensure_dir(storage.datasets_dir())
    np.asarray(tokens, dtype=CORPUS_DTYPE).tofile(storage.sft_corpus_path())
    np.asarray(mask, dtype=np.uint8).tofile(storage.sft_mask_path())
    storage.sft_segments_path().write_text(
        json.dumps(
            {"eot_id": eot_id, "documents": [asdict(d) for d in documents]}, indent=2
        ),
        encoding="utf-8",
    )

    return SftStats(
        conversations=len(documents),
        total_tokens=len(tokens),
        trainable_tokens=int(sum(mask)),
        skipped=skipped,
    )


def load() -> tuple[np.ndarray, np.ndarray]:
    """The corpus and its mask, checked for alignment.

    A length mismatch means the dataset was built inconsistently and every
    window's mask slice would be shifted — fail cleanly rather than train on a
    misaligned target.
    """
    corpus = np.fromfile(storage.sft_corpus_path(), dtype=CORPUS_DTYPE)
    mask = np.fromfile(storage.sft_mask_path(), dtype=np.uint8)
    if len(corpus) != len(mask):
        raise ValueError(
            f"SFT corpus/mask length mismatch: {len(corpus)} tokens vs {len(mask)} mask bytes"
        )
    return corpus, mask


def load_segments() -> tuple[int, list[DocSpan], list[dict]]:
    path = storage.sft_segments_path()
    if not path.is_file():
        return 0, [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["eot_id"], [DocSpan(**d) for d in raw["documents"]], raw.get("stages", [])


def is_built() -> bool:
    return storage.sft_corpus_path().is_file() and storage.sft_mask_path().is_file()
