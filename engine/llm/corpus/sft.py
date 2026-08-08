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
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.chat.template import Reasoning, Role, ThinkBudget, Turn, build_training_example
from llm.corpus import bpe, special
from llm.corpus.pack import CORPUS_DTYPE, FLUSH_TOKENS, DocSpan


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


def read_conversations(files: list[Path]) -> Iterator[dict]:
    """Every conversation across the given JSONL files, one at a time.

    A malformed line is skipped rather than fatal: these files are generated in
    bulk, and one bad row should not cost the whole fine-tune.

    Yields rather than returning a list, and reads line by line rather than
    `read_text().splitlines()`, because the SFT set runs to hundreds of megabytes
    of JSONL: materializing it as parsed dicts costs several times its size on
    disk, on top of the corpus being built from it.
    """
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


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

    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)

    documents: list[DocSpan] = []
    skipped = 0
    total_tokens = 0
    trainable_tokens = 0

    storage.ensure_dir(storage.datasets_dir())

    # Both files are streamed for the same reason `pack.pack` is — a Python list
    # boxes every token, so buffering the whole corpus costs many times its packed
    # size — and both go through a `.partial` so a failed or empty build cannot
    # leave a half-written pair behind. They must stay aligned 1:1; `load()`
    # rejects a length mismatch, and writing them in lockstep here is what makes
    # that check pass.
    corpus_final, mask_final = storage.sft_corpus_path(), storage.sft_mask_path()
    corpus_partial = corpus_final.with_suffix(".bin.partial")
    mask_partial = mask_final.with_suffix(".bin.partial")

    token_buffer: list[int] = []
    mask_buffer: list[int] = []

    def flush() -> None:
        if token_buffer:
            corpus_out.write(np.asarray(token_buffer, dtype=CORPUS_DTYPE).tobytes())
            mask_out.write(np.asarray(mask_buffer, dtype=np.uint8).tobytes())
            token_buffer.clear()
            mask_buffer.clear()

    with corpus_partial.open("wb") as corpus_out, mask_partial.open("wb") as mask_out:
        for raw in read_conversations(files):
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

            documents.append(DocSpan(dataset="sft", start=total_tokens, length=len(ids)))
            token_buffer.extend(ids)
            mask_buffer.extend(example_mask)
            # The separator is scaffold between conversations, never a target: the
            # model must not learn to predict one conversation from another.
            token_buffer.append(eot_id)
            mask_buffer.append(0)

            total_tokens += len(ids) + 1
            trainable_tokens += sum(example_mask)

            if len(token_buffer) >= FLUSH_TOKENS:
                flush()

        flush()

    if not documents:
        corpus_partial.unlink(missing_ok=True)
        mask_partial.unlink(missing_ok=True)
        raise ValueError(
            f"No conversations found in {storage.sft_dir()} — SFT needs .jsonl "
            f"files of {{system?, turns: [{{role, text}}], budget?}}"
        )

    corpus_partial.replace(corpus_final)
    mask_partial.replace(mask_final)

    storage.sft_segments_path().write_text(
        json.dumps(
            {"eot_id": eot_id, "documents": [asdict(d) for d in documents]}, indent=2
        ),
        encoding="utf-8",
    )

    return SftStats(
        conversations=len(documents),
        total_tokens=total_tokens,
        trainable_tokens=trainable_tokens,
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
