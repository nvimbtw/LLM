"""Building the supervised fine-tuning dataset.

SFT trains a *pretrained* model on chat conversations, computing loss only on
the assistant's replies. Its dataset is two aligned files, plus a sidecar:

    datasets/sft/<key>/corpus.bin    flat uint16 token ids, same format as corpus.bin
    datasets/sft/<key>/mask.bin      one byte per token, 1 = train / 0 = ignore
    datasets/sft/<key>/segments.json document spans + what the build was made from

The mask comes straight from `chat.template`, the single source of the format the
model is prompted with — so what the model is trained on and what it is served
cannot drift apart. The build renders a batch of conversations at a time
(`encode_examples`) where a chat turn renders one (`build_training_example`);
both go through the same segmenting and the same assembly, which is what keeps
that guarantee true of the fast path as well.

`<key>` is derived from the request: which sets were selected and whether
reasoning was kept. Both change every token and every mask byte, so a build is
not interchangeable with another and they cannot share a path — see
`dataset_key`.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.chat import template
from llm.chat.template import Reasoning, Role, ThinkBudget, Turn
from llm.corpus import bpe, sources, special
from llm.corpus.pack import CORPUS_DTYPE, ENCODE_BATCH_CHARS, FLUSH_TOKENS, DocSpan

SFT_PATTERN = "*.jsonl"

# How much of the selection survives into the directory name before the hash
# takes over. Purely for a human reading `ls datasets/sft/`; the digest is what
# makes the key unique.
KEY_SLUG_SETS = 2
KEY_DIGEST_CHARS = 10


@dataclass
class SftStats:
    conversations: int
    total_tokens: int
    trainable_tokens: int
    skipped: int
    # Which reserved marker ids this dataset actually contained, sorted. Mirrors
    # `PackStats.special_ids_seen`, and collected the same way: accumulated per
    # conversation during the build, off the few hundred ids that conversation
    # produced. The whole-corpus alternative is what OOM-killed the run — see
    # `special_ids_in`.
    special_ids_seen: list[int] = field(default_factory=list)

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_tokens / self.total_tokens if self.total_tokens else 0.0


def sets() -> list[str]:
    """Every selectable SFT set on disk, sorted."""
    return sources.names_in(storage.sft_dir(), SFT_PATTERN)


def catalogue() -> list[dict]:
    """What SFT data is on disk, for a caller composing a fine-tune."""
    return sources.catalogue_in(storage.sft_dir(), SFT_PATTERN)


def conversation_files(names=None) -> list[Path]:
    """The JSONL files for the named sets, or every set when `names` is empty."""
    return [f.path for f in sources.collect_in(storage.sft_dir(), names, SFT_PATTERN)]


def resolve(names=None) -> list[tuple[str, int | None]]:
    """The concrete `(name, cap)` selection a request resolves to right now.

    An empty request means "every set on disk", which is a *different* corpus
    once another set is fetched. Expanding it here, before the key is derived,
    is what stops a run reusing a build that predates the data it asked for.
    """
    if names:
        return [sources.selection(item) for item in names]
    return [(name, None) for name in sets()]


def dataset_key(selection: list[tuple[str, int | None]], reasoning: Reasoning) -> str:
    """A stable directory name for the dataset this request builds.

    Order is preserved rather than sorted: the sets are concatenated in the
    order given, and with `shuffle` seeded, a different concatenation order is a
    different training run. Two orderings therefore key apart — an extra build
    is the cheap mistake here, reusing the wrong corpus is the expensive one.
    """
    payload = json.dumps(
        {"sets": [[name, cap] for name, cap in selection], "reasoning": reasoning.value}
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:KEY_DIGEST_CHARS]
    slug = "-".join(
        re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        for name, _ in selection[:KEY_SLUG_SETS]
    )
    return f"{slug or 'empty'}-{reasoning.value}-{digest}"


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
    training signal, and one ending on a user turn or a tool result contributes
    a mask of all zeros while still costing context.
    """
    turns: list[Turn] = []
    for turn in raw.get("turns", []):
        role = turn.get("role", "").lower()
        if role not in ("user", "assistant", "tool"):
            return None
        turns.append(Turn(role=Role(role), text=turn.get("text", "")))

    if not turns or turns[-1].role is not Role.ASSISTANT:
        return None
    return turns


def clear_legacy() -> int:
    """Delete the pre-keyed singleton build.

    Its provenance was never recorded, so there is no key it can honestly be
    adopted into. Rebuilding is cheap; training on an unidentifiable corpus is
    the bug this replaced.
    """
    removed = 0
    for path in storage.legacy_sft_paths():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def build(
    tokenizer: Tokenizer,
    selection: list[tuple[str, int | None]],
    reasoning: Reasoning = Reasoning.KEEP,
) -> SftStats:
    """Render every conversation in `selection` into the aligned corpus/mask pair."""
    key = dataset_key(selection, reasoning)
    files = conversation_files([{"name": name, "count": cap} for name, cap in selection])

    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)

    # Intersected against each conversation as it is rendered, so the answer to
    # "which markers does this dataset use?" is known by the time the build ends
    # and the corpus never has to be scanned for it.
    marker_ids = {
        tokenizer.token_to_id(name)
        for name in special.token_list()
        if tokenizer.token_to_id(name) is not None
    }
    seen: set[int] = set()

    documents: list[DocSpan] = []
    skipped = 0
    total_tokens = 0
    trainable_tokens = 0

    storage.ensure_dir(storage.sft_dataset_dir(key))

    # Both files are streamed for the same reason `pack.pack` is — a Python list
    # boxes every token, so buffering the whole corpus costs many times its packed
    # size — and both go through a `.partial` so a failed or empty build cannot
    # leave a half-written pair behind. They must stay aligned 1:1; `load()`
    # rejects a length mismatch, and writing them in lockstep here is what makes
    # that check pass.
    corpus_final, mask_final = storage.sft_corpus_path(key), storage.sft_mask_path(key)
    corpus_partial = corpus_final.with_suffix(".bin.partial")
    mask_partial = mask_final.with_suffix(".bin.partial")

    token_buffer: list[int] = []
    mask_buffer: list[int] = []
    # The token histogram, accumulated on the way to disk exactly as `pack.pack`
    # does it, and for the same reason: computing it later means reading the whole
    # corpus back before the first training step.
    vocab = bpe.vocab_size(tokenizer)
    counts = np.zeros(vocab, dtype=np.int64)

    def flush() -> None:
        if token_buffer:
            block = np.asarray(token_buffer, dtype=CORPUS_DTYPE)
            counts[:] += np.bincount(block, minlength=vocab)[:vocab]
            corpus_out.write(block.tobytes())
            mask_out.write(np.asarray(mask_buffer, dtype=np.uint8).tobytes())
            token_buffer.clear()
            mask_buffer.clear()

    # Conversations are cut into segments first and encoded a batch at a time, so
    # the tokenizer gets every core instead of one — see `template.encode_examples`.
    # The batch is bounded by the characters it holds rather than by how many
    # conversations it holds: a chat set is millions of short conversations, and a
    # count would be a different amount of work in every set.
    pending: list[list[tuple[str, bool]]] = []
    pending_chars = 0

    def drain() -> None:
        """Encode the conversations held so far, in order, and append them."""
        nonlocal pending_chars, skipped, total_tokens, trainable_tokens
        if not pending:
            return

        for ids, example_mask in template.encode_examples(pending, tokenizer):
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
            seen.update(marker_ids.intersection(ids))

            if len(token_buffer) >= FLUSH_TOKENS:
                flush()

        pending.clear()
        pending_chars = 0

    with corpus_partial.open("wb") as corpus_out, mask_partial.open("wb") as mask_out:
        for raw in read_conversations(files):
            turns = _turns(raw)
            if turns is None:
                skipped += 1
                continue

            budget = ThinkBudget.parse(raw["budget"]) if raw.get("budget") else None
            segments = template.training_segments(
                raw.get("system"), turns, budget, reasoning
            )
            pending.append(segments)
            pending_chars += sum(len(text) for text, _ in segments)
            if pending_chars >= ENCODE_BATCH_CHARS:
                drain()

        drain()
        flush()

    if not documents:
        corpus_partial.unlink(missing_ok=True)
        mask_partial.unlink(missing_ok=True)
        named = ", ".join(name for name, _ in selection) or "(none on disk)"
        raise ValueError(
            f"No conversations found in {storage.sft_dir()} for sets [{named}] — SFT "
            f"needs .jsonl files of {{system?, turns: [{{role, text}}], budget?}} "
            f"with role one of user, assistant, tool"
        )

    corpus_partial.replace(corpus_final)
    mask_partial.replace(mask_final)

    # The selection is written beside the documents so a built dataset says what
    # it is. `is_built` compares against this rather than trusting the directory
    # name, which is the check that makes reuse safe.
    storage.sft_segments_path(key).write_text(
        json.dumps(
            {
                "eot_id": eot_id,
                "sets": [[name, cap] for name, cap in selection],
                "reasoning": reasoning.value,
                "special_ids_seen": sorted(seen),
                "token_counts": counts.tolist(),
                "documents": [asdict(d) for d in documents],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SftStats(
        conversations=len(documents),
        total_tokens=total_tokens,
        trainable_tokens=trainable_tokens,
        skipped=skipped,
        special_ids_seen=sorted(seen),
    )


def load(key: str) -> tuple[np.ndarray, np.ndarray]:
    """The corpus and its mask, checked for alignment.

    A length mismatch means the dataset was built inconsistently and every
    window's mask slice would be shifted — fail cleanly rather than train on a
    misaligned target.

    Mapped, not read, for the reason `pack.load_corpus` is: `fromfile` makes both
    files resident and private — 1.7 GiB on the full conversation set, on a
    machine with 7 GiB — while training only ever reads small contiguous spans of
    them (`EpochStream.gather`). A mapping is backed by the file, so the kernel
    reclaims it under pressure instead of the process being killed.
    """
    corpus = np.memmap(storage.sft_corpus_path(key), dtype=CORPUS_DTYPE, mode="r")
    mask = np.memmap(storage.sft_mask_path(key), dtype=np.uint8, mode="r")
    if len(corpus) != len(mask):
        raise ValueError(
            f"SFT corpus/mask length mismatch: {len(corpus)} tokens vs {len(mask)} mask bytes"
        )
    return corpus, mask


def load_segments(key: str) -> tuple[int, list[DocSpan], dict]:
    """Document boundaries plus what the build was made from."""
    path = storage.sft_segments_path(key)
    if not path.is_file():
        return 0, [], {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    info = {
        "sets": raw.get("sets", []),
        "reasoning": raw.get("reasoning"),
        # Absent on a build made before it was recorded; None means "unknown", so
        # the caller scans rather than concluding the dataset uses no markers.
        "special_ids_seen": raw.get("special_ids_seen"),
        # Per-token frequencies for the diagnostics, as `pack.load_segments`
        # returns them. None on a build that predates them.
        "token_counts": raw.get("token_counts"),
    }
    return raw["eot_id"], [DocSpan(**d) for d in raw["documents"]], info


# How many tokens to test at a time when a dataset has to be scanned. 8M tokens
# is a 16 MiB window of the mapping and a bounded `np.unique` over it, so peak
# memory is flat no matter how long the corpus is.
SCAN_CHUNK = 1 << 23


def special_ids_in(corpus: np.ndarray, marker_ids: set[int], separator_id: int) -> list[int]:
    """Which of `marker_ids` occur in `corpus`, scanned in bounded chunks.

    The fallback for a build whose sidecar predates `special_ids_seen`; `build`
    records the answer directly and nothing needs to come here.

    Chunked and vectorized rather than `marker_ids.intersection(corpus.tolist())`,
    which is how this was written and what OOM-killed the engine: `tolist` boxes
    every token into a Python int at ~36 bytes against the 2 the mapping holds,
    so a 587M-token dataset asked the kernel for ~21 GiB to answer a question
    about thirty ids.

    `separator_id` is excluded because this has to answer the *same* question
    `build` and `pack.pack` answer, and both count only a document's own tokens —
    the separator is written between documents, outside the ids they record. A
    pretrain corpus of plain text therefore never lists it as seen, so counting it
    here would mark the stop token as one the base never trained and warm-start
    would reset the embedding row the base leans on hardest to stop generating.
    """
    found: set[int] = set()
    wanted = marker_ids - {separator_id}
    for start in range(0, len(corpus), SCAN_CHUNK):
        chunk = np.unique(np.asarray(corpus[start : start + SCAN_CHUNK]))
        found.update(wanted.intersection(chunk.tolist()))
        if found == wanted:
            break
    return sorted(found)


def is_built(key: str, selection: list[tuple[str, int | None]], reasoning: Reasoning) -> bool:
    """Whether a dataset on disk was built from *this* request.

    Existence is not enough. The old check asked only whether two files were
    present, so a run with different options skipped the build and fine-tuned on
    whatever was there — the reason the sidecar records its selection at all.
    """
    if not (storage.sft_corpus_path(key).is_file() and storage.sft_mask_path(key).is_file()):
        return False
    _, _, info = load_segments(key)
    if not info:
        return False
    recorded = [(name, cap) for name, cap in info.get("sets", [])]
    return recorded == list(selection) and info.get("reasoning") == reasoning.value
