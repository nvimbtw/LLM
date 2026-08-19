"""Built vocabularies: one BPE tokenizer per selection of corpora, kept.

A tokenizer used to be a singleton that every pretrain overwrote. It is now a
named artifact you build once and point runs at, which is the same arrangement
`sft.py` already gives a built dataset, in the same four parts: resolve the
request, key it, write what it was built from beside it, and compare against
that record rather than against the mere existence of a file.

A run that names no vocabulary still gets one — `auto_id` derives a *stable* id
from the build's own inputs, so two identical runs share a vocabulary (and
therefore a packed corpus) exactly as they did when there was only one, while a
changed selection keys apart instead of silently reusing.
"""

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

from tokenizers import Tokenizer

from llm import storage
from llm.corpus import bpe, remote, special
from llm.train.progress import ProgressCallback, Tokenize

#: Namespace for `auto_id`. Fixed forever: changing it re-keys every vocabulary
#: an unnamed run would otherwise have reused.
AUTO_NAMESPACE = uuid.UUID("6f9b1f2e-3c4d-5a6b-8c7d-9e0f1a2b3c4d")

#: How much of the vocabulary's digest identifies it. A corpus is keyed on this,
#: so it has to survive being read by a person in a directory listing.
FINGERPRINT_CHARS = 16

#: A request resolved against the data engine — what everything here keys,
#: records and reads. See `remote.Selected`.
Selection = list[remote.Selected]


@dataclass
class VocabStats:
    """What a build produced. `fingerprint` is what a packed corpus is keyed on."""

    datasets: Selection
    vocab_size: int
    merges: int
    fingerprint: str
    # How much of that selection the merges were actually fitted to.
    sampled_chunks: int = 0
    sampled_bytes: int = 0


def resolve(client, selections) -> Selection:
    """What a request resolves to right now, against the data engine.

    Resolved before anything is keyed, which is what stops a run reusing a build
    that predates the data it asked for: a dataset pulled again since the last
    build holds more documents under the same id.
    """
    return remote.resolve(client, selections)


def auto_id(selection: Selection, max_vocab: int) -> str:
    """The id an unnamed build gets: derived from its inputs, not minted.

    Two runs asking for the same corpora at the same vocabulary size want the
    same vocabulary — that is what made skipping tokenization worth having. A
    random id per run would rebuild the tokenizer *and* re-pack the corpus every
    time, since the corpus is keyed on the vocabulary.
    """
    return str(uuid.uuid5(AUTO_NAMESPACE, _payload(selection, max_vocab)))


def build(
    client,
    tokenizer_id: str,
    selection: Selection,
    max_vocab: int,
    on_progress: ProgressCallback | None = None,
    sample_bytes: int = bpe.SAMPLE_BYTES,
) -> VocabStats:
    """Learn a vocabulary from the selected corpora and record what it came from.

    The merges are learned from a bounded sample of the selection, not from all
    of it: `bpe.shares` splits the budget across the datasets and `remote.sample`
    draws each share strided across its dataset. The selection still decides
    *what* is sampled, so the vocabulary is still a function of the datasets
    asked for.
    """
    prose = [selected for selected in selection if selected.taken > 0]
    if not prose:
        raise ValueError(
            f"Nothing to learn merges from: {_named(selection)} "
            f"{'hold no documents' if selection else 'is an empty selection'}"
        )

    if on_progress is not None:
        on_progress(Tokenize(documents=sum(s.taken for s in prose), progress=0.05))

    directory = storage.ensure_dir(storage.tokenizer_build_dir(tokenizer_id))

    # Each dataset's slice of the budget, then that slice drawn strided across
    # the dataset — the two halves of ADR 0004, unchanged by the move. What used
    # to be file sizes on disk is the dataset's own recorded `byte_size`, scaled
    # to the capped part of it.
    budgets = bpe.shares({s.id: _selected_bytes(s) for s in prose}, sample_bytes)
    by_id = {s.id: s for s in prose}
    drawn = (
        chunk
        for dataset_id, share in budgets
        for chunk in remote.sample(
            client, by_id[dataset_id], share, take_each=bpe.SAMPLE_TAKE_BYTES
        )
    )

    # The sample is written under the build's own directory rather than held in
    # memory: it is bounded at `sample_bytes`, the trainer counts across cores
    # over the files it is given, and a tokenizer that looks wrong is otherwise
    # unanswerable after the fact. It is the one place text still touches this
    # disk, and it is a build artifact rather than a corpus.
    sample_dir = storage.tokenizer_sample_dir(tokenizer_id)
    shutil.rmtree(sample_dir, ignore_errors=True)
    written = bpe.write_chunks(drawn, sample_dir)
    if not written:
        raise ValueError(
            f"Nothing to learn merges from: {_named(selection)} returned no prose. "
            f"A chat dataset holds conversations, and a vocabulary is learned from "
            f"text — fine-tune on those instead"
        )
    tokenizer = bpe.train(written, max_vocab, out_dir=directory)

    stats = VocabStats(
        datasets=list(selection),
        vocab_size=bpe.vocab_size(tokenizer),
        merges=bpe.merge_count(tokenizer),
        fingerprint=file_fingerprint(storage.tokenizer_file(directory)),
        sampled_chunks=len(written),
        sampled_bytes=sum(path.stat().st_size for path in written),
    )

    # Written after the vocabulary lands, so a directory with no sidecar is an
    # interrupted build: `info` returns None and nothing will reuse it.
    storage.tokenizer_meta_path(tokenizer_id).write_text(
        json.dumps(
            {
                "tokenizer_id": tokenizer_id,
                "max_vocab": max_vocab,
                "vocab_size": stats.vocab_size,
                "merges": stats.merges,
                # What the merges were fitted to, as opposed to what was
                # selected. `matches` compares the budget, so raising it rebuilds
                # rather than silently reusing merges learned from less text.
                "sample_bytes": sample_bytes,
                "sampled_chunks": stats.sampled_chunks,
                "sampled_bytes": stats.sampled_bytes,
                "datasets": [selected.record() for selected in selection],
                # The reserved block this was trained against. A later change to
                # `special.token_list()` is then visible on an old vocabulary,
                # rather than being discovered as garbage marker ids.
                "special_tokens_sha": special_tokens_sha(),
                "fingerprint": stats.fingerprint,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if on_progress is not None:
        on_progress(
            Tokenize(
                vocab=stats.vocab_size,
                merges=stats.merges,
                progress=1.0,
                done=True,
            )
        )
    return stats


def _selected_bytes(selected) -> int:
    """How much text a selection actually offers, capped part included.

    The dataset records its whole size; a cap takes a prefix of its documents,
    so the share it should get is scaled by the same fraction. Without this a
    capped dataset would claim its full weight in the split and take a share it
    cannot fill, shrinking the sample by the difference — the failure `shares`
    already water-fills against, arriving one level earlier.
    """
    if not selected.cap or selected.cap >= selected.documents or not selected.documents:
        return selected.bytes
    return int(selected.bytes * selected.taken / selected.documents)


def _named(selection) -> str:
    return ", ".join(str(selected) for selected in selection) or "(nothing)"


def load(tokenizer_id: str) -> Tokenizer:
    if info(tokenizer_id) is None:
        raise FileNotFoundError(f"No built vocabulary {tokenizer_id}; build one first.")
    return bpe.load(storage.tokenizer_build_dir(tokenizer_id))


def for_corpus(key: str) -> Tokenizer:
    """The vocabulary a packed corpus was encoded with, named by its sidecar.

    The only honest way to read a corpus: its ids mean nothing without the
    vocabulary that produced them. For the ad-hoc tools under `explore/`, which
    used to reach for the shared singleton and get whichever vocabulary the last
    pretrain happened to leave behind.
    """
    from llm.corpus import pack

    _, _, recorded = pack.load_segments(key)
    tokenizer_id = recorded.get("tokenizer_id")
    if not tokenizer_id:
        raise FileNotFoundError(f"The corpus at {key} does not name the vocabulary it used")
    return load(tokenizer_id)


def info(tokenizer_id: str) -> dict | None:
    """What a build was made from, or None when there is no complete build."""
    path = storage.tokenizer_meta_path(tokenizer_id)
    if not path.is_file() or not storage.tokenizer_file(path.parent).is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def fingerprint(tokenizer_id: str) -> str:
    """The vocabulary's digest, read back rather than recomputed."""
    recorded = info(tokenizer_id)
    if recorded is None:
        raise FileNotFoundError(f"No built vocabulary {tokenizer_id}")
    return recorded["fingerprint"]


def matches(
    tokenizer_id: str,
    selection: Selection,
    max_vocab: int,
    sample_bytes: int = bpe.SAMPLE_BYTES,
) -> bool:
    """Whether the build on disk is the one this request asks for.

    Compared against the sidecar rather than the directory name, for the reason
    `sft.is_built` gives: existence is not evidence. An `auto_id` collision, a
    hand-edited directory or a change to the reserved block all fail closed into
    a rebuild.
    """
    recorded = info(tokenizer_id)
    if recorded is None:
        return False
    on_disk = recorded.get("datasets", [])
    return (
        on_disk == [selected.record() for selected in selection]
        and recorded.get("max_vocab") == max_vocab
        and recorded.get("special_tokens_sha") == special_tokens_sha()
        # A build made before the budget existed learned from everything
        # selected, which is not what this run is asking for.
        and recorded.get("sample_bytes") == sample_bytes
    )


def catalogue() -> list[dict]:
    """Every complete build on disk, newest first, with what it occupies."""
    directory = storage.tokenizers_dir()
    if not directory.is_dir():
        return []
    built = []
    for child in directory.iterdir():
        recorded = info(child.name) if child.is_dir() else None
        if recorded is not None:
            built.append({**recorded, "bytes": _bytes_in(child)})
    return sorted(built, key=lambda b: b.get("built_at") or "", reverse=True)


def delete(tokenizer_id: str) -> bool:
    """Remove a build. The models trained on it keep their own copy."""
    directory = storage.tokenizer_build_dir(tokenizer_id)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True


def file_fingerprint(path) -> str:
    """A digest of the saved vocabulary.

    A corpus is keyed on this and never on the id: rebuilding under one id
    renumbers every token, and an id-keyed corpus would then decode to different
    text while looking perfectly valid.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:FINGERPRINT_CHARS]


def special_tokens_sha() -> str:
    return hashlib.sha1("\n".join(special.token_list()).encode("utf-8")).hexdigest()[:10]


def _payload(selection: Selection, max_vocab: int) -> str:
    return json.dumps(
        {
            "datasets": [selected.identity() for selected in selection],
            "max_vocab": max_vocab,
            "specials": special_tokens_sha(),
        }
    )


def _bytes_in(directory) -> int:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
