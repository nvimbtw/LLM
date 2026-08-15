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
from llm.corpus import bpe, sources, special
from llm.train.progress import ProgressCallback, Tokenize

#: Namespace for `auto_id`. Fixed forever: changing it re-keys every vocabulary
#: an unnamed run would otherwise have reused.
AUTO_NAMESPACE = uuid.UUID("6f9b1f2e-3c4d-5a6b-8c7d-9e0f1a2b3c4d")

#: How much of the vocabulary's digest identifies it. A corpus is keyed on this,
#: so it has to survive being read by a person in a directory listing.
FINGERPRINT_CHARS = 16

Selection = list[tuple[str, int | None]]


@dataclass
class VocabStats:
    """What a build produced. `fingerprint` is what a packed corpus is keyed on."""

    files: int
    vocab_size: int
    merges: int
    datasets: Selection
    fingerprint: str
    # How much of that selection the merges were actually fitted to.
    sampled_files: int = 0
    sampled_bytes: int = 0


def resolve(names=None) -> Selection:
    """The `(name, cap)` pairs a request resolves to right now.

    An empty request means "every corpus under input/", which is a *different*
    vocabulary once another corpus is fetched. Expanding it here, before anything
    is keyed, is what stops a run reusing a build that predates the data it asked
    for — the same step `sft.resolve` exists for.
    """
    if names:
        return [sources.selection(item) for item in names]
    return [(name, None) for name in sources.datasets()]


def auto_id(selection: Selection, max_vocab: int) -> str:
    """The id an unnamed build gets: derived from its inputs, not minted.

    Two runs asking for the same corpora at the same vocabulary size want the
    same vocabulary — that is what made skipping tokenization worth having. A
    random id per run would rebuild the tokenizer *and* re-pack the corpus every
    time, since the corpus is keyed on the vocabulary.
    """
    return str(uuid.uuid5(AUTO_NAMESPACE, _payload(selection, max_vocab)))


def build(
    tokenizer_id: str,
    selection: Selection,
    max_vocab: int,
    on_progress: ProgressCallback | None = None,
    sample_bytes: int = bpe.SAMPLE_BYTES,
) -> VocabStats:
    """Learn a vocabulary from the selected corpora and record what it came from.

    The merges are learned from a bounded sample of the selection, not from all
    of it — see `bpe.plan_sample`. The selection still decides *what* is sampled,
    so the vocabulary is still a function of the corpora asked for.
    """
    files = sources.collect([{"name": name, "count": cap} for name, cap in selection])
    if not files:
        raise FileNotFoundError(
            f"No .txt files under {storage.input_dir()} for {[n for n, _ in selection]}"
            + _sft_hint(selection)
        )
    if on_progress is not None:
        on_progress(Tokenize(files=len(files), progress=0.05))

    directory = storage.ensure_dir(storage.tokenizer_build_dir(tokenizer_id))
    drawn = bpe.plan_sample(files, sample_bytes)
    sampled_bytes = sum(take for _, take in drawn)

    # The sample is written under the build's own directory rather than /tmp:
    # it is a few MB, it says exactly what the merges were fitted to, and a
    # tokenizer that looks wrong is otherwise unanswerable after the fact.
    sample_dir = storage.tokenizer_sample_dir(tokenizer_id)
    shutil.rmtree(sample_dir, ignore_errors=True)
    tokenizer = bpe.train(bpe.write_sample(drawn, sample_dir), max_vocab, out_dir=directory)

    stats = VocabStats(
        files=len(files),
        vocab_size=bpe.vocab_size(tokenizer),
        merges=bpe.merge_count(tokenizer),
        datasets=list(selection),
        fingerprint=file_fingerprint(storage.tokenizer_file(directory)),
        sampled_files=len(drawn),
        sampled_bytes=sampled_bytes,
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
                "files": stats.files,
                # What the merges were fitted to, as opposed to what was
                # selected. `matches` compares the budget, so raising it rebuilds
                # rather than silently reusing merges learned from less text.
                "sample_bytes": sample_bytes,
                "sampled_files": stats.sampled_files,
                "sampled_bytes": stats.sampled_bytes,
                "datasets": [[name, cap] for name, cap in selection],
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
                files=stats.files,
                vocab=stats.vocab_size,
                merges=stats.merges,
                progress=1.0,
                done=True,
            )
        )
    return stats


def _sft_hint(selection) -> str:
    """The rest of the sentence, when the corpora asked for are chat sets.

    A run is a fine-tune if and only if it names a base model, so a fine-tune
    that loses that one field on the wire arrives here as a *pretrain* over SFT
    set names — which is a vocabulary build over `.txt` files that do not exist.
    The bare "no files found" is true and says nothing; naming the other root and
    what it means is the whole difference between a puzzle and a fix.
    """
    chat = sources.chat_sets_among(name for name, _ in selection)
    if not chat:
        return ""
    return (
        f" — but {chat} are SFT sets, under {storage.sft_dir()}. A run trains on those "
        f"only as a fine-tune, which is a run that names the model it starts from: "
        f"queue it at /sft with `base_model` set. Without that field a run is a "
        f"pretrain, and a pretrain learns its vocabulary from the .txt corpora."
    )


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
    on_disk = [(name, cap) for name, cap in recorded.get("datasets", [])]
    return (
        on_disk == list(selection)
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
            "datasets": [[name, cap] for name, cap in selection],
            "max_vocab": max_vocab,
            "specials": special_tokens_sha(),
        }
    )


def _bytes_in(directory) -> int:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
