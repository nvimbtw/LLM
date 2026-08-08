"""The tokenizer: training, loading, encoding, decoding.

Wraps HuggingFace `tokenizers` (byte-level BPE) rather than reimplementing the
Rust crate's BPE — it is the same algorithm and is itself Rust underneath. What
this module owns is the *policy* the library has no opinion on: the reserved
special-token block, and saving/loading as one self-contained file.
"""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

from llm import storage
from llm.corpus import special

# Merges seen fewer times than this are noise rather than structure.
MIN_FREQUENCY = 2


def train(files: list[Path], max_vocab: int, out_dir: Path | None = None) -> Tokenizer:
    """Train a byte-level BPE tokenizer over `files` and save it.

    Saved as a single `tokenizer.json` rather than the `vocab.json`/`merges.txt`
    pair: that pair does NOT carry the added-token configuration, so a tokenizer
    reloaded from it splits every `<|marker|>` back into byte merges — the model
    then never sees the single ids it was trained on.
    """
    out_dir = storage.ensure_dir(out_dir or storage.tokenizer_dir())

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(f) for f in files],
        vocab_size=max_vocab,
        min_frequency=MIN_FREQUENCY,
        special_tokens=special.token_list(),
        show_progress=False,
    )

    tok.save(str(storage.tokenizer_file(out_dir)))
    return load(out_dir)


def load(directory: Path | None = None) -> Tokenizer:
    """Load a trained tokenizer. Defaults to the shared build; pass a model's
    directory to load the copy saved beside its weights."""
    path = storage.tokenizer_file(directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Train a model first — the training "
            f"pipeline builds one and copies it next to the weights."
        )
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def vocab_size(tok: Tokenizer) -> int:
    return tok.get_vocab_size()


def token_id(tok: Tokenizer, token: str) -> int:
    """The single id for a special token. Raises rather than returning None —
    a missing marker means the tokenizer was trained without the reserved block,
    and every downstream document boundary would be silently wrong."""
    tid = tok.token_to_id(token)
    if tid is None:
        raise ValueError(f"{token!r} is not in the tokenizer's vocabulary")
    return tid
