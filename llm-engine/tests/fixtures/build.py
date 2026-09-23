"""Regenerate the committed `tiny/` fixture model.

Run only when the checkpoint format or the special-token block changes; the
output is committed so the dataflow trace does not depend on torch's RNG
staying stable across versions.

    llm-engine/.venv/bin/python -m tests.fixtures.build
"""

import shutil
import tempfile
from pathlib import Path

import torch

from llm.corpus import bpe
from llm.model import checkpoint
from llm.model.transformer import Transformer

HERE = Path(__file__).resolve().parent
TINY = HERE / "tiny"

# Enough repetition for BPE to find merges, short enough to train instantly.
CORPUS = (
    "the cat sat on the mat. the cat ate the fish. "
    "a dog sat on a log. the dog saw the cat. "
    "birds sing in the morning and the morning is bright. "
) * 40

SEED = 20260813
DIMENSIONS = 32
HEADS = 4
LAYERS = 2
FF_MULTIPLIER = 4
CTX_WINDOW = 128
MAX_VOCAB = 320


def build() -> Path:
    if TINY.exists():
        shutil.rmtree(TINY)
    TINY.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "corpus.txt"
        source.write_text(CORPUS, encoding="utf-8")
        tokenizer = bpe.train([source], MAX_VOCAB, out_dir=TINY)

    torch.manual_seed(SEED)
    model = Transformer(
        vocab=bpe.vocab_size(tokenizer),
        dimensions=DIMENSIONS,
        heads=HEADS,
        layers=LAYERS,
        ff_multiplier=FF_MULTIPLIER,
        ctx_window=CTX_WINDOW,
    )
    checkpoint.save(model, TINY / "model.pt")
    return TINY


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}")
    for entry in sorted(out.iterdir()):
        print(f"  {entry.name}  {entry.stat().st_size:,} bytes")
