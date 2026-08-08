"""Where the fetch scripts put what they download.

One place, resolved from this file's own location, because the alternative is an
absolute path per script — and those had all drifted to
`/home/x_x/Programming/LLM/...`, missing a directory, so every script was
writing into a tree nothing reads.

The engine owns the data. These directories are its input side:

    engine/data/input/<Dataset>/*.txt   pretrain corpora, one folder per dataset
    engine/data/sft/*.jsonl             chat conversations for fine-tuning
    engine/data/eval/corpus/*.txt       held-out prose the benchmark scores against
    engine/data/eval/chat/*.jsonl       held-out conversations

`eval/` is deliberately not under `input/`: everything under input is selectable
as training data, and a held-out set that can be trained on by accident is not a
held-out set.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "engine", "data")


def dataset_dir(name: str) -> str:
    """A pretrain corpus folder. The name is what the training UI offers."""
    return os.path.join(DATA, "input", name)


def sft_dir() -> str:
    return os.path.join(DATA, "sft")


def eval_corpus_dir() -> str:
    return os.path.join(DATA, "eval", "corpus")


def eval_chat_dir() -> str:
    return os.path.join(DATA, "eval", "chat")
