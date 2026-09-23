"""Where the fetch scripts put what they download.

One place, resolved from this file's own location, because the alternative is an
absolute path per script — and those had all drifted to
`/home/x_x/Programming/LLM/...`, missing a directory, so every script was
writing into a tree nothing reads.

The LLM engine owns the data. These directories are its input side:

    llm-engine/data/input/<Dataset>/*.txt   pretrain corpora, one folder per dataset
    llm-engine/data/sft/<Set>/*.jsonl       chat conversations, one folder per SFT set
    llm-engine/data/eval/corpus/*.txt       held-out prose the benchmark scores against
    llm-engine/data/eval/chat/*.jsonl       held-out conversations

Both training roots are one-folder-per-selectable-thing, because a fine-tune
picks SFT sets exactly as a pretrain picks corpora — see llm-engine/llm/corpus/
sources.py. Loose `.jsonl` left at the SFT root still trains, as a set named
`root`, but only when a run asks for it by that name.

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


def sft_set_dir(name: str) -> str:
    """One selectable SFT set. The name is what a fine-tune's `datasets` asks for."""
    return os.path.join(DATA, "sft", name)


def eval_corpus_dir() -> str:
    return os.path.join(DATA, "eval", "corpus")


def eval_chat_dir() -> str:
    return os.path.join(DATA, "eval", "chat")
