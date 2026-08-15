"""Single source of truth for every on-disk path. Zero sibling imports.

Layout under `engine/data/` (mirrors backend/crates/storage):

    input/<dataset>/*.txt        source text, one subfolder per dataset
    sft/<set>/*.jsonl            chat conversations, one subfolder per set
    datasets/pretrain/<key>/     one packed corpus per (vocabulary x selection)
    datasets/sft/<key>/          one built SFT dataset per selection (see below)
    artifacts/tokenizers/<id>/   one built vocabulary, and what it was built from
    artifacts/models/<id>/       weights, a copy of the tokenizer, meta.json

    datasets/corpus.bin          the pre-keyed singletons, left where they are
    datasets/segments.json       (see `legacy_pretrain_paths`)
    artifacts/tokenizer/
"""

import os
from pathlib import Path


def root() -> Path:
    """Root of all data. Override with DATA_DIR.

    Anchored to the package location rather than the working directory: uvicorn
    can be started from anywhere, and a CWD-relative default silently writes a
    run's weights to whichever directory the server happened to launch from.
    """
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def input_dir() -> Path:
    return root() / "input"


def sft_dir() -> Path:
    """Chat conversations for supervised fine-tuning, as JSONL.

    Kept out of `input/` on purpose: everything under input/ is selectable as
    pretrain text, and SFT consumes a different format entirely.

    One subfolder per selectable set, mirroring `input/`, so a fine-tune can ask
    for the non-thinking conversations or the thinking ones and get two
    comparable models rather than one blend.
    """
    return root() / "sft"


def sft_set_dir(name: str) -> Path:
    """One selectable SFT set. The name is what a run's `datasets` asks for."""
    return sft_dir() / name


def eval_dir() -> Path:
    """Benchmark fixtures: `eval/corpus/*.txt` is held-out prose, `eval/chat/*.jsonl`
    held-out conversations.

    Kept out of `input/` and `sft/` on purpose — anything under those is
    selectable as training data, and a held-out set that can be accidentally
    trained on is not a held-out set.
    """
    return root() / "eval"


def datasets_dir() -> Path:
    return root() / "datasets"


def pretrain_dataset_dir(key: str) -> Path:
    """One packed pretrain corpus, keyed by the vocabulary and the selection that
    produced it (see `llm.corpus.pack.corpus_key`).

    Keyed for the reason the SFT build is, and one more: the same `corpus.bin`
    bytes mean *different text* under a different vocabulary, so a corpus shared
    between two tokenizers is not merely the wrong data, it is unreadable data
    that loads without complaint.
    """
    return datasets_dir() / "pretrain" / key


def pretrain_corpus_path(key: str) -> Path:
    return pretrain_dataset_dir(key) / "corpus.bin"


def pretrain_segments_path(key: str) -> Path:
    return pretrain_dataset_dir(key) / "segments.json"


def legacy_pretrain_paths() -> list[Path]:
    """The pre-keyed singleton corpus and the shared tokenizer beside it.

    Left on disk rather than deleted: unlike the SFT singleton these are what
    `explore/` and `bench.py` still read, and neither records which datasets or
    which vocabulary produced it — so nothing here can be adopted into a key, and
    nothing new reads them.
    """
    return [datasets_dir() / "corpus.bin", datasets_dir() / "segments.json", tokenizer_dir()]


def sft_dataset_dir(key: str) -> Path:
    """One built SFT dataset, in its own directory keyed by the request that
    produced it (see `llm.corpus.sft.dataset_key`).

    Keyed rather than singleton because the build is not a function of the data
    alone: which sets were selected and whether reasoning was kept both change
    every token and every mask byte. Sharing one path meant a second run reused
    the first run's corpus whenever `rebuild_dataset` was false — training on the
    opposite of what was asked for, while `meta.json` claimed otherwise.
    """
    return datasets_dir() / "sft" / key


def sft_corpus_path(key: str) -> Path:
    return sft_dataset_dir(key) / "corpus.bin"


def sft_mask_path(key: str) -> Path:
    return sft_dataset_dir(key) / "mask.bin"


def sft_segments_path(key: str) -> Path:
    return sft_dataset_dir(key) / "segments.json"


def legacy_sft_paths() -> list[Path]:
    """The pre-keyed singleton build, wherever it still exists.

    Its provenance was never recorded, so it cannot be adopted into a key — it
    is deleted on the next build rather than guessed at.
    """
    return [
        datasets_dir() / name
        for name in ("sft_corpus.bin", "sft_mask.bin", "sft_segments.json")
    ]


def tokenizer_dir() -> Path:
    """The pre-keyed shared tokenizer, overwritten by every pretrain that ran
    before vocabularies were built and named. Legacy — see
    `legacy_pretrain_paths`."""
    return root() / "artifacts" / "tokenizer"


def tokenizers_dir() -> Path:
    return root() / "artifacts" / "tokenizers"


def tokenizer_build_dir(tokenizer_id: str) -> Path:
    """One built vocabulary. The id is the directory, so a name can change
    without moving bytes — the same arrangement `model_dir` has."""
    return tokenizers_dir() / tokenizer_id


def tokenizer_meta_path(tokenizer_id: str) -> Path:
    """What a vocabulary was built from, beside the vocabulary itself."""
    return tokenizer_build_dir(tokenizer_id) / "build.json"


def tokenizer_sample_dir(tokenizer_id: str) -> Path:
    """The text a vocabulary's merges were actually fitted to.

    A build learns from a bounded sample of the selection (`bpe.plan_sample`), so
    "which corpora" no longer answers "which text". Kept inside the build so it
    travels with it and `delete` takes it too — a few MB against a selection that
    can run to tens of GB."""
    return tokenizer_build_dir(tokenizer_id) / "sample"


def tokenizer_file(directory: Path | None = None) -> Path:
    """The tokenizer's single-file form. Pass the directory holding it — a build
    or a model dir. The default is the legacy shared build, and only the tools
    that predate named builds still take it."""
    return (directory or tokenizer_dir()) / "tokenizer.json"


def models_dir() -> Path:
    return root() / "artifacts" / "models"


def model_dir(model_name: str) -> Path:
    return models_dir() / model_name


def weights_path(model_name: str) -> Path:
    return model_dir(model_name) / "model.pt"


def meta_path(model_name: str) -> Path:
    return model_dir(model_name) / "meta.json"


def checkpoints_dir(model_name: str) -> Path:
    return model_dir(model_name) / "checkpoints"


def ensure_dir(path: Path) -> Path:
    """Create `path` and any missing parents; no-op if it already exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
