"""Single source of truth for every on-disk path. Zero sibling imports.

Layout under `engine/data/` (mirrors backend/crates/storage):

    input/<dataset>/*.txt        source text, one subfolder per dataset
    sft/<set>/*.jsonl            chat conversations, one subfolder per set
    datasets/                    corpus.bin + segments.json (training-ready)
    datasets/sft/<key>/          one built SFT dataset per selection (see below)
    artifacts/tokenizer/         the trained tokenizer
    artifacts/models/<name>/     weights, a copy of the tokenizer, meta.json
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


def corpus_path() -> Path:
    return datasets_dir() / "corpus.bin"


def segments_path() -> Path:
    return datasets_dir() / "segments.json"


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
    return root() / "artifacts" / "tokenizer"


def tokenizer_file(directory: Path | None = None) -> Path:
    """The tokenizer's single-file form. Defaults to the shared build; pass a
    model dir for the copy saved beside that model's weights."""
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
