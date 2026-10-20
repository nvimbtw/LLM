"""Single source of truth for every on-disk path. Zero sibling imports.

Layout under `llm-engine/data/`. There is no source text here: the corpora a run
trains on belong to the data engine and are read a page at a time through the
backend (`llm.corpus.remote`). What is left is derived — and `eval/`, which is
held out from training and therefore not training data.

    eval/corpus/*.txt            held-out prose, and eval/chat/*.jsonl
    datasets/pretrain/<key>/     one packed corpus per (vocabulary x selection)
    datasets/sft/<key>/          one built SFT dataset per selection (see below)
    artifacts/tokenizers/<id>/   one built vocabulary, and what it was built from
    artifacts/models/<id>/       weights, a copy of the tokenizer, meta.json,
                                 and train_state.pt where the run left one
    artifacts/.recovery-staging/ unpublished checkpoint directories
    artifacts/validations/<uuid>/ immutable validation graphs/reports
    artifacts/definitions/<sha>/ immutable resolved definition/source snapshots
    artifacts/.runtime/<uuid>/   supervised worker session materializations
    artifacts/.runtime/engine.lock engine lifetime exclusion

    datasets/corpus.bin          the pre-keyed singletons, left where they are
    datasets/segments.json       (see `legacy_pretrain_paths`)
    artifacts/tokenizer/
"""

import os
import re
import uuid
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


def pretrain_spans_path(key: str) -> Path:
    """The document table, beside the corpus it indexes.

    Split out of `segments.json` because it is the one part that scales with the
    number of documents rather than with what the build was made from — see
    `llm.corpus.spans`. The sidecar keeps the provenance, this keeps the rows.
    """
    return pretrain_dataset_dir(key) / "spans.npy"


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


def sft_spans_path(key: str) -> Path:
    """The conversation table. `pretrain_spans_path`, for the other build."""
    return sft_dataset_dir(key) / "spans.npy"


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

    A build learns from a bounded sample of the selection (`bpe.SAMPLE_BYTES`),
    so "which datasets" no longer answers "which text". Kept inside the build so
    it travels with it and `delete` takes it too — a few MB against a selection
    that can run to tens of GB.

    It is also the only text the LLM engine still writes down, and deliberately so:
    the trainer counts across cores over files it is given, and holding the same
    64 MiB as Python strings costs more than writing it. A build artifact, not a
    corpus."""
    return tokenizer_build_dir(tokenizer_id) / "sample"


def tokenizer_file(directory: Path | None = None) -> Path:
    """The tokenizer's single-file form. Pass the directory holding it — a build
    or a model dir. The default is the legacy shared build, and only the tools
    that predate named builds still take it."""
    return (directory or tokenizer_dir()) / "tokenizer.json"


def models_dir() -> Path:
    return root() / "artifacts" / "models"


def recovery_staging_dir() -> Path:
    return root() / "artifacts" / ".recovery-staging"


def validations_dir() -> Path:
    """Return the engine-owned parent of immutable validation artifacts without creating it."""
    return root() / "artifacts" / "validations"


def validation_dir(run_id: uuid.UUID | str) -> Path:
    """Resolve a validated run UUID to its report/graph directory without filesystem effects."""
    return validations_dir() / str(uuid.UUID(str(run_id)))


def definitions_dir() -> Path:
    """Return the immutable semantic-definition artifact parent without creating it."""
    return root() / "artifacts" / "definitions"


def definition_dir(definition_id: str) -> Path:
    """Resolve one lowercase SHA-256 definition identity, rejecting path traversal or names."""
    if re.fullmatch(r"[0-9a-f]{64}", definition_id) is None:
        raise ValueError("definition_id must be a lowercase SHA-256 digest")
    return definitions_dir() / definition_id


def runtime_root() -> Path:
    """Return private supervised worker materializations, separate from published models."""
    return root() / "artifacts" / ".runtime"


def runtime_session_dir(worker_id: uuid.UUID | str) -> Path:
    """Resolve a validated worker UUID beneath its engine-owned runtime directory."""
    return runtime_root() / str(uuid.UUID(str(worker_id)))


def runtime_lock_path() -> Path:
    """Return the single engine-lifetime lock path; the supervisor owns locking and cleanup."""
    return runtime_root() / "engine.lock"


def model_dir(model_name: str) -> Path:
    return models_dir() / model_name


def weights_path(model_name: str) -> Path:
    return model_dir(model_name) / "model.pt"


def meta_path(model_name: str) -> Path:
    return model_dir(model_name) / "meta.json"


def train_state_path(model_name: str) -> Path:
    """The optimizer moments, the RNG and the position a run stopped at.

    Inside the model directory rather than in a `checkpoints/` subtree of the run
    that produced it, because a checkpoint *is* a model: its own id, its own
    directory, and this one extra file. That replaced a per-epoch
    `checkpoints/epochNNN.pt` holding weights alone — which nothing could resume
    from, chat with, or benchmark.
    """
    return model_dir(model_name) / "train_state.pt"


def ensure_dir(path: Path) -> Path:
    """Create `path` and any missing parents; no-op if it already exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
