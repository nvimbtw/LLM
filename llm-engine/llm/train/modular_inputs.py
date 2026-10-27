"""Worker verification of private prepared data grants before source construction or optimizer work."""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from llm.architecture.tokenizer_binding import TOKENIZER_BYTES, binding_from_bytes
from llm.architecture.package_integrity import read_bytes
from llm.corpus.spans import SPAN_DTYPE, Spans
from llm.train import mapped_data
from llm.runtime.protocol import Train
from llm.runtime.worker import WorkerContext
from llm.train.modular_records import MAX_TRAINING_DOCUMENTS, TrainingPreparation
from llm.train.plan import DataPlan, Stage


@dataclass(frozen=True)
class TrainingInputs:
    """Own readonly corpus/span mappings and their verified schedule for the training request."""

    corpus: np.ndarray
    plan: DataPlan
    tokenizer: bytes
    mask: np.ndarray | None = None


def load_mask(prep: TrainingPreparation, context: WorkerContext) -> np.ndarray | None:
    """Verify aligned binary SFT targets in bounded readonly scans before any source/model work."""
    if prep.sft is None:
        return None
    if prep.sft.mask.size_bytes != prep.data.tokens:
        raise ValueError("SFT mask length differs from the prepared corpus")
    mask = np.memmap(context.artifacts.verify(prep.sft.mask), dtype=np.uint8, mode="r")
    targets = 0
    for start in range(0, len(mask), 1 << 20):
        context.check_cancelled()
        block = mask[start:start + (1 << 20)]
        if np.any(block > 1):
            raise ValueError("SFT mask contains a nonbinary target flag")
        targets += int(block.sum())
        mapped_data.release_pages(mask)
    if targets != prep.sft.dataset.trainable_tokens:
        raise ValueError("SFT target count differs from actual mask bytes")
    if targets == 0:
        raise ValueError("SFT corpus has no trainable targets")
    return mask


def validate_rows(rows: np.ndarray, prep: TrainingPreparation, check_cancelled: Callable[[], None]) -> None:
    """Bound each structured row's document/dataset/stage values in finite chunks without pickle or per-document objects."""
    if rows.dtype != SPAN_DTYPE or rows.ndim != 1 or len(rows) > MAX_TRAINING_DOCUMENTS:
        raise ValueError("training span table has an unsupported dtype/shape")
    for start in range(0, len(rows), 65536):
        check_cancelled()
        block = rows[start:start + 65536]
        if (np.any(block["start"] < 0) or np.any(block["start"] > prep.data.tokens)
                or np.any(block["length"] < 0) or np.any(block["length"] > prep.data.tokens - block["start"])
                or np.any(block["dataset"] < 0) or np.any(block["dataset"] >= len(prep.data.names))
                or np.any(block["stage"] < 0) or np.any(block["stage"] >= max(1, len(prep.data.stages)))):
            raise ValueError("training span table contains out-of-range document metadata")
        mapped_data.release_pages(rows)


def load_inputs(command: Train, prep: TrainingPreparation, context: WorkerContext) -> TrainingInputs:
    """Verify job, exact tokenizer, corpus IDs, span table and schedule fingerprint before any model allocation."""
    if command.job_id != prep.validation.run_id or command.corpus is None:
        raise ValueError("training preparation belongs to another job or has no packed corpus")
    binding = prep.validation.request.tokenizer
    raw = read_bytes(context.artifacts.verify(prep.tokenizer), TOKENIZER_BYTES)
    actual = binding_from_bytes(binding.tokenizer_id, raw, tuple(binding.reserved_token_ids))
    if actual.model_copy(update={"derived_from": binding.derived_from}) != binding:
        raise ValueError("training tokenizer grant differs from finalized facts")
    corpus_path = context.artifacts.verify(command.corpus)
    if command.corpus.size_bytes != prep.data.tokens * 2:
        raise ValueError("training corpus length/dtype differs from prepared data")
    corpus = np.memmap(corpus_path, dtype=np.uint16, mode="r")
    for start in range(0, len(corpus), 1 << 20):
        context.check_cancelled()
        if np.any(corpus[start:start + (1 << 20)] >= binding.vocabulary_size):
            raise ValueError("training corpus contains an ID outside the final vocabulary")
        mapped_data.release_pages(corpus)
    rows = np.load(context.artifacts.verify(prep.data.spans), mmap_mode="r", allow_pickle=False)
    if not isinstance(rows, np.ndarray):
        raise ValueError("training span grant is not one numeric array")
    validate_rows(rows, prep, context.check_cancelled)
    documents = Spans(list(prep.data.names), rows)
    stages = [Stage(name=stage.name, epochs=stage.epochs, documents=documents.in_stage(index))
              for index, stage in enumerate(prep.data.stages)]
    plan = DataPlan(documents=documents, eot_id=prep.data.eot_id, shuffle=prep.data.shuffle, seed=prep.data.seed, stages=stages)
    mask = load_mask(prep, context)
    if prep.sft is not None and len(documents) != prep.sft.dataset.conversations:
        raise ValueError("SFT conversation count differs from the actual span table")
    arrays = (corpus,) if mask is None else (corpus, mask)
    if mapped_data.fingerprint(*arrays, plan=plan, check_cancelled=context.check_cancelled) != prep.data.fingerprint:
        raise ValueError("training packed-data/schedule fingerprint changed")
    return TrainingInputs(corpus, plan, raw, mask)
