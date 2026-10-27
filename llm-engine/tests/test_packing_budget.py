"""Streamed pretraining output budgets preserve complete packs and exact token/span semantics."""

import numpy as np
import pytest

from llm import storage
from llm.corpus import pack
from llm.corpus.baseline import byte_alphabet


@pytest.mark.parametrize("ceiling", [0, -1, True, 1.5])
def test_packing_requires_positive_integer_budgets(ceiling):
    """Invalid byte or row ceilings fail before any source or output work."""
    for values in ((ceiling, 1), (1, ceiling)):
        with pytest.raises(ValueError, match="positive integer"):
            pack.PackBudget(*values)


@pytest.mark.parametrize("budget,message", [
    (pack.PackBudget(7, 2), "byte budget"),
    (pack.PackBudget(8, 1), "document row budget"),
])
def test_budget_refusal_discards_partial_bytes_and_preserves_complete_build(data_root, monkeypatch, budget, message):
    """A second document crossing either cap cannot replace prior corpus, spans or sidecar, even after flushing."""
    tokenizer = byte_alphabet()
    documents = [pack.SourceDoc("first", "a"), pack.SourceDoc("second", "b")]
    pack.pack(documents, tokenizer, key="budget")
    directory = storage.pretrain_dataset_dir("budget")
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(pack, "FLUSH_TOKENS", 1)
    monkeypatch.setattr(pack, "ENCODE_BATCH_CHARS", 1)
    with pytest.raises(ValueError, match=message):
        pack.pack(documents, tokenizer, key="budget", budget=budget)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_byte_budget_counts_the_document_separator_before_writing(data_root):
    """A three-byte ceiling refuses one uint16 token plus its uint16 separator and leaves no build artifacts."""
    with pytest.raises(ValueError, match="byte budget"):
        pack.pack([pack.SourceDoc("d", "a")], byte_alphabet(), key="separator", budget=pack.PackBudget(3, 1))
    assert not list(storage.pretrain_dataset_dir("separator").iterdir())


def test_exact_budget_keeps_tokens_spans_empty_documents_and_chunk_boundaries_identical(data_root, monkeypatch):
    """Admitted packing produces byte-identical token/span/sidecar output across boundaries and empty documents."""
    tokenizer = byte_alphabet()
    documents = [pack.SourceDoc("first", "a b c"), pack.SourceDoc("empty", ""), pack.SourceDoc("last", "d")]
    stats = pack.pack(documents, tokenizer, key="exact")
    directory = storage.pretrain_dataset_dir("exact")
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(pack, "ENCODE_BATCH_CHARS", 1)
    monkeypatch.setattr(pack.bpe, "CHUNK_CHARS", 1)
    repeated = pack.pack(documents, tokenizer, key="exact",
                         budget=pack.PackBudget(stats.corpus_len * np.dtype(pack.CORPUS_DTYPE).itemsize, stats.documents))
    assert repeated == stats
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_source_failure_discards_partial_output_without_losing_the_prior_build(data_root, monkeypatch):
    """A read failure after a flushed batch propagates and removes the incomplete token file."""
    tokenizer = byte_alphabet()
    pack.pack([pack.SourceDoc("d", "old")], tokenizer, key="failed-source")
    directory = storage.pretrain_dataset_dir("failed-source")
    original = {path.name: path.read_bytes() for path in directory.iterdir()}

    def failed_source():
        """Supply one actual document before reporting a failed upstream page."""
        yield pack.SourceDoc("d", "new")
        raise RuntimeError("upstream page failed")

    monkeypatch.setattr(pack, "ENCODE_BATCH_CHARS", 1)
    monkeypatch.setattr(pack, "FLUSH_TOKENS", 1)
    with pytest.raises(RuntimeError, match="upstream page failed"):
        pack.pack(failed_source(), tokenizer, key="failed-source", budget=pack.PackBudget(1024, 10))
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
