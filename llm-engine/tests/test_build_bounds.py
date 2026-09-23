"""What keeps a build inside the machine it runs on.

Three costs used to scale with the data rather than with the run: the tokenizer
learned its merges from every selected file, the packer computed nothing it could
have recorded, and the training loop derived the diagnostics' token ranking by
reading the whole corpus before its first step. Measured on the 51 GB input tree
and the 8.3 GB packed corpus, that was ~15 GB of peak RSS and a minute of page
faults on a 12 GB machine.

These are the assertions that keep those three bounded.
"""

import numpy as np

from llm.corpus import bpe, pack, vocab
from llm.train import diagnostics
from tests.fixtures.tiny_env import TINY_DATASET_ID


# ── the tokenizer's sample ──────────────────────────────────────────────────


def test_every_selected_dataset_reaches_the_sample():
    """Without a floor the dominant corpus takes nearly all of the budget, and
    the merges are fitted to one distribution while the corpus has another. ADR
    0004 measured an equal split giving FineWeb 27.9% of the sample while it was
    98.4% of the tokens actually encoded."""
    sizes = {"Big": 16_000_000, "Mid": 800_000, "Tiny": 1_000}

    split = dict(bpe.shares(sizes, 1 << 20))

    assert set(split) == set(sizes)
    # Proportionality survives the floor: the dominant corpus still leads.
    assert split["Big"] > split["Mid"] > split["Tiny"]
    assert all(share > 0 for share in split.values())


def test_the_sample_is_bounded_by_the_budget_not_by_the_selection():
    """The ceiling that was missing. The trainer holds a count per distinct
    pretoken, so its peak follows the variety of what it is shown — and it was
    shown everything selected."""
    sizes = {"Big": 16_000_000, "Mid": 800_000, "Tiny": 1_000}
    budget = 1 << 16

    split = dict(bpe.shares(sizes, budget))

    assert sum(sizes.values()) > 4 * budget, "fixture too small to constrain anything"
    assert sum(split.values()) <= budget


def test_a_dataset_smaller_than_its_share_hands_the_difference_back():
    """Water-filling. Without it, selecting one 400 KB corpus alongside a large
    one silently shrinks the whole sample by the difference."""
    budget = 1 << 20
    sizes = {"Big": 16_000_000, "Small": 4_000}

    split = dict(bpe.shares(sizes, budget))

    assert split["Small"] == 4_000
    assert split["Big"] >= budget - 4_000 - 1


def test_a_build_records_what_its_merges_were_fitted_to(data_root, data_client):
    """"Which corpora" stopped answering "which text" the moment a budget
    existed, and a vocabulary that looks wrong is otherwise unanswerable after
    the fact."""
    selection = vocab.resolve(data_client, [TINY_DATASET_ID])
    tokenizer_id = vocab.auto_id(selection, 512)

    stats = vocab.build(data_client, tokenizer_id, selection, 512)
    recorded = vocab.info(tokenizer_id)

    assert stats.sampled_bytes > 0
    assert recorded["sampled_bytes"] == stats.sampled_bytes
    assert recorded["sample_bytes"] == bpe.SAMPLE_BYTES
    # The sample itself is kept beside the build that learned from it.
    assert list(storage_sample_files(tokenizer_id))


def test_a_build_made_under_a_different_budget_is_not_reused(data_root, data_client):
    """Raising the budget is a request for merges fitted to more text. Reusing
    the old ones would answer a question nobody asked."""
    selection = vocab.resolve(data_client, [TINY_DATASET_ID])
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(data_client, tokenizer_id, selection, 512)

    assert vocab.matches(tokenizer_id, selection, 512)
    assert not vocab.matches(tokenizer_id, selection, 512, sample_bytes=1 << 10)


def storage_sample_files(tokenizer_id: str):
    from llm import storage

    return sorted(storage.tokenizer_sample_dir(tokenizer_id).glob("*.txt"))


# ── the packed corpus ───────────────────────────────────────────────────────


def test_a_pack_records_the_histogram_it_already_had(packed):
    """Every id passes through the packer once, as a numpy array, at the one
    moment the whole corpus is in hand. Deriving it later cost 64 s and 8.3 GB
    of resident mapping before the first training step of every run."""
    key, stats = packed
    _, _, info = pack.load_segments(key)
    counts = info["token_counts"]

    assert counts is not None
    # Every token in the corpus is counted exactly once.
    assert sum(counts) == stats.corpus_len


def test_the_recorded_histogram_ranks_what_a_full_count_ranks(packed):
    """The recorded counts are the answer, not an approximation of it."""
    key, _ = packed
    corpus = pack.load_corpus(key)
    _, _, info = pack.load_segments(key)

    exact = diagnostics.histogram(corpus, len(info["token_counts"]))

    assert list(info["token_counts"]) == list(exact)
    assert diagnostics.rank_tokens(info["token_counts"]) == diagnostics.rank_tokens(exact)


def test_batched_encoding_is_the_same_encoding(packed, tokenizer):
    """Packing hands the encoder a batch at a time to use every core. The batch
    is a scheduling detail — if it were not, every corpus packed since would be
    silently different from the text it claims to be."""
    texts = ["the cat sat on the mat.", "", "birds sing in the morning.\nand at night."]

    assert bpe.encode_batch(tokenizer, texts) == [bpe.encode(tokenizer, t) for t in texts]


# ── the diagnostics ─────────────────────────────────────────────────────────


def test_the_sampled_histogram_reads_a_bounded_slice_of_a_corpus(monkeypatch):
    """The fallback for a corpus packed before the histogram was recorded. It has
    to skip *pages*, not elements: a `[::16]` stride reads every 16th two-byte
    token, which still touches every 4 KiB page and so still reads all of it."""
    read = []

    corpus = np.arange(1000, dtype=np.uint16) % 7
    monkeypatch.setattr(diagnostics, "_COUNT_BLOCK", 10)
    monkeypatch.setattr(diagnostics, "_SAMPLE_BLOCKS", 4)

    class Watched(np.ndarray):
        def __getitem__(self, item):
            if isinstance(item, slice):
                read.append(item)
            return super().__getitem__(item)

    diagnostics.sampled_histogram(corpus.view(Watched), 7)

    assert len(read) == 4
    # Contiguous blocks, spread from the head to the tail.
    assert all(s.stop - s.start == 10 for s in read)
    assert read[0].start == 0
    assert read[-1].stop == len(corpus)


def test_a_short_corpus_is_counted_exactly_rather_than_sampled():
    """Sampling something smaller than the sample is only a way to get it wrong."""
    corpus = np.array([1, 1, 2, 3, 3, 3], dtype=np.uint16)

    assert list(diagnostics.sampled_histogram(corpus, 4)) == [0, 2, 1, 3]
    assert diagnostics.frequent_tokens(corpus, 2) == [3, 1]


def test_the_collector_is_not_built_until_something_diagnoses(data_root, monkeypatch):
    """It was constructed at the top of every run, including runs that never emit
    a frame — `bench.py` paid the full corpus scan on every timing it took."""
    from llm.config import TrainConfig
    from llm.corpus.spans import DocSpan
    from llm.model.transformer import Transformer
    from llm.train import loop
    from llm.train.plan import DataPlan

    built = []
    monkeypatch.setattr(loop, "Collector", lambda *args: built.append(args) or _Stub())

    corpus = np.arange(4096, dtype=np.uint16) % 61
    model = Transformer(
        vocab=61, dimensions=16, heads=2, layers=1, ff_multiplier=2, ctx_window=16
    )
    plan = DataPlan(documents=[DocSpan("x", 0, len(corpus))], eot_id=0, shuffle=False, seed=0)
    cfg = TrainConfig(model_name="unused", ctx_window=16, batch_size=2, epochs=1,
                      dimensions=16, heads=2, layers=1)

    loop.run(model, corpus, plan, cfg, "cpu", on_progress=None)

    assert built == []


class _Stub:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None
