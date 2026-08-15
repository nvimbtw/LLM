"""What keeps a build inside the machine it runs on.

Three costs used to scale with the data rather than with the run: the tokenizer
learned its merges from every selected file, the packer computed nothing it could
have recorded, and the training loop derived the diagnostics' token ranking by
reading the whole corpus before its first step. Measured on the 51 GB input tree
and the 8.3 GB packed corpus, that was ~15 GB of peak RSS and a minute of page
faults on a 12 GB machine.

These are the assertions that keep those three bounded.
"""

from pathlib import Path

import numpy as np
import pytest

from llm.corpus import bpe, pack, sources, vocab
from llm.train import diagnostics
from tests.fixtures.tiny_env import TINY_DATASET


class Source:
    """The duck `plan_sample` types: a path and the dataset it came from."""

    def __init__(self, path: Path, dataset: str):
        self.path = path
        self.dataset = dataset


@pytest.fixture
def spread(tmp_path) -> list[Source]:
    """Three datasets of very different shapes, as the real input tree has:
    a dominant sharded corpus, a mid-sized one, and one tiny folder."""
    files = []
    for name, shards, size in (("Big", 400, 40_000), ("Mid", 40, 20_000), ("Tiny", 2, 500)):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(shards):
            path = folder / f"{i:04d}.txt"
            # Distinct per shard, so a sample can be traced back to where it came
            # from — which is the whole point of the stride.
            path.write_text(f"shard {name} {i}\n" * (size // 16), encoding="utf-8")
            files.append(Source(path, name))
    return files


# ── the tokenizer's sample ──────────────────────────────────────────────────


def test_the_sample_is_bounded_by_the_budget_not_by_the_selection(spread):
    """The ceiling that was missing. The trainer holds a count per distinct
    pretoken, so its peak follows the variety of what it is shown — and it was
    shown everything selected."""
    selected = sum(f.path.stat().st_size for f in spread)
    budget = 1 << 16

    drawn = bpe.plan_sample(spread, budget)

    assert selected > 4 * budget, "fixture too small to constrain anything"
    assert sum(take for _, take in drawn) <= budget


def test_every_selected_dataset_reaches_the_sample(spread):
    """Without a floor the dominant corpus takes nearly all of it, and the merges
    are fitted to one distribution while the corpus has another."""
    drawn = bpe.plan_sample(spread, 1 << 18)

    per = {}
    for path, take in drawn:
        per[path.parent.name] = per.get(path.parent.name, 0) + take

    assert set(per) == {"Big", "Mid", "Tiny"}
    # Proportionality survives the floor: the dominant corpus still leads.
    assert per["Big"] > per["Mid"] > per["Tiny"]


def test_a_dataset_is_sampled_across_its_whole_range(spread):
    """ADR 0004's guard, ported. A sharded corpus is ordered, so its first N
    shards are one region of it and not a sample of it — the Rust engine read
    FineWeb from 50 of its 5,498 shards for exactly this reason, and the
    diagnostics histogram showed the same trap here: the first 200M tokens of the
    packed corpus contain 16 of its true top 32 tokens."""
    drawn = bpe.plan_sample(spread, 1 << 18)

    indices = sorted(int(path.stem) for path, _ in drawn if path.parent.name == "Big")

    assert len(indices) > 1
    # Every quarter of the dataset is represented — the property that fails when
    # a sample is taken from the front, however much of it is taken.
    for quarter in range(4):
        lo, hi = quarter * 100, (quarter + 1) * 100
        assert any(lo <= i < hi for i in indices), f"nothing drawn from shards {lo}-{hi}"


def test_a_single_large_file_cannot_eat_its_dataset_s_share(tmp_path):
    """`input/Books` holds a 28 MB file. Taken whole against a 2 MB share it
    would be 14x the budget for that dataset and re-skew the composition the
    shares exist to control."""
    folder = tmp_path / "Books"
    folder.mkdir()
    big = folder / "one.txt"
    big.write_text("a line of a book\n" * 60_000, encoding="utf-8")

    drawn = bpe.plan_sample([Source(big, "Books")], 1 << 14)

    assert sum(take for _, take in drawn) <= 1 << 14
    assert big.stat().st_size > 4 * (1 << 14)


def test_the_sample_is_the_same_one_every_time(spread):
    """A vocabulary is reused across runs and a corpus is keyed on it. A sample
    that drifted would re-learn the merges and re-pack the corpus."""
    assert bpe.plan_sample(spread, 1 << 17) == bpe.plan_sample(spread, 1 << 17)


def test_a_written_sample_is_cut_at_a_line_boundary(tmp_path):
    """A byte-level BPE pretokenizer splits on whitespace, so a cut mid-word
    invents a pretoken the text does not contain and spends merges on it."""
    source = tmp_path / "src.txt"
    source.write_text("alpha beta gamma\n" * 100, encoding="utf-8")

    written = bpe.write_sample([(source, 100)], tmp_path / "out")
    text = written[0].read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert all(line == "alpha beta gamma" for line in text.splitlines())


def test_a_build_records_what_its_merges_were_fitted_to(data_root):
    """"Which corpora" stopped answering "which text" the moment a budget
    existed, and a vocabulary that looks wrong is otherwise unanswerable after
    the fact."""
    selection = [(TINY_DATASET, None)]
    tokenizer_id = vocab.auto_id(selection, 512)

    stats = vocab.build(tokenizer_id, selection, 512)
    recorded = vocab.info(tokenizer_id)

    assert stats.sampled_bytes > 0
    assert recorded["sampled_bytes"] == stats.sampled_bytes
    assert recorded["sample_bytes"] == bpe.SAMPLE_BYTES
    # The sample itself is kept beside the build that learned from it.
    assert list(storage_sample_files(tokenizer_id))


def test_a_build_made_under_a_different_budget_is_not_reused(data_root):
    """Raising the budget is a request for merges fitted to more text. Reusing
    the old ones would answer a question nobody asked."""
    selection = [(TINY_DATASET, None)]
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(tokenizer_id, selection, 512)

    assert vocab.matches(tokenizer_id, selection, 512)
    assert not vocab.matches(tokenizer_id, selection, 512, sample_bytes=1 << 10)


def storage_sample_files(tokenizer_id: str):
    from llm import storage

    return sorted(storage.tokenizer_sample_dir(tokenizer_id).glob("*.txt"))


# ── the packed corpus ───────────────────────────────────────────────────────


@pytest.fixture
def packed(data_root):
    """The fixture corpus, packed the way a run packs it."""
    selection = [(TINY_DATASET, None)]
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(tokenizer_id, selection, 512)
    tokenizer = vocab.load(tokenizer_id)
    fingerprint = vocab.fingerprint(tokenizer_id)
    key = pack.corpus_key(fingerprint, [(None, selection)])

    stats = pack.pack(
        sources.collect([TINY_DATASET]),
        tokenizer,
        key=key,
        stages=[{"name": None, "epochs": 1, "datasets": [[TINY_DATASET, None]]}],
        tokenizer_id=tokenizer_id,
        fingerprint=fingerprint,
    )
    return key, stats


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
    from llm.corpus.pack import DocSpan
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
