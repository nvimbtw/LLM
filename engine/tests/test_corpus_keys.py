"""What a built vocabulary and a packed corpus are functions of.

The pretrain build used to be a singleton decided by "do these two files exist",
so a run with different corpora trained on whatever the last run left behind.
These are the assertions that keep the answer honest.
"""

import uuid

import pytest

from llm import storage
from llm.config import CurriculumStage, TrainConfig
from llm.corpus import pack, vocab
from llm.orchestrate import train as train_pipeline
from tests.fixtures.tiny_env import TINY_DATASET


@pytest.fixture
def built(data_root):
    """A vocabulary learned from the fixture corpus."""
    selection = [(TINY_DATASET, None)]
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(tokenizer_id, selection, 512)
    return tokenizer_id


def stages(*datasets) -> list[pack.StageSelection]:
    return [(None, list(group)) for group in datasets]


# ── the vocabulary ──────────────────────────────────────────────────────────


def test_an_unnamed_build_is_keyed_by_its_inputs(data_root):
    """Two identical runs must share a vocabulary. A fresh id per run would
    re-learn the merges and re-pack the corpus every time, which is the cost
    skipping tokenization existed to avoid."""
    first = vocab.auto_id([(TINY_DATASET, None)], 512)
    again = vocab.auto_id([(TINY_DATASET, None)], 512)
    capped = vocab.auto_id([(TINY_DATASET, 1)], 512)
    larger = vocab.auto_id([(TINY_DATASET, None)], 1024)

    assert first == again
    assert first != capped
    assert first != larger


def test_a_build_records_what_it_was_made_from(built):
    recorded = vocab.info(built)

    assert recorded["datasets"] == [[TINY_DATASET, None]]
    assert recorded["max_vocab"] == 512
    assert recorded["vocab_size"] > 0
    assert recorded["merges"] > 0
    assert recorded["fingerprint"] == vocab.fingerprint(built)


def test_a_build_is_reused_only_for_the_request_it_was_built_for(built):
    assert vocab.matches(built, [(TINY_DATASET, None)], 512)
    assert not vocab.matches(built, [(TINY_DATASET, 1)], 512)
    assert not vocab.matches(built, [(TINY_DATASET, None)], 1024)


def test_a_vocabulary_with_no_sidecar_is_not_a_build(built):
    """An interrupted build leaves a directory. Existence is not evidence."""
    storage.tokenizer_meta_path(built).unlink()

    assert vocab.info(built) is None
    assert not vocab.matches(built, [(TINY_DATASET, None)], 512)


# ── the corpus ──────────────────────────────────────────────────────────────


def test_a_corpus_is_keyed_by_the_vocabulary_as_well_as_the_text():
    """The same bytes decode to different text under a different vocabulary, so
    two vocabularies can never share a packed corpus."""
    one = stages([(TINY_DATASET, None)])

    assert pack.corpus_key("aaaaaaaaaaaa", one) != pack.corpus_key("bbbbbbbbbbbb", one)


def test_a_corpus_key_follows_the_selection_and_its_order():
    fingerprint = "aaaaaaaaaaaa"
    both = stages([("A", None), ("B", None)])

    assert pack.corpus_key(fingerprint, both) == pack.corpus_key(fingerprint, both)
    assert pack.corpus_key(fingerprint, both) != pack.corpus_key(
        fingerprint, stages([("B", None), ("A", None)])
    )
    assert pack.corpus_key(fingerprint, both) != pack.corpus_key(
        fingerprint, stages([("A", 1), ("B", None)])
    )
    # Two stages over the same corpora is a curriculum, not the flat run.
    assert pack.corpus_key(fingerprint, both) != pack.corpus_key(
        fingerprint, stages([("A", None)], [("B", None)])
    )


def test_epochs_and_stage_names_do_not_key_a_corpus_apart(data_root):
    """Neither changes a token. Re-packing a byte-identical corpus because a
    stage was renamed is the mistake the key exists to avoid."""
    flat = TrainConfig(model_name="x", datasets=[TINY_DATASET], epochs=1)
    longer = TrainConfig(
        model_name="x",
        curriculum=[CurriculumStage(name="renamed", datasets=[TINY_DATASET], epochs=9)],
    )

    assert pack.corpus_key("aaaaaaaaaaaa", pack.resolve(flat.stages())) == pack.corpus_key(
        "aaaaaaaaaaaa", pack.resolve(longer.stages())
    )


def test_an_empty_selection_resolves_before_it_is_keyed(data_root):
    """"All the corpora" is a different corpus once another is fetched, so it is
    expanded at build time rather than stored as a wildcard."""
    resolved = pack.resolve(TrainConfig(model_name="x").stages())

    assert resolved == [(None, [(TINY_DATASET, None)])]


def test_a_packed_corpus_is_reused_only_for_its_own_request(built, data_root):
    from llm.corpus import sources

    selection = stages([(TINY_DATASET, None)])
    fingerprint = vocab.fingerprint(built)
    key = pack.corpus_key(fingerprint, selection)

    pack.pack(
        sources.collect([TINY_DATASET]),
        vocab.load(built),
        key=key,
        stages=[{"name": None, "epochs": 1, "datasets": [[TINY_DATASET, None]]}],
        tokenizer_id=built,
        fingerprint=fingerprint,
    )

    assert pack.is_built(key, selection, fingerprint)
    # A different vocabulary, or a different selection, is a different corpus —
    # even though the files under `key` are exactly where they were.
    assert not pack.is_built(key, selection, "0000000000000000")
    assert not pack.is_built(key, stages([(TINY_DATASET, 1)]), fingerprint)
    assert not pack.is_built("never-packed", selection, fingerprint)


# ── the pipelines ───────────────────────────────────────────────────────────


def test_a_tokenize_run_leaves_a_vocabulary_a_training_run_can_name(data_root, trained_id):
    from llm.config import TokenizerConfig
    from llm.orchestrate import tokenize as tokenize_pipeline

    tokenizer_id = str(uuid.uuid4())
    result = tokenize_pipeline.run(
        TokenizerConfig(tokenizer_id=tokenizer_id, max_vocab=512, datasets=[TINY_DATASET]),
        on_progress=None,
    )

    assert result["vocab_size"] > 0
    assert vocab.info(tokenizer_id) is not None

    trained = train_pipeline.run(
        TrainConfig(
            model_name=trained_id,
            tokenizer_id=tokenizer_id,
            datasets=[TINY_DATASET],
            dimensions=16,
            heads=2,
            layers=1,
            ctx_window=16,
            batch_size=2,
            epochs=1,
        ),
        on_progress=None,
    )

    assert trained["tokenizer_id"] == tokenizer_id
    # The copy beside the weights is the named build itself, byte for byte —
    # nothing re-learned a vocabulary the run was handed.
    assert (
        storage.tokenizer_file(storage.model_dir(trained_id)).read_bytes()
        == storage.tokenizer_file(storage.tokenizer_build_dir(tokenizer_id)).read_bytes()
    )


def test_a_run_naming_a_vocabulary_that_was_never_built_says_so(data_root, trained_id):
    with pytest.raises(FileNotFoundError, match="No built vocabulary"):
        train_pipeline.run(
            TrainConfig(model_name=trained_id, tokenizer_id="not-a-build"), on_progress=None
        )


# ── what a reused corpus reports ────────────────────────────────────────────


def test_a_reused_corpus_still_reports_which_markers_it_contained(data_root, trained_id):
    """`special_ids_seen` is what a later SFT warm-starts from. A reused corpus
    used to report the string "reused" instead, so `sft._base_special_ids`
    returned None and the fine-tune reset nothing — silently."""
    from llm.orchestrate import sft

    cfg = TrainConfig(
        model_name=trained_id,
        datasets=[TINY_DATASET],
        max_vocab=512,
        dimensions=16,
        heads=2,
        layers=1,
        ctx_window=16,
        batch_size=2,
        epochs=1,
    )
    first = train_pipeline.run(cfg, on_progress=None)
    assert first["dataset"]["special_ids_seen"] is not None

    second = train_pipeline.run(cfg.model_copy(update={"model_name": trained_id}), on_progress=None)

    assert second["dataset"]["reused"] is True
    assert second["dataset"]["special_ids_seen"] == first["dataset"]["special_ids_seen"]
    assert sft._base_special_ids(trained_id) == first["dataset"]["special_ids_seen"]
