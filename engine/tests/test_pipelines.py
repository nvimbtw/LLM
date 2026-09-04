"""The two pipelines the dataflow trace does not reach: SFT and eval.

Both run for real against the tiny fixture. The point is that the step sequence
and the report shape are pinned before Phase 3 rewrites the code around them.
"""

import pytest

from llm.chat.template import Reasoning
from llm.config import EvalConfig, SftConfig
from llm.corpus import sft as sft_corpus
from llm.orchestrate import evaluate as eval_pipeline
from llm.orchestrate import train as train_pipeline
from llm.orchestrate import sft as sft_pipeline
from tests.fixtures.tiny_env import TINY_CHAT_ID, TINY_DATASET_ID, TINY_MODEL

SUITES = ["language", "markup", "assistant"]


@pytest.fixture
def sft_config():
    return SftConfig(
        model_name="tuned",
        base_model=TINY_MODEL,
        datasets=[TINY_CHAT_ID],
        epochs=1,
        batch_size=2,
        lr=0.0005,
    )


def test_sft_mask_selects_a_strict_subset_of_the_corpus(tokenizer, data_client):
    """Loss is computed on assistant replies only — never the whole window."""
    selection = sft_corpus.resolve(data_client, [TINY_CHAT_ID])
    stats = sft_corpus.build(data_client, tokenizer, selection, Reasoning.KEEP)

    assert stats.conversations > 0
    assert stats.total_tokens > 0
    assert 0 < stats.trainable_tokens < stats.total_tokens
    assert 0 < stats.trainable_fraction < 1


def test_rendering_a_batch_renders_each_conversation_the_same(tokenizer, data_client):
    """The build hands every conversation's segments to the tokenizer in one call
    to use all six cores. The batch is a scheduling detail — and it has to stay
    one, because the dataset key says nothing about how the renderer ran: a batch
    that tokenized differently would write a different corpus under the same key,
    and the mask would move with it, silently training on the wrong positions.
    """
    from llm.chat import template

    selection = sft_corpus.resolve(data_client, [TINY_CHAT_ID])
    conversations = [
        (raw.get("system"), sft_corpus._turns(raw))
        for raw in sft_corpus.read_conversations(data_client, selection)
    ]
    conversations = [(system, turns) for system, turns in conversations if turns]
    assert len(conversations) > 1

    one_at_a_time = [
        template.build_training_example(system, turns, tokenizer)
        for system, turns in conversations
    ]
    batched = template.encode_examples(
        [template.training_segments(system, turns) for system, turns in conversations],
        tokenizer,
    )

    assert batched == one_at_a_time


def test_sft_corpus_and_mask_are_aligned(tokenizer, data_client):
    selection = sft_corpus.resolve(data_client, [TINY_CHAT_ID])
    stats = sft_corpus.build(data_client, tokenizer, selection, Reasoning.KEEP)
    corpus, mask = sft_corpus.load(sft_corpus.dataset_key(selection, Reasoning.KEEP))

    assert len(corpus) == len(mask)
    assert int(mask.sum()) == stats.trainable_tokens


def test_sft_run_inherits_the_base_architecture(sft_config):
    """A fine-tune loads the base weights wholesale, so its shape is the base's."""
    from llm import storage
    from llm.model import checkpoint

    sft_pipeline.run(sft_config, on_progress=None)

    _, base_arch = checkpoint.load(storage.weights_path(TINY_MODEL), "cpu")
    _, tuned_arch = checkpoint.load(storage.weights_path("tuned"), "cpu")
    assert tuned_arch == base_arch


def test_sft_run_reports_its_own_architecture(sft_config):
    """The base's shape has to reach the API somewhere, since SftConfig never
    carries it — this is that somewhere, and it must agree with the checkpoint's
    own arch dict rather than a copy that could drift from it."""
    from llm import storage
    from llm.model import checkpoint
    from llm.orchestrate.train import ARCHITECTURE_SUMMARY_FIELDS

    result = sft_pipeline.run(sft_config, on_progress=None)

    _, base_arch = checkpoint.load(storage.weights_path(TINY_MODEL), "cpu")
    assert result["architecture"] == {name: base_arch[name] for name in ARCHITECTURE_SUMMARY_FIELDS}


def test_sft_run_changes_the_weights(sft_config):
    from llm import storage
    from llm.model import checkpoint

    sft_pipeline.run(sft_config, on_progress=None)
    base, _ = checkpoint.load(storage.weights_path(TINY_MODEL), "cpu")
    tuned, _ = checkpoint.load(storage.weights_path("tuned"), "cpu")
    assert not _identical(base, tuned)


def test_sft_reuses_the_base_tokenizer(sft_config):
    """Retraining BPE would renumber the vocabulary and make the tune gibberish."""
    from llm import storage

    sft_pipeline.run(sft_config, on_progress=None)
    base = (storage.model_dir(TINY_MODEL) / "tokenizer.json").read_bytes()
    tuned = (storage.model_dir("tuned") / "tokenizer.json").read_bytes()
    assert base == tuned


def test_sft_requires_an_existing_base(sft_config):
    with pytest.raises(FileNotFoundError, match="pretrain one first"):
        sft_pipeline.run(sft_config.model_copy(update={"base_model": "nope"}), on_progress=None)


def test_a_pretrain_over_a_chat_dataset_says_what_kind_of_run_it_should_be(data_client):
    """Naming `base_model` is the *only* thing that makes a run a fine-tune, so a
    fine-tune that loses that field on the wire arrives as a pretrain over a chat
    dataset — and goes looking for prose in conversations.

    That is how it reached a user: a fine-tune of two chat sets queued happily,
    ran as a pretrain, and failed minutes later. The message has to name the
    remedy, because the state it describes is not "missing data" but "wrong kind
    of run".
    """
    from llm.corpus import vocab

    selection = vocab.resolve(data_client, [TINY_CHAT_ID])

    with pytest.raises(ValueError, match="fine-tune"):
        vocab.build(data_client, "unused", selection, 512)


@pytest.mark.parametrize("suite", SUITES)
def test_eval_report_carries_a_versioned_score(suite):
    """Every suite scores 0-100 and stamps the version that produced it — two
    scores are only comparable at the same version."""
    report = _evaluate(suite)

    assert report["suite"] == suite
    assert isinstance(report["score"], dict)
    assert report["score"]["version"] >= 1
    assert 0.0 <= report["score"]["score"] <= 100.0


@pytest.mark.parametrize("suite", SUITES)
def test_eval_writes_the_report_beside_the_weights(suite):
    """A score travels with the model it describes."""
    import json

    from llm import storage

    _evaluate(suite)
    path = storage.model_dir(TINY_MODEL) / f"benchmark-{suite}.json"
    assert json.loads(path.read_text())["suite"] == suite


def test_the_three_suites_are_scored_independently():
    """They share a range and nothing else — never compared, never averaged."""
    scores = {suite: _evaluate(suite)["score"] for suite in SUITES}

    assert set(scores["language"]) != set(scores["markup"])
    assert set(scores["markup"]) != set(scores["assistant"])
    # Each versions on its own, so one suite's rescoring cannot invalidate another.
    assert all("version" in s for s in scores.values())


@pytest.mark.parametrize("suite", SUITES)
def test_every_report_says_which_model_it_describes(suite):
    """A report is read long after the run, so it carries the model rather than
    leaving a reader to join it back to one."""
    model = _evaluate(suite)["model"]

    assert model["id"] == TINY_MODEL
    # Filled in by whoever owns the row; the pipeline cannot resolve a name.
    assert model["name"] is None
    assert (model["layers"], model["dimensions"], model["heads"]) == (2, 32, 4)
    assert model["parameters"] > 0


def test_eval_rejects_an_unknown_suite():
    with pytest.raises(ValueError, match="unknown suite"):
        eval_pipeline.run(EvalConfig(model_name=TINY_MODEL, suite="nonsense"))


def test_eval_needs_weights():
    with pytest.raises(FileNotFoundError):
        eval_pipeline.run(EvalConfig(model_name="no-such-model", suite="language"))


def test_sft_diagnostics_split_the_loss_by_mask_position_and_conversation(sft_config):
    """The three readings a fine-tune needs that a pretrain run does not, all
    taken off the one recording forward `Collector.observe` already ran."""
    from llm.train.diagnostics import POSITION_BUCKETS

    sampled = _diagnostics(sft_config)
    assert sampled

    split = [d for d in sampled if d.get("loss_masked") is not None]
    assert split, "no diagnosed window carried a trainable token"
    for diag in split:
        # The prompt region is not optimized, so the two are different numbers —
        # equal would mean the mask never reached the split.
        assert diag["loss_masked"] != diag["loss_unmasked"]
        assert len(diag["loss_by_position"]) == POSITION_BUCKETS
        assert len(diag["conversation_loss"]) == len(diag["conversation_start"])
        assert diag["conversation_loss"] == sorted(diag["conversation_loss"], reverse=True)
        # Every conversation is a real offset into the corpus; -1 is the
        # separator between two of them and belongs to neither.
        assert all(start >= 0 for start in diag["conversation_start"])


def test_a_conversation_is_named_by_its_offset_in_the_corpus(sft_config, data_client):
    """Offsets, not row indices. The epoch is reshuffled every pass, so a row
    index would name a different conversation each epoch and a ranking built from
    it would be an average over the whole dataset.
    """
    sampled = _diagnostics(sft_config.model_copy(update={"epochs": 2}))

    key = sft_corpus.dataset_key(
        sft_corpus.resolve(data_client, sft_config.datasets), Reasoning.KEEP
    )
    _, documents, _ = sft_corpus.load_segments(key)
    starts = set(documents.starts.tolist())

    seen = {start for d in sampled for start in d.get("conversation_start", [])}
    assert seen and seen <= starts


def test_the_base_model_probes_are_absent_when_they_cannot_be_reached(sft_config):
    """The fixture base has no meta.json, so no corpus_key names a pretrain
    corpus and the run never asked for the KL anchor. Both stay out of the frame
    rather than arriving as a zero that reads as a real measurement."""
    for diag in _diagnostics(sft_config):
        assert "pretrain_loss" not in diag
        assert "base_kl" not in diag


def test_the_pretrain_probe_comes_from_the_base_model_s_own_corpus(
    sft_config, monkeypatch
):
    """The reachability this whole diagnostic rests on: a pretrain run records
    `corpus_key`, the key names a directory, and the fine-tune reads the corpus
    the base was built from without asking anything outside the engine."""
    import numpy as np

    from llm import storage
    from llm.corpus import pack
    from llm.train.diagnostics import DIAGNOSTIC_ROWS

    key = "probe-fixture"
    storage.ensure_dir(storage.pretrain_dataset_dir(key))
    # Ids inside the base model's vocabulary, and long enough for both rows.
    tokens = (np.arange(4096, dtype=np.int64) % 300).astype(pack.CORPUS_DTYPE)
    storage.pretrain_corpus_path(key).write_bytes(tokens.tobytes())
    monkeypatch.setattr(train_pipeline, "run_record", lambda _: {"corpus_key": key})

    probe = sft_pipeline._pretrain_probe(sft_config.base_model, ctx_window=128, vocab=320)

    assert probe is not None
    assert probe.shape == (DIAGNOSTIC_ROWS, 129)
    # Spread across the corpus, not taken twice off the front: `corpus.bin` is
    # laid out in dataset order and a probe from the head measures one dataset.
    assert not np.array_equal(probe[0], probe[-1])

    # And it reaches the frame, which is the only reason to have built it.
    sampled = _diagnostics(sft_config)
    assert sampled and all(d["pretrain_loss"] > 0.0 for d in sampled)


def test_the_pretrain_probe_refuses_a_corpus_from_another_vocabulary(
    sft_config, monkeypatch
):
    """An id past the embedding table would take the run down thousands of steps
    in, on a forward nothing is waiting on."""
    import numpy as np

    from llm import storage
    from llm.corpus import pack

    key = "probe-wrong-vocab"
    storage.ensure_dir(storage.pretrain_dataset_dir(key))
    tokens = np.full(4096, 511, dtype=pack.CORPUS_DTYPE)
    storage.pretrain_corpus_path(key).write_bytes(tokens.tobytes())
    monkeypatch.setattr(train_pipeline, "run_record", lambda _: {"corpus_key": key})

    assert sft_pipeline._pretrain_probe(sft_config.base_model, 128, vocab=320) is None


def test_the_kl_anchor_is_reported_only_when_the_run_asks_for_it(sft_config):
    """It keeps a second copy of the base model resident, so it is opt-in — and
    when it is on, it is measured against held-out chat the run never trains on."""
    assert all("base_kl" not in d for d in _diagnostics(sft_config))

    anchored = _diagnostics(sft_config.model_copy(update={"measure_base_kl": True}))
    assert anchored and all(d["base_kl"] >= 0.0 for d in anchored)


def test_an_unmeasured_diagnostic_is_absent_from_the_frame(sft_config):
    """The contract the panel depends on: a pretrain run's frames carry none of
    the fine-tuning keys at all, so the panel can tell the two modes apart."""
    from llm.train.diagnostics import Diagnostics

    frame = Diagnostics(step=7).as_dict()

    assert frame["step"] == 7
    for key in (
        "pretrain_loss",
        "loss_masked",
        "loss_unmasked",
        "loss_by_position",
        "conversation_loss",
        "conversation_start",
        "base_kl",
    ):
        assert key not in frame


def _diagnostics(cfg) -> list[dict]:
    """Every diagnostics sample an SFT run emits."""
    from llm.train.progress import Progress

    frames: list = []
    sft_pipeline.run(cfg, on_progress=frames.append)
    return [f.diag for f in frames if isinstance(f, Progress) and f.diag]


def _evaluate(suite: str) -> dict:
    return eval_pipeline.run(
        EvalConfig(model_name=TINY_MODEL, suite=suite, max_tokens=16, seed=5)
    )


def _identical(a, b) -> bool:
    import torch

    return all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values())
    )
