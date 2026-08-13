"""The two pipelines the dataflow trace does not reach: SFT and eval.

Both run for real against the tiny fixture. The point is that the step sequence
and the report shape are pinned before Phase 3 rewrites the code around them.
"""

import pytest

from llm.chat.template import Reasoning
from llm.config import EvalConfig, SftConfig
from llm.corpus import sft as sft_corpus
from llm.orchestrate import evaluate as eval_pipeline
from llm.orchestrate import sft as sft_pipeline
from tests.fixtures.tiny_env import TINY_DATASET, TINY_MODEL

SUITES = ["language", "markup", "assistant"]


@pytest.fixture
def sft_config():
    return SftConfig(
        model_name="tuned",
        base_model=TINY_MODEL,
        sft_datasets=[TINY_DATASET],
        epochs=1,
        batch_size=2,
        lr=0.0005,
    )


def test_sft_mask_selects_a_strict_subset_of_the_corpus(tokenizer):
    """Loss is computed on assistant replies only — never the whole window."""
    selection = sft_corpus.resolve([TINY_DATASET])
    stats = sft_corpus.build(tokenizer, selection, Reasoning.KEEP)

    assert stats.conversations > 0
    assert stats.total_tokens > 0
    assert 0 < stats.trainable_tokens < stats.total_tokens
    assert 0 < stats.trainable_fraction < 1


def test_sft_corpus_and_mask_are_aligned(tokenizer):
    selection = sft_corpus.resolve([TINY_DATASET])
    stats = sft_corpus.build(tokenizer, selection, Reasoning.KEEP)
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
