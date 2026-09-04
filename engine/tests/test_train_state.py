"""Checkpointing as the training core sees it: what a state file holds, when
one is written, and what resuming from one is worth.

The database half of this lives in `test_checkpoints.py`. Nothing here knows
there is a database — a checkpoint is a directory and a frame, and that is the
whole of `llm/`'s side of it.
"""

import json
import random

import numpy as np
import pytest
import torch

from llm import storage
from llm.config import TrainConfig
from llm.corpus import bpe
from llm.corpus.spans import DocSpan
from llm.model import checkpoint
from llm.model.checkpoint import TrainState
from llm.model.transformer import Transformer
from llm.orchestrate import train as train_pipeline
from llm.train import loop
from llm.train.plan import DataPlan
from llm.train.progress import Checkpoint, checkpoint_every, report_every

#: Small enough that a few hundred steps run in seconds, large enough that the
#: corpus yields the ~127 batches an epoch that the cadence arithmetic below is
#: stated in terms of.
ARCH = {"dimensions": 16, "heads": 2, "layers": 1, "ff_multiplier": 2, "ctx_window": 16}
VOCAB = 61
BATCHES_PER_EPOCH = 128


def tiny_corpus() -> np.ndarray:
    return np.arange(4096, dtype=np.uint16) % VOCAB


def tiny_plan(corpus) -> DataPlan:
    return DataPlan(
        documents=[DocSpan("x", 0, len(corpus))], eot_id=0, shuffle=True, seed=7
    )


def tiny_config(**overrides) -> TrainConfig:
    return TrainConfig(
        model_name="unused", batch_size=2, warmup_steps=5, **{**ARCH, **overrides}
    )


def tiny_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(vocab=VOCAB, **ARCH)


class RecordingWriter:
    """A checkpoint writer that writes nothing.

    The cadence tests are about *when* the loop asks, which is a different
    question from what a directory is — and paying for a directory per
    checkpoint would make them slow enough that nobody runs them.
    """

    def __init__(self):
        self.taken: list[Checkpoint] = []

    def __call__(self, model, optimizer, *, step, total_steps, epoch, stage,
                 stage_name, loss) -> Checkpoint:
        frame = Checkpoint(
            model_id=f"id-{len(self.taken)}",
            index=len(self.taken) + 1,
            step=step,
            total_steps=total_steps,
            epoch=epoch,
            stage=stage,
            stage_name=stage_name,
            path=f"/nowhere/{step}.pt",
            loss=loss,
        )
        self.taken.append(frame)
        return frame


def train(epochs: int, *, writer=None, resume=None, frames=None, **overrides):
    """One short run over the synthetic corpus."""
    corpus = tiny_corpus()
    plan = tiny_plan(corpus)
    cfg = tiny_config(epochs=epochs, **overrides)
    model = tiny_model()
    summary = loop.run(
        model, corpus, plan, cfg, "cpu",
        on_progress=(frames.append if frames is not None else None),
        on_checkpoint=writer,
        resume=resume,
    )
    return model, summary


# ── The state file ──────────────────────────────────────────────────────────


def test_an_adamw_state_and_the_rng_survive_the_weights_only_loader(tmp_path):
    """`torch.load` stays `weights_only=True`, so every entry has to be
    something the restricted unpickler can build. Numpy's generator state is the
    one that is not, and it is converted rather than the unpickler widened."""
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    model(torch.zeros(1, 4, dtype=torch.int64))[0].sum().backward()
    optimizer.step()

    saved = checkpoint.save_state(
        tmp_path / "train_state.pt",
        model=model,
        optimizer=optimizer,
        step=17,
        epoch=2,
        stage=0,
        total_steps=100,
        rng=checkpoint.rng_state(),
        extra={"corpus_key": "abc", "loss": 1.5, "config": {"epochs": 3}},
    )
    state = checkpoint.load_state(saved)

    assert state.step == 17 and state.epoch == 2 and state.total_steps == 100
    assert state.extra["corpus_key"] == "abc"
    assert state.arch["dimensions"] == ARCH["dimensions"]

    moments = state.optimizer["state"]
    assert moments, "AdamW keeps a moment pair per parameter and none survived"
    for entry in moments.values():
        assert {"exp_avg", "exp_avg_sq"} <= set(entry)
        assert torch.is_tensor(entry["exp_avg"])
    assert state.optimizer["param_groups"][0]["betas"] == (0.9, 0.95)

    # The loaded state has to be usable, not merely present: a state dict that
    # round-trips but will not load back into an optimizer is the same loss.
    reloaded = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    reloaded.load_state_dict(state.optimizer)


def test_the_generators_come_back_where_they_were(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    saved = checkpoint.save_state(
        tmp_path / "state.pt", model=model, optimizer=optimizer, step=1, epoch=1,
        stage=0, total_steps=2, rng=checkpoint.rng_state(), extra={},
    )
    expected = (torch.randn(3).tolist(), np.random.rand(3).tolist(), random.random())

    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    checkpoint.restore_rng(checkpoint.load_state(saved).rng)

    assert (torch.randn(3).tolist(), np.random.rand(3).tolist(), random.random()) == expected


def test_a_weights_file_reads_as_a_state_with_cold_moments(tmp_path):
    """`resume_from` names a model id, and a *finished* model is one — so the
    loader has to read a plain v1 weights file and say, by leaving the optimizer
    absent, that this continuation starts AdamW from nothing."""
    saved = checkpoint.save(tiny_model(), tmp_path / "model.pt")
    state = checkpoint.load_state(saved)

    assert state.optimizer is None
    assert state.step == 0 and state.rng == {}
    assert state.arch["heads"] == ARCH["heads"]


def test_a_state_file_still_loads_as_plain_weights(tmp_path):
    """The two files are one shape on purpose: inference must not have to know
    which of them it was handed."""
    model = tiny_model()
    saved = checkpoint.save_state(
        tmp_path / "train_state.pt", model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        step=3, epoch=1, stage=0, total_steps=9, rng={}, extra={},
    )
    loaded, arch = checkpoint.load(saved)

    assert arch["vocab"] == VOCAB
    assert torch.equal(loaded.state_dict()["embedding.weight"],
                       model.state_dict()["embedding.weight"])


# ── The cadence ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("epochs", [1, 2, 4])
def test_a_checkpoint_lands_on_a_reporting_step_and_never_twice(epochs):
    writer = RecordingWriter()
    frames = []
    train(epochs, writer=writer, frames=frames)

    total = BATCHES_PER_EPOCH * epochs
    steps = [frame.step for frame in writer.taken]
    assert steps == sorted(set(steps)), "one step, one checkpoint"
    # The interval is snapped up to a multiple of `report_every` so that it
    # never invents a step of its own; the epoch and final rules land on the
    # frame the epoch publishes for itself. Either way there is a frame there.
    reported = {frame.step for frame in frames if not isinstance(frame, Checkpoint)}
    assert set(steps) <= reported
    assert all(
        step % report_every(total) == 0 or step % BATCHES_PER_EPOCH == 0
        for step in steps
    )
    assert steps[-1] == total, "the last step of a run always checkpoints"
    assert [frame.index for frame in writer.taken] == list(range(1, len(steps) + 1))
    assert all(frame.total_steps == total for frame in writer.taken)


@pytest.mark.parametrize("epochs", [1, 2, 4])
def test_every_epoch_ends_with_one(epochs):
    """The per-epoch checkpoint predates the interval and is the point a person
    reaches back to by name. It survived the move to whole-model checkpoints."""
    writer = RecordingWriter()
    train(epochs, writer=writer)

    ends = {BATCHES_PER_EPOCH * (n + 1) for n in range(epochs)}
    assert ends <= {frame.step for frame in writer.taken}
    assert {frame.epoch for frame in writer.taken} == set(range(1, epochs + 1))


def test_an_epoch_ending_on_the_interval_is_written_once():
    """`checkpoint_every` snaps onto a reporting step, so the two rules
    coinciding is the ordinary case rather than a corner one — and twice would
    be the same weights under two ids."""
    writer = RecordingWriter()
    train(2, writer=writer, checkpoint_target=2)

    assert checkpoint_every(2 * BATCHES_PER_EPOCH, 2, 2) == BATCHES_PER_EPOCH
    assert [frame.step for frame in writer.taken] == [BATCHES_PER_EPOCH,
                                                      2 * BATCHES_PER_EPOCH]


def test_a_target_of_zero_leaves_none_at_all():
    writer = RecordingWriter()
    train(2, writer=writer, checkpoint_target=0)

    assert writer.taken == []


def test_a_run_with_no_writer_takes_none():
    """`bench.py` and every test that only wants the arithmetic pass none, and a
    run that wrote a directory anyway would make timing a step mean writing the
    model ten times."""
    frames = []
    train(1, frames=frames)

    assert all(frame.checkpoint_path is None for frame in frames)


def test_the_frame_and_the_progress_path_carry_the_same_checkpoint():
    """Both halves are read by something: the frame becomes a row, the path is
    what the run appendix has always shown."""
    frames = []
    writer = RecordingWriter()
    train(1, writer=writer, frames=frames)

    published = [f for f in frames if isinstance(f, Checkpoint)]
    assert [f.model_id for f in published] == [f.model_id for f in writer.taken]

    paths = {f.checkpoint_path for f in frames if not isinstance(f, Checkpoint)}
    assert {f.path for f in writer.taken} <= paths


# ── Resuming ────────────────────────────────────────────────────────────────


def _checkpoint_writer(tokenizer, cfg) -> train_pipeline.CheckpointWriter:
    return train_pipeline.CheckpointWriter(
        tokenizer=tokenizer,
        config=cfg.model_dump(),
        provenance={"corpus_key": "test-corpus", "vocab": VOCAB},
        state_extra={"corpus_key": "test-corpus"},
    )


def _written(tokenizer, cfg) -> list[Checkpoint]:
    """One run whose checkpoints are real directories, and the frames it
    published for them."""
    corpus = tiny_corpus()
    frames = []
    loop.run(
        tiny_model(), corpus, tiny_plan(corpus), cfg, "cpu",
        on_progress=frames.append,
        on_checkpoint=_checkpoint_writer(tokenizer, cfg),
    )
    return [frame for frame in frames if isinstance(frame, Checkpoint)]


def test_a_checkpoint_directory_is_a_model_directory(tokenizer):
    """Chatting with a checkpoint, benchmarking one and fine-tuning from one are
    all supposed to be the code that already loads a model. That only holds if
    the directory is the same shape."""
    cfg = tiny_config(epochs=1, checkpoint_target=1)
    taken = _written(tokenizer, cfg)

    directory = storage.model_dir(taken[-1].model_id)
    assert {"model.pt", "tokenizer.json", "meta.json", "train_state.pt"} <= {
        path.name for path in directory.iterdir()
    }
    assert taken[-1].path == str(storage.train_state_path(directory.name))

    written = json.loads(storage.meta_path(directory.name).read_text())
    assert written["result"]["model_name"] == directory.name
    assert written["result"]["checkpoint"]["step"] == taken[-1].step
    assert written["config"]["epochs"] == 1

    # Loadable by the two paths every other consumer already goes through.
    loaded, arch = checkpoint.load(storage.weights_path(directory.name))
    assert arch["vocab"] == VOCAB and loaded is not None
    assert bpe.load(directory) is not None


def test_resuming_arrives_where_the_uninterrupted_run_did(tokenizer):
    """The whole promise of a deterministic continuation, and what tells a
    restored optimizer from a merely reloaded one: cold moments would take a
    visibly different next hundred steps."""
    cfg = tiny_config(epochs=2, checkpoint_target=2)
    corpus = tiny_corpus()
    plan = tiny_plan(corpus)

    straight = tiny_model()
    frames = []
    loop.run(straight, corpus, plan, cfg, "cpu", on_progress=frames.append,
             on_checkpoint=_checkpoint_writer(tokenizer, cfg))
    expected = {name: value.clone() for name, value in straight.state_dict().items()}

    halfway = next(
        frame for frame in frames
        if isinstance(frame, Checkpoint) and frame.step == BATCHES_PER_EPOCH
    )
    state = checkpoint.load_state(storage.train_state_path(halfway.model_id))
    assert state.step == BATCHES_PER_EPOCH and state.optimizer is not None

    picked_up = Transformer(**state.arch)
    picked_up.load_state_dict(state.state_dict)
    loop.run(picked_up, corpus, plan, cfg, "cpu", on_progress=None, resume=state)

    for name, tensor in picked_up.state_dict().items():
        assert torch.allclose(tensor, expected[name], rtol=0, atol=1e-6), name
    # And it did train the second half rather than stopping where it started.
    assert not torch.allclose(
        picked_up.state_dict()["embedding.weight"],
        state.state_dict["embedding.weight"],
    )


def test_the_fast_forward_skips_the_data_rather_than_walking_it(monkeypatch):
    """`EpochStream` is addressable precisely so that this is arithmetic. A
    resumed run that re-gathered everything it had already trained would pay the
    whole first half again for nothing."""
    from llm.train import plan as plan_mod

    gathered = []
    original = plan_mod.EpochStream.gather

    def counting(self, lo, hi, tokens, weights=None):
        gathered.append(lo)
        return original(self, lo, hi, tokens, weights)

    monkeypatch.setattr(plan_mod.EpochStream, "gather", counting)

    state = TrainState(
        arch={}, state_dict={}, step=BATCHES_PER_EPOCH + 100,
        total_steps=2 * BATCHES_PER_EPOCH, epoch=2,
    )
    train(2, resume=state)

    assert len(gathered) == 2 * BATCHES_PER_EPOCH - state.step
    assert min(gathered) > 0, "the skipped head of the epoch was gathered anyway"


# ── Which continuation it is, and what is refused ───────────────────────────


def state_of(**overrides) -> TrainState:
    arch = {"vocab": VOCAB, "rope_base": 10000.0, "learned_temperature": False, **ARCH}
    return TrainState(
        arch=arch,
        state_dict={},
        step=overrides.pop("step", 50),
        total_steps=overrides.pop("total_steps", 2 * BATCHES_PER_EPOCH),
        extra=overrides.pop("extra", {"corpus_key": "test-corpus"}),
        **overrides,
    )


def test_a_request_that_restates_the_architecture_differently_is_refused():
    cfg = tiny_config(epochs=1, resume_from="somewhere", dimensions=32, heads=4)

    with pytest.raises(ValueError, match="architecture comes from the checkpoint"):
        train_pipeline.resumed_architecture(cfg, state_of(), VOCAB)


def test_a_request_that_restates_nothing_takes_the_checkpoints_shape():
    """Pydantic fills every architecture field with a default, so comparing all
    of them would refuse every resume that did not restate the model exactly."""
    cfg = TrainConfig(model_name="x", resume_from="somewhere")

    assert cfg.dimensions != ARCH["dimensions"]
    assert train_pipeline.resumed_architecture(cfg, state_of(), VOCAB) == state_of().arch


def test_a_vocabulary_the_checkpoint_was_not_trained_on_is_refused():
    cfg = TrainConfig(model_name="x", resume_from="somewhere")

    with pytest.raises(ValueError, match="vocabulary"):
        train_pipeline.resumed_architecture(cfg, state_of(), VOCAB + 1)


def test_a_stated_tokenizer_that_disagrees_is_refused():
    cfg = TrainConfig(model_name="x", resume_from="somewhere", tokenizer_id="other-tok")

    with pytest.raises(ValueError, match="tokenizer comes from the checkpoint"):
        train_pipeline.resumed_tokenizer(
            cfg, state_of(extra={"corpus_key": "test-corpus", "tokenizer_id": "ckpt-tok"})
        )


def test_a_stated_tokenizer_that_agrees_is_allowed():
    cfg = TrainConfig(model_name="x", resume_from="somewhere", tokenizer_id="ckpt-tok")

    assert (
        train_pipeline.resumed_tokenizer(
            cfg, state_of(extra={"corpus_key": "test-corpus", "tokenizer_id": "ckpt-tok"})
        )
        == "ckpt-tok"
    )


def test_a_request_that_restates_nothing_takes_the_checkpoints_tokenizer():
    cfg = TrainConfig(model_name="x", resume_from="somewhere")

    assert (
        train_pipeline.resumed_tokenizer(
            cfg, state_of(extra={"corpus_key": "test-corpus", "tokenizer_id": "ckpt-tok"})
        )
        == "ckpt-tok"
    )


def test_resuming_a_finished_model_falls_back_to_its_meta_json():
    """A finished model's `train_state.pt` has no `extra` at all — its tokenizer
    lives in meta.json instead, the same provenance lookup SFT uses for a base
    model's (`run_record`)."""
    model_id = "finished-model"
    storage.ensure_dir(storage.model_dir(model_id))
    storage.meta_path(model_id).write_text(
        json.dumps({"config": {}, "result": {"tokenizer_id": "base-tok"}})
    )
    cfg = TrainConfig(model_name="x", resume_from=model_id)

    assert train_pipeline.resumed_tokenizer(cfg, state_of(extra={})) == "base-tok"


def test_resuming_with_no_recorded_tokenizer_anywhere_is_refused():
    cfg = TrainConfig(model_name="x", resume_from="nowhere-recorded")

    with pytest.raises(ValueError, match="recorded no tokenizer_id"):
        train_pipeline.resumed_tokenizer(cfg, state_of(extra={}))


def test_resuming_from_nothing_says_where_it_looked():
    with pytest.raises(FileNotFoundError, match="resume_from"):
        train_pipeline.resume_state(str(random.random()))


def test_the_same_corpus_and_schedule_is_a_resume_and_anything_else_is_not():
    corpus = tiny_corpus()
    plan = tiny_plan(corpus)
    cfg = tiny_config(epochs=2, resume_from="somewhere")

    same = train_pipeline._continuation(cfg, state_of(), corpus, plan, "test-corpus")
    assert same.record["resume"]["mode"] == "resumed"
    assert same.state.step == 50

    other_corpus = train_pipeline._continuation(cfg, state_of(), corpus, plan, "elsewhere")
    assert other_corpus.record["resume"]["mode"] == "continued"
    # The moments carry over; the position and the generators do not, because
    # they name a schedule and an order this run does not have.
    assert other_corpus.state.step == 0 and other_corpus.state.rng == {}

    shorter = train_pipeline._continuation(
        cfg, state_of(total_steps=99), corpus, plan, "test-corpus"
    )
    assert shorter.record["resume"]["mode"] == "continued"


def test_no_resume_records_nothing():
    corpus = tiny_corpus()
    continuation = train_pipeline._continuation(
        tiny_config(epochs=1), None, corpus, tiny_plan(corpus), "test-corpus"
    )

    assert continuation.state is None and continuation.record == {}
