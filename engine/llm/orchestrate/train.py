"""The training pipeline.

    1. resolve the vocabulary          \
    2. resolve the datasets             > skipped when the build already matches
    3. pack the corpus                 /
    4. build the model
    5. run the training loop
    6. save weights + tokenizer + meta.json

Steps 1-3 are a separate phase from 5 on purpose, mirroring the Rust
build_dataset/run_training split: tokenizing is expensive and only has to happen
when the *data* changes, not when the architecture or the learning rate does.

"Already matches" is now a question with an answer. The vocabulary is either one
a `tokenize` run built and this run named, or one derived from this run's own
selection; the corpus is keyed on that vocabulary and that selection, and its
sidecar is compared field by field. It used to be "do a corpus and a tokenizer
exist anywhere", which is why a run could train on data nobody asked for.
"""

import json
import time
import uuid
from dataclasses import dataclass, replace
from threading import Event

from tokenizers import Tokenizer

from llm import storage
from llm.config import TrainConfig
from llm.corpus import bpe, pack, remote, spans as spans_mod, vocab
from llm.model import checkpoint
from llm.model.checkpoint import TrainState
from llm.model.transformer import Transformer
from llm.train import diagnostics, loop
from llm.train.plan import DataPlan, Stage
from llm.train.progress import Checkpoint, ProgressCallback, Tokenize, log_progress

#: Architecture a `TrainConfig` can state. A resumed run takes these from the
#: checkpoint; the point of the list is which ones a *request* is then compared
#: against, because pydantic cannot tell "unset" from "set to the default" any
#: other way than `model_fields_set`.
_ARCHITECTURE = frozenset(
    {"dimensions", "heads", "layers", "ff_multiplier", "ctx_window", "rope_base",
     "learned_temperature"}
)

#: The shape fields shown wherever a model's architecture is surfaced — a
#: pretrain's live on its own `TrainConfig` under `config.*`; `sft.py` reads
#: this same tuple off the base checkpoint's `arch` to report the same five
#: for a fine-tune, which has no `TrainConfig` of its own to carry them.
ARCHITECTURE_SUMMARY_FIELDS = ("dimensions", "heads", "layers", "ff_multiplier", "ctx_window")


class CheckpointWriter:
    """What a checkpoint *directory* is — the half of checkpointing the loop
    does not own.

    A checkpoint is a model. It gets a uuid of its own and a directory laid out
    exactly like a finished model's — `model.pt`, `tokenizer.json`, `meta.json` —
    plus the `train_state.pt` a later run picks it up from. Identical in shape is
    the entire point: chatting with a checkpoint, benchmarking one, and
    fine-tuning from one all fall out of the code that already loads a model,
    and none of those paths learns what a checkpoint is.

    Used by `orchestrate/sft.py` as well as by this module. It is here rather
    than in a module of its own because the two orchestrators differ only in what
    they know to record, which is exactly what they hand the constructor.
    """

    def __init__(self, *, tokenizer, config: dict, provenance: dict, state_extra: dict):
        self.tokenizer = tokenizer
        self.config = config
        self.provenance = provenance
        self.state_extra = state_extra
        self.index = 0

    def __call__(
        self,
        model,
        optimizer,
        *,
        step: int,
        total_steps: int,
        epoch: int,
        stage: int,
        stage_name: str | None,
        loss: float | None,
    ) -> Checkpoint:
        self.index += 1
        model_id = str(uuid.uuid4())
        directory = storage.ensure_dir(storage.model_dir(model_id))

        checkpoint.save(model, storage.weights_path(model_id))
        self.tokenizer.save(str(storage.tokenizer_file(directory)))
        state = checkpoint.save_state(
            storage.train_state_path(model_id),
            model=model,
            optimizer=optimizer,
            step=step,
            epoch=epoch,
            stage=stage,
            total_steps=total_steps,
            rng=checkpoint.rng_state(),
            extra={**self.state_extra, "config": self.config, "loss": loss},
        )
        storage.meta_path(model_id).write_text(
            json.dumps(
                {
                    "config": self.config,
                    "result": {
                        "model_name": model_id,
                        **self.provenance,
                        "checkpoint": {
                            "index": self.index,
                            "step": step,
                            "total_steps": total_steps,
                            "epoch": epoch,
                            "stage": stage,
                            "loss": loss,
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return Checkpoint(
            model_id=model_id,
            index=self.index,
            step=step,
            total_steps=total_steps,
            epoch=epoch,
            stage=stage,
            stage_name=stage_name,
            path=str(state),
            loss=loss,
            # Off the model being saved rather than out of `self.config`, so it
            # cannot disagree with the `arch` in the file written two lines
            # above — and so it is right for a resumed run, whose config states
            # no architecture at all.
            architecture={name: getattr(model, name) for name in ARCHITECTURE_SUMMARY_FIELDS},
        )


def run_record(model_id: str) -> dict:
    """What the run that produced a model recorded about itself.

    Two meta.json shapes are in circulation and both are live, because two
    writers produce them. A run driven straight through the engine keeps what
    `run`/SFT's `run` wrote here: `{"config", "result"}`. A run driven by the
    backend queue has its meta *rewritten* on the way out by
    `training::write_meta`, which stores this whole result dict under the key
    `"dataset"` instead.

    Reading only the first spelling made `sft._base_special_ids` return None for
    every backend-launched run — this is that same lookup, generalized so a
    resumed pretrain can recover a checkpoint's `tokenizer_id` the same way an
    SFT run recovers a base model's.
    """
    path = storage.meta_path(model_id)
    if not path.is_file():
        return {}
    meta = json.loads(path.read_text(encoding="utf-8"))
    run = meta.get("result")
    if not isinstance(run, dict):
        run = meta.get("dataset")
    return run if isinstance(run, dict) else {}


def resume_state(model_id: str | None) -> TrainState | None:
    """The training state a run was told to carry on from, or None.

    One id covers both a finished model and one of its checkpoints: they are the
    same directory shape and the difference is only whether a `train_state.pt`
    is in it. So resuming from a finished model works, and it starts AdamW cold —
    which is `load_state` reading a plain weights file, not a branch here.
    """
    if not model_id:
        return None
    state_file = storage.train_state_path(model_id)
    if state_file.is_file():
        return checkpoint.load_state(state_file)
    weights = storage.weights_path(model_id)
    if weights.is_file():
        return checkpoint.load_state(weights)
    raise FileNotFoundError(
        f"Nothing to resume from at {storage.model_dir(model_id)} — `resume_from` "
        f"names a model id, finished or a checkpoint of one"
    )


def resumed_architecture(cfg: TrainConfig, state: TrainState, vocab_size: int) -> dict:
    """The shape a resumed run trains at, which is the checkpoint's.

    A request may restate it, and then the two have to agree. Letting the request
    win is not on the table — the weights would win regardless, silently, and the
    run would report an architecture it did not train. Refusing is the same rule
    `SftConfig`'s docstring states for a fine-tune, applied to the other way of
    starting from existing weights.

    Only fields the request actually set are compared: a resume naming nothing
    but `resume_from` is asking for the checkpoint's shape, not for the defaults
    pydantic filled in behind it.
    """
    arch = dict(state.arch)
    disagreements = [
        f"{name} {getattr(cfg, name)} against the checkpoint's {arch[name]}"
        for name in sorted(_ARCHITECTURE & cfg.model_fields_set)
        if name in arch and arch[name] != getattr(cfg, name)
    ]
    if arch["vocab"] != vocab_size:
        disagreements.append(
            f"a vocabulary of {vocab_size} tokens against the checkpoint's {arch['vocab']}"
        )
    if disagreements:
        raise ValueError(
            f"Cannot resume from {cfg.resume_from}: the architecture comes from the "
            f"checkpoint and this request asks for " + "; ".join(disagreements)
        )
    return arch


def resumed_config(cfg: TrainConfig, state: TrainState | None, vocab_size: int) -> TrainConfig:
    """The config this run actually trains under: the request, wearing the
    checkpoint's shape.

    Applied once, here, and *before* anything reads a number off it — because
    the architecture is not only what the model is built from. `ctx_window` is
    what the loop cuts its windows at and what phase 1 measures the corpus
    against, and every record a run leaves behind (`meta.json`, each
    checkpoint's config, the model row's `context`) is this config dumped. A
    run that resolved the shape at `Transformer(...)` alone trained a 1024-token
    checkpoint on 256-token windows and then wrote the request's numbers down as
    though they were what it had done.

    Only the architecture moves. Everything else about a resumed run — its
    schedule, its data, its seed — is this request's, which is what makes a
    continuation a run of its own rather than a replay.
    """
    if state is None:
        return cfg
    arch = resumed_architecture(cfg, state, vocab_size)
    return cfg.model_copy(update={name: arch[name] for name in _ARCHITECTURE & arch.keys()})


def resumed_tokenizer(cfg: TrainConfig, state: TrainState) -> str:
    """The vocabulary a resumed run trains against, which is the checkpoint's.

    The same rule as `resumed_architecture`, for the one shape fact it does not
    cover: a corpus packed under a different vocabulary than the checkpoint's
    embedding table would index into it out of bounds, or — worse, if the two
    vocabularies happen to be the same size — quietly mean different tokens by
    the same ids. Letting the request state a different one, or re-deriving one
    from this run's own dataset selection the way a fresh pretrain does, both
    lead there.

    `state.extra` carries `tokenizer_id` when the checkpoint was written by this
    pipeline's `CheckpointWriter`; resuming from a *finished* model — a plain
    weights file, no `train_state.pt` — has no `extra` at all, so this falls
    back to that model's own meta.json, the same provenance lookup
    `orchestrate/sft.py` does for a base model's tokenizer (`run_record`).
    """
    tokenizer_id = state.extra.get("tokenizer_id") or run_record(cfg.resume_from).get(
        "tokenizer_id"
    )
    if not tokenizer_id:
        raise ValueError(
            f"Cannot resume from {cfg.resume_from}: it recorded no tokenizer_id to "
            f"continue from, so the vocabulary it was trained under cannot be "
            f"recovered."
        )
    if (
        "tokenizer_id" in cfg.model_fields_set
        and cfg.tokenizer_id
        and cfg.tokenizer_id != tokenizer_id
    ):
        raise ValueError(
            f"Cannot resume from {cfg.resume_from}: the tokenizer comes from the "
            f"checkpoint ({tokenizer_id}) and this request asks for {cfg.tokenizer_id}"
        )
    return tokenizer_id


def build_dataset(
    cfg: TrainConfig,
    client,
    tokenizer: Tokenizer,
    *,
    key: str,
    selection: list[pack.StageSelection],
    tokenizer_id: str,
    fingerprint: str,
    on_progress: ProgressCallback | None = None,
) -> pack.PackStats:
    """Phase 1: text on disk -> corpus.bin + segments.json, under `key`.

    The vocabulary arrives already built — by a `tokenize` run, or by this run
    moments ago — because the corpus is keyed on it and cannot be packed before
    it is known.

    Reported as well as printed: this phase is minutes of a run during which the
    training figures have nothing to show, so a watcher who is told nothing
    cannot tell it from a run that is stuck.
    """

    def report(**fields) -> None:
        if on_progress is not None:
            on_progress(Tokenize(**fields))

    stages = cfg.stages()
    expected = sum(selected.taken for _, datasets in selection for selected in datasets)
    if not expected:
        raise ValueError(
            f"Nothing to pack: {_named(selection)} hold no documents between them. "
            f"A chat dataset holds conversations rather than prose — fine-tune on "
            f"those instead, which is a run that names `base_model`."
        )
    merges = bpe.merge_count(tokenizer)
    report(documents=expected, vocab=bpe.vocab_size(tokenizer), merges=merges, progress=0.6)

    # Per-dataset counts, because a cap that silently failed to apply is otherwise
    # invisible until the run's wall-clock comes in several times over.
    breakdown = ", ".join(
        f"{selected.name} {selected.taken:,}"
        for _, datasets in selection
        for selected in datasets
    )
    print(f"[2/6] Reading {expected:,} document(s): {breakdown}")

    print(f"[3/6] Packing corpus into {key}")
    stats = pack.pack(
        _documents(client, selection),
        tokenizer,
        key=key,
        # The *resolved* selection, not the request: a stage asking for "all"
        # must record the corpora that meant at build time, or `is_built` would
        # compare an expanded request against an unexpanded record and rebuild
        # every time.
        stages=[
            {
                "name": st.name,
                "epochs": st.epochs,
                "datasets": [selected.record() for selected in datasets],
            }
            for st, (_, datasets) in zip(stages, selection)
        ],
        tokenizer_id=tokenizer_id,
        fingerprint=fingerprint,
    )
    print(
        f"      {stats.corpus_len:,} tokens, {stats.documents:,} documents, "
        f"vocab {stats.vocab_size:,}"
    )
    # The frame that closes phase 1 is `_phase_one`'s, not this function's: what
    # a fresh build produced is measured off the corpus it wrote, by the same
    # code that measures one it is reusing.
    return stats


def _phase_one(cfg, tokenizer, dataset, corpus, documents, info) -> Tokenize:
    """The frame that closes phase 1, for a fresh build and a reused one alike.

    One function for both branches because a measurement that only a fresh build
    made reads as missing on every cached one, which is the lie the "omit what
    was not measured" convention exists to prevent. What actually differs is only
    whether a count came from the pack that just ran or from the sidecar it left
    behind — and everything below is a property of the corpus on disk.

    A build packed before these were recorded answers None for the ones it cannot
    reconstruct, which is the honest reading: nobody measured.
    """
    counts = info.get("token_counts")
    fit = spans_mod.window_fit(documents, cfg.ctx_window, len(corpus))
    ribbon = spans_mod.span_ribbon(documents, cfg.ctx_window, len(corpus))
    return Tokenize(
        documents=len(documents) or None,
        chars=dataset.chars if dataset else _chars(info),
        vocab=bpe.vocab_size(tokenizer),
        merges=bpe.merge_count(tokenizer),
        total_tokens=len(corpus),
        fertility=pack.fertility(info),
        dead_vocab=diagnostics.dead_ids(counts) if counts else None,
        merge_sequence=bpe.merge_sequence(tokenizer, counts) if counts else None,
        # Only when something was actually measured against it. On its own it
        # describes the run and not the build, and the panel reads a present
        # `ctx_window` as "these window-relative numbers are real".
        ctx_window=cfg.ctx_window if fit or ribbon else None,
        boundary_straddle=fit.straddle if fit else None,
        doc_length_hist=fit.length_hist if fit else None,
        docs_over_ctx=fit.over_ctx if fit else None,
        span_ribbon=ribbon.as_dict() if ribbon else None,
        progress=1.0,
        done=True,
    )


def _chars(info: dict) -> int | None:
    """How much text a corpus was packed from, summed back out of the per-dataset
    counters — so a reused build answers this rather than leaving it unmeasured."""
    recorded = info.get("fertility")
    return sum(one["chars"] for one in recorded) if recorded else None


def _documents(client, selection: list[pack.StageSelection]):
    """Every document of every stage, in order, tagged with its stage.

    A generator handed straight to the packer: one page is in memory at a time,
    and a dataset named by two stages is read twice on purpose — that is what
    "train on it again later" means, and the spans keep the two apart.
    """
    for index, (_, datasets) in enumerate(selection):
        for selected in datasets:
            for row in client.documents(selected.id, cap=selected.cap):
                text = remote.text_of(row)
                if text:
                    yield pack.SourceDoc(dataset=selected.name, text=text, stage=index)


def _named(selection: list[pack.StageSelection]) -> str:
    return ", ".join(
        str(selected) for _, datasets in selection for selected in datasets
    ) or "(nothing)"


def _vocabulary(
    cfg: TrainConfig,
    client,
    selection: list[pack.StageSelection],
    on_progress: ProgressCallback | None,
    state: TrainState | None = None,
) -> str:
    """The id of the vocabulary this run trains against.

    A resumed run (`state` is not None) is not a choice at all: it is the
    checkpoint's own tokenizer, via `resumed_tokenizer` — never re-derived from
    this run's dataset selection and never a stated `tokenizer_id` overriding it,
    for the same reason `resumed_architecture` refuses a disagreeing shape.

    Otherwise it is either one a `tokenize` run built and this run named, or —
    when it named none — one derived from this run's own selection. The derived
    id is a function of those inputs rather than a fresh uuid, so two identical
    runs still share a vocabulary and the corpus packed under it, which is what
    made skipping tokenization worth having in the first place.
    """
    if state is not None:
        tokenizer_id = resumed_tokenizer(cfg, state)
        print(f"[1/6] Vocabulary {tokenizer_id} (inherited from {cfg.resume_from})")
        return tokenizer_id

    if cfg.tokenizer_id:
        if vocab.info(cfg.tokenizer_id) is None:
            raise FileNotFoundError(
                f"No built vocabulary {cfg.tokenizer_id} — build one, or leave it unset "
                f"to learn one from this run's own corpora."
            )
        print(f"[1/6] Vocabulary {cfg.tokenizer_id} (built earlier)")
        return cfg.tokenizer_id

    # One entry per distinct selection across the curriculum, in first-seen order:
    # a corpus trained on twice is one body of text to learn merges from, and
    # counting it twice would shift the vocabulary for no reason a person asked for.
    flat = list(dict.fromkeys(selected for _, datasets in selection for selected in datasets))
    tokenizer_id = vocab.auto_id(flat, cfg.max_vocab)
    if not cfg.rebuild_dataset and vocab.matches(tokenizer_id, flat, cfg.max_vocab):
        print(f"[1/6] Reusing the vocabulary already learned from these corpora ({tokenizer_id})")
        return tokenizer_id

    print(f"[1/6] Learning a vocabulary (max_vocab {cfg.max_vocab})")
    stats = vocab.build(client, tokenizer_id, flat, cfg.max_vocab, on_progress)
    print(
        f"      {stats.vocab_size:,} tokens from {stats.sampled_bytes / 1e6:.1f} MB "
        f"drawn across {stats.sampled_chunks:,} point(s) in {len(flat)} dataset(s)"
    )
    return tokenizer_id


def run(
    cfg: TrainConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
    owner: str | None = None,
) -> dict:
    """The whole pipeline. Returns a summary of the run."""
    started = time.time()

    # Held open across phase 1 and closed before training starts: the documents
    # are read while the corpus is packed and never afterwards.
    with remote.Client(owner) as client:
        selection = pack.resolve(client, cfg.stages())
        # Resolved once, ahead of the vocabulary: a resumed run's tokenizer is
        # the checkpoint's own rather than anything derived from `selection`, and
        # `_train` needs this same state again to build the model and decide
        # resumed-vs-continued, so it is not resolved twice.
        state = resume_state(cfg.resume_from)
        tokenizer_id = _vocabulary(cfg, client, selection, on_progress, state)
        return _train(cfg, client, selection, tokenizer_id, state, on_progress, cancel, started)


def _train(cfg, client, selection, tokenizer_id, state, on_progress, cancel, started) -> dict:
    tokenizer = vocab.load(tokenizer_id)
    fingerprint = vocab.fingerprint(tokenizer_id)
    vocab_size = bpe.vocab_size(tokenizer)
    cfg = resumed_config(cfg, state, vocab_size)

    key = pack.corpus_key(fingerprint, selection)
    if cfg.rebuild_dataset or not pack.is_built(key, selection, fingerprint):
        dataset = build_dataset(
            cfg,
            client,
            tokenizer,
            key=key,
            selection=selection,
            tokenizer_id=tokenizer_id,
            fingerprint=fingerprint,
            on_progress=on_progress,
        )
    else:
        print(f"[2-3/6] Reusing the corpus packed at {key}")
        dataset = None

    corpus = pack.load_corpus(key)
    eot_id, documents, info = pack.load_segments(key)
    stage_meta = info.get("stages", [])

    if on_progress is not None:
        on_progress(_phase_one(cfg, tokenizer, dataset, corpus, documents, info))

    if documents:
        # Stage *membership* comes from the sidecar: the corpus on disk decides
        # which documents were packed into which stage, and a config that has
        # since changed would otherwise stage documents that were never packed
        # that way.
        #
        # Epoch counts come from the *request*. They are a training parameter,
        # not a property of the packed data — reading them off the sidecar means
        # reusing a dataset silently reimposes the epochs of whatever run built
        # it, so asking for 2 epochs over a corpus packed by a 3-epoch curriculum
        # quietly trains 3.
        requested = cfg.stages()
        stages = [
            Stage(
                # Names come from the request for the same reason epochs do, and
                # they are not in the corpus key — so a reused corpus must not
                # relabel this run's stages with whatever the packing run called
                # them.
                name=(requested[i].name if i < len(requested) else meta.get("name")),
                epochs=(
                    requested[i].epochs
                    if i < len(requested)
                    else meta.get("epochs", cfg.epochs)
                ),
                documents=documents.in_stage(i),
            )
            for i, meta in enumerate(stage_meta)
        ]
        data_plan = DataPlan(
            documents=documents,
            eot_id=eot_id,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            stages=stages,
        )
    else:
        data_plan = DataPlan.whole_corpus(len(corpus))

    device = loop.pick_device()
    model = _model(cfg, state, vocab_size)
    print(
        f"[4/6] Model: {model.parameter_count():,} parameters, "
        f"{model.layers}L/{model.heads}H/{model.dimensions}d on {device}"
    )

    plan_stages = data_plan.ordered_stages(cfg.epochs)
    print(
        f"[5/6] Training {len(plan_stages)} stage(s) over {len(corpus):,} tokens: "
        + ", ".join(
            f"{st.name or 'all'} x{st.epochs} ({len(st.documents)} docs)" for st in plan_stages
        )
    )

    provenance = {
        "parameters": model.parameter_count(),
        "device": device,
        # The shape this run trained at, recorded the same way `sft.py` records
        # it and read back through the same `architectureOf`. A from-scratch
        # pretrain also states it on `config.*`; a *resumed* one does not — the
        # request it stored asked for the checkpoint's shape by saying nothing —
        # so this is the only place either kind of continuation answers "what
        # shape is this", and the model row takes its `context` from here.
        "architecture": {name: getattr(cfg, name) for name in ARCHITECTURE_SUMMARY_FIELDS},
        # Recorded because they change throughput by ~2.5x and precision changes
        # numerics: two runs with the same config and very different curves or
        # wall-clocks are otherwise indistinguishable after the fact.
        "precision": cfg.precision,
        "compiled": cfg.compile_model,
        "tf32": cfg.allow_tf32,
        "vocab": model.vocab,
        "tokenizer_id": tokenizer_id,
        "tokenizer_fingerprint": fingerprint,
        "corpus_key": key,
        "corpus_tokens": len(corpus),
        "documents": len(data_plan.documents),
        # A dict in both branches. A reused corpus used to report the *string*
        # "reused", which made `sft._base_special_ids` return None — so every
        # model trained on one entered SFT with its chat markers at random init,
        # silently, which is the failure `llm/train/warm_start.py` exists to
        # prevent. The keyed sidecar records them, so reuse can answer too.
        "dataset": (
            dataset.__dict__
            if dataset
            else {
                "reused": True,
                "corpus_len": len(corpus),
                "documents": len(documents),
                "vocab_size": info.get("vocab_size"),
                "special_ids_seen": info.get("special_ids_seen"),
            }
        ),
    }
    continuation = _continuation(cfg, state, corpus, data_plan, key)

    summary = loop.run(
        model,
        corpus,
        data_plan,
        cfg,
        device,
        on_progress,
        cancel,
        token_counts=info.get("token_counts"),
        on_checkpoint=CheckpointWriter(
            tokenizer=tokenizer,
            config=cfg.model_dump(),
            provenance=provenance,
            state_extra={
                "corpus_key": key,
                "tokenizer_id": tokenizer_id,
                "tokenizer_fingerprint": fingerprint,
            },
        ),
        resume=continuation.state,
    )

    print("[6/6] Saving")
    model_dir = storage.ensure_dir(storage.model_dir(cfg.model_name))
    checkpoint.save(model, storage.weights_path(cfg.model_name))

    # A copy of the tokenizer beside the weights, so inference loads both from
    # one directory and a later retrain cannot leave a model paired with a
    # tokenizer it was never trained against.
    tokenizer.save(str(storage.tokenizer_file(model_dir)))

    result = {
        "model_name": cfg.model_name,
        **provenance,
        **continuation.record,
        **summary,
        "total_secs": round(time.time() - started, 1),
    }

    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(), "result": result}, indent=2),
        encoding="utf-8",
    )
    return result


def _model(cfg: TrainConfig, state: TrainState | None, vocab_size: int) -> Transformer:
    """The model this run trains, at the shape the config now states.

    A resumed run's config already carries the checkpoint's architecture rather
    than the request's — `resumed_config` put it there before anything read a
    number off it — so there is one shape here and not two.
    """
    model = Transformer(
        vocab=vocab_size,
        dimensions=cfg.dimensions,
        heads=cfg.heads,
        layers=cfg.layers,
        ff_multiplier=cfg.ff_multiplier,
        ctx_window=cfg.ctx_window,
        rope_base=cfg.rope_base,
        learned_temperature=cfg.learned_temperature,
    )
    if state is not None:
        model.load_state_dict(state.state_dict)
    return model


@dataclass
class _Continuation:
    """What the loop is handed, and what the run records about having been one."""

    state: TrainState | None
    record: dict


def _continuation(cfg, state, corpus, data_plan, corpus_key: str) -> _Continuation:
    """Which of the two things `resume_from` means, decided once.

    **Resumed** — the state was taken from a run over this corpus on this
    schedule, so the position, the optimizer and the generators are all still
    meaningful and the continuation is deterministic: it arrives where the
    uninterrupted run would have.

    **Continued** — anything else. The weights and the moments carry over and
    nothing else does, because nothing else still refers to the same thing: a
    step index into a different schedule is not a position, and a generator
    restored mid-way through data this run will never see is not an order. Same
    relationship a fine-tune has to its base model.

    Which of the two it was is recorded, because after the fact the two are
    indistinguishable from the weights and produce very different curves.
    """
    if state is None:
        return _Continuation(None, {})

    _, total_steps = loop.step_plan(corpus, data_plan, cfg)
    resumed = (
        state.step > 0
        and state.total_steps == total_steps
        and state.extra.get("corpus_key") == corpus_key
    )
    print(
        f"      {'Resuming' if resumed else 'Continuing'} from {cfg.resume_from}"
        + (f" at step {state.step}/{total_steps}" if resumed else " with fresh data and schedule")
    )
    return _Continuation(
        state if resumed else replace(state, step=0, rng={}),
        {
            "resume": {
                "model": cfg.resume_from,
                "mode": "resumed" if resumed else "continued",
                "step": state.step if resumed else 0,
                "optimizer": state.optimizer is not None,
            }
        },
    )
