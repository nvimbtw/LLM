"""The training pipeline.

    1. resolve the vocabulary          \
    2. collect source files             > skipped when the build already matches
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
from collections import Counter
from threading import Event

from tokenizers import Tokenizer

from llm import storage
from llm.config import TrainConfig
from llm.corpus import bpe, pack, sources, vocab
from llm.model import checkpoint
from llm.model.transformer import Transformer
from llm.train import loop
from llm.train.plan import DataPlan, Stage
from llm.train.progress import ProgressCallback, Tokenize, log_progress


def build_dataset(
    cfg: TrainConfig,
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
    files = sources.collect_stages(stages)
    if not files:
        chat = sources.chat_sets_among(cfg.datasets)
        raise ValueError(
            f"No .txt files found under {storage.input_dir()}"
            + (f" for datasets {cfg.datasets}" if cfg.datasets else "")
            # The same misroute `vocab._sft_hint` explains, reached instead of it
            # whenever the run named a vocabulary and so never built one.
            + (
                f" — but {chat} are SFT sets. A run trains on those only as a "
                f"fine-tune, and a fine-tune is a run that names `base_model`."
                if chat
                else ""
            )
        )
    merges = bpe.merge_count(tokenizer)
    report(files=len(files), vocab=bpe.vocab_size(tokenizer), merges=merges, progress=0.6)

    # Per-dataset counts, because a cap that silently failed to apply is otherwise
    # invisible until the run's wall-clock comes in several times over.
    per_dataset = Counter(f.dataset for f in files)
    breakdown = ", ".join(f"{name} {n:,}" for name, n in sorted(per_dataset.items()))
    print(f"[2/6] Collected {len(files):,} file(s) from {len(per_dataset)} dataset(s): {breakdown}")

    print(f"[3/6] Packing corpus into {key}")
    stats = pack.pack(
        files,
        tokenizer,
        key=key,
        # The *resolved* selection, not the request: a stage asking for "all"
        # must record the corpora that meant at build time, or `is_built` would
        # compare an expanded request against an unexpanded record and rebuild
        # every time.
        stages=[
            {"name": st.name, "epochs": st.epochs, "datasets": [[n, c] for n, c in datasets]}
            for st, (_, datasets) in zip(stages, selection)
        ],
        tokenizer_id=tokenizer_id,
        fingerprint=fingerprint,
    )
    print(
        f"      {stats.corpus_len:,} tokens, {stats.documents:,} documents, "
        f"vocab {stats.vocab_size:,}"
    )
    report(
        files=stats.files,
        chars=stats.chars,
        vocab=stats.vocab_size,
        merges=merges,
        total_tokens=stats.corpus_len,
        progress=1.0,
        done=True,
    )
    return stats


def _vocabulary(
    cfg: TrainConfig,
    selection: list[pack.StageSelection],
    on_progress: ProgressCallback | None,
) -> str:
    """The id of the vocabulary this run trains against.

    Either one a `tokenize` run built and this run named, or — when it named none
    — one derived from this run's own selection. The derived id is a function of
    those inputs rather than a fresh uuid, so two identical runs still share a
    vocabulary and the corpus packed under it, which is what made skipping
    tokenization worth having in the first place.
    """
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
    flat = list(dict.fromkeys(pair for _, datasets in selection for pair in datasets))
    tokenizer_id = vocab.auto_id(flat, cfg.max_vocab)
    if not cfg.rebuild_dataset and vocab.matches(tokenizer_id, flat, cfg.max_vocab):
        print(f"[1/6] Reusing the vocabulary already learned from these corpora ({tokenizer_id})")
        return tokenizer_id

    print(f"[1/6] Learning a vocabulary (max_vocab {cfg.max_vocab})")
    stats = vocab.build(tokenizer_id, flat, cfg.max_vocab, on_progress)
    print(
        f"      {stats.vocab_size:,} tokens from {stats.sampled_bytes / 1e6:.1f} MB "
        f"sampled across {stats.sampled_files:,} of {stats.files:,} file(s)"
    )
    return tokenizer_id


def run(
    cfg: TrainConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
) -> dict:
    """The whole pipeline. Returns a summary of the run."""
    started = time.time()

    selection = pack.resolve(cfg.stages())
    tokenizer_id = _vocabulary(cfg, selection, on_progress)
    tokenizer = vocab.load(tokenizer_id)
    fingerprint = vocab.fingerprint(tokenizer_id)

    key = pack.corpus_key(fingerprint, selection)
    if cfg.rebuild_dataset or not pack.is_built(key, selection, fingerprint):
        dataset = build_dataset(
            cfg,
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

    if dataset is None and on_progress is not None:
        # A reused corpus is still the data this run trains on, so phase 1 reports
        # what is on disk rather than nothing. Files and characters belong to the
        # build that is not happening, and stay unmeasured.
        on_progress(
            Tokenize(
                vocab=bpe.vocab_size(tokenizer),
                merges=bpe.merge_count(tokenizer),
                total_tokens=len(corpus),
                progress=1.0,
                done=True,
            )
        )

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
                documents=[d for d in documents if d.stage == i],
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
    model = Transformer(
        vocab=bpe.vocab_size(tokenizer),
        dimensions=cfg.dimensions,
        heads=cfg.heads,
        layers=cfg.layers,
        ff_multiplier=cfg.ff_multiplier,
        ctx_window=cfg.ctx_window,
        rope_base=cfg.rope_base,
        learned_temperature=cfg.learned_temperature,
    )
    print(
        f"[4/6] Model: {model.parameter_count():,} parameters, "
        f"{cfg.layers}L/{cfg.heads}H/{cfg.dimensions}d on {device}"
    )

    plan_stages = data_plan.ordered_stages(cfg.epochs)
    print(
        f"[5/6] Training {len(plan_stages)} stage(s) over {len(corpus):,} tokens: "
        + ", ".join(
            f"{st.name or 'all'} x{st.epochs} ({len(st.documents)} docs)" for st in plan_stages
        )
    )
    summary = loop.run(
        model,
        corpus,
        data_plan,
        cfg,
        device,
        on_progress,
        cancel,
        token_counts=info.get("token_counts"),
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
        "parameters": model.parameter_count(),
        "device": device,
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
        **summary,
        "total_secs": round(time.time() - started, 1),
    }

    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(), "result": result}, indent=2),
        encoding="utf-8",
    )
    return result
