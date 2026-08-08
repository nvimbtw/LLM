"""The training pipeline.

    1. collect source files
    2. train the tokenizer            \
    3. pack the corpus                 > skipped when the dataset is current
    4. build the model
    5. run the training loop
    6. save weights + tokenizer + meta.json

Steps 2-3 are a separate phase from 5 on purpose, mirroring the Rust
build_dataset/run_training split: tokenizing is expensive and only has to happen
when the *data* changes, not when the architecture or the learning rate does.
"""

import json
import time
from collections import Counter
from threading import Event

from llm import storage
from llm.config import TrainConfig
from llm.corpus import bpe, pack, sources
from llm.model import checkpoint
from llm.model.transformer import Transformer
from llm.train import loop
from llm.train.plan import DataPlan, Stage
from llm.train.progress import ProgressCallback, log_progress


def build_dataset(cfg: TrainConfig) -> pack.PackStats:
    """Phase 1: text on disk -> tokenizer + corpus.bin + segments.json."""
    stages = cfg.stages()
    files = sources.collect_stages(stages)
    if not files:
        raise ValueError(
            f"No .txt files found under {storage.input_dir()}"
            + (f" for datasets {cfg.datasets}" if cfg.datasets else "")
        )

    # Per-dataset counts, because a cap that silently failed to apply is otherwise
    # invisible until the run's wall-clock comes in several times over.
    per_dataset = Counter(f.dataset for f in files)
    breakdown = ", ".join(f"{name} {n:,}" for name, n in sorted(per_dataset.items()))
    print(f"[1/6] Collected {len(files):,} file(s) from {len(per_dataset)} dataset(s): {breakdown}")

    print(f"[2/6] Training tokenizer (max_vocab {cfg.max_vocab})")
    tokenizer = bpe.train([f.path for f in files], cfg.max_vocab)

    print("[3/6] Packing corpus")
    stats = pack.pack(
        files,
        tokenizer,
        [{"name": st.name, "epochs": st.epochs} for st in stages],
    )
    print(
        f"      {stats.corpus_len:,} tokens, {stats.documents:,} documents, "
        f"vocab {stats.vocab_size:,}"
    )
    return stats


def run(
    cfg: TrainConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
) -> dict:
    """The whole pipeline. Returns a summary of the run."""
    started = time.time()

    if cfg.rebuild_dataset or not pack.is_built():
        dataset = build_dataset(cfg)
    else:
        print("[1-3/6] Reusing the corpus already on disk (rebuild_dataset=true to force)")
        dataset = None

    tokenizer = bpe.load()
    corpus = pack.load_corpus()
    eot_id, documents, stage_meta = pack.load_segments()

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
                name=meta.get("name"),
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
    summary = loop.run(model, corpus, data_plan, cfg, device, on_progress, cancel)

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
        "corpus_tokens": len(corpus),
        "documents": len(data_plan.documents),
        "dataset": dataset.__dict__ if dataset else "reused",
        **summary,
        "total_secs": round(time.time() - started, 1),
    }

    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(), "result": result}, indent=2),
        encoding="utf-8",
    )
    return result
