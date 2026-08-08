"""The supervised fine-tuning pipeline.

    1. load the base model + its tokenizer
    2. render conversations into an aligned corpus/mask pair
    3. warm-start markers the base model's pretrain corpus never contained
    4. train with loss on assistant replies only
    5. save weights + tokenizer + meta.json

The architecture comes from the base checkpoint rather than from the request, so
a fine-tune cannot silently disagree with the weights it started from.
"""

import json
import time
from threading import Event

from llm import storage
from llm.chat.template import Reasoning
from llm.config import SftConfig, TrainConfig
from llm.corpus import bpe, sft as sft_corpus, special
from llm.model import checkpoint
from llm.train import loop
from llm.train.plan import DataPlan
from llm.train.progress import ProgressCallback, log_progress
from llm.train.warm_start import untrained_marker_ids, warm_start_embeddings


def _base_special_ids(base_model: str) -> list[int] | None:
    """Which markers the base model's pretrain corpus contained, or None when
    that was never recorded (a model trained before the field existed)."""
    path = storage.meta_path(base_model)
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    dataset = meta.get("result", {}).get("dataset")
    if not isinstance(dataset, dict):
        return None
    return dataset.get("special_ids_seen")


def run(
    cfg: SftConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
) -> dict:
    started = time.time()

    weights = storage.weights_path(cfg.base_model)
    if not weights.is_file():
        raise FileNotFoundError(
            f"No base model {cfg.base_model!r} at {weights} — pretrain one first"
        )

    device = loop.pick_device()
    model, arch = checkpoint.load(weights, device)
    print(
        f"[1/5] Base {cfg.base_model}: {model.parameter_count():,} parameters, "
        f"{arch['layers']}L/{arch['heads']}H/{arch['dimensions']}d on {device}"
    )

    # The base model's own tokenizer, not the shared build: a later pretrain may
    # have replaced the shared one, and a fine-tune tokenized differently from
    # the weights it starts from produces confident nonsense.
    tokenizer = bpe.load(storage.model_dir(cfg.base_model))

    reasoning = Reasoning.STRIP if cfg.strip_thinking else Reasoning.KEEP
    if cfg.rebuild_dataset or not sft_corpus.is_built():
        stats = sft_corpus.build(tokenizer, reasoning)
        print(
            f"[2/5] {stats.conversations:,} conversations, {stats.total_tokens:,} tokens, "
            f"{stats.trainable_tokens:,} trainable ({stats.trainable_fraction:.0%})"
            + (f", {stats.skipped} skipped" if stats.skipped else "")
        )
    else:
        stats = None
        print("[2/5] Reusing the SFT dataset already on disk")

    corpus, mask = sft_corpus.load()
    eot_id, documents, _ = sft_corpus.load_segments()

    # Markers this dataset uses that the base never saw, warm-started before the
    # first step — see llm/train/warm_start.py for why the caller decides this.
    marker_ids = {
        tokenizer.token_to_id(name)
        for name in special.token_list()
        if tokenizer.token_to_id(name) is not None
    }
    used = sorted(marker_ids.intersection(corpus.tolist()))
    to_reset = untrained_marker_ids(used, _base_special_ids(cfg.base_model))
    reset = warm_start_embeddings(model, to_reset)
    print(
        f"[3/5] Warm-start: reset {reset} embedding row(s)"
        + (" — this dataset uses markers the base never trained" if reset else " (none needed)")
    )

    data_plan = (
        DataPlan(documents=documents, eot_id=eot_id, shuffle=cfg.shuffle, seed=cfg.seed)
        if documents
        else DataPlan.whole_corpus(len(corpus))
    )

    # The training loop takes its shape from a TrainConfig; SFT's architecture
    # comes from the checkpoint, so the two are merged here rather than making
    # every caller below understand two config types.
    train_cfg = TrainConfig(
        model_name=cfg.model_name,
        dimensions=arch["dimensions"],
        heads=arch["heads"],
        layers=arch["layers"],
        ff_multiplier=arch["ff_multiplier"],
        ctx_window=arch["ctx_window"],
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        warmup_steps=cfg.warmup_steps,
        grad_clip=cfg.grad_clip,
        weight_decay=cfg.weight_decay,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )

    print(f"[4/5] Fine-tuning {cfg.epochs} epoch(s) at lr {cfg.lr:g}")
    summary = loop.run(model, corpus, data_plan, train_cfg, device, on_progress, cancel, mask)

    print("[5/5] Saving")
    model_dir = storage.ensure_dir(storage.model_dir(cfg.model_name))
    checkpoint.save(model, storage.weights_path(cfg.model_name))
    # The base tokenizer travels with the fine-tuned weights, so inference loads
    # both from one directory.
    tokenizer.save(str(storage.tokenizer_file(model_dir)))

    result = {
        "model_name": cfg.model_name,
        "base_model": cfg.base_model,
        "parameters": model.parameter_count(),
        "device": device,
        "warm_started": reset,
        "dataset": stats.__dict__ if stats else "reused",
        "trainable_tokens": int(mask.sum()),
        "total_tokens": len(corpus),
        **summary,
        "total_secs": round(time.time() - started, 1),
    }

    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(), "result": result}, indent=2),
        encoding="utf-8",
    )
    return result
