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
    """Which markers the base model was trained on, or None when that was never
    recorded (a model trained before the field existed).

    Two meta.json shapes are in circulation and both are live, because two
    writers produce them. A run driven straight through the engine keeps what
    `run` wrote here: `{"config", "result"}`. A run driven by the backend queue
    has its meta *rewritten* on the way out by `training::write_meta`, which
    stores this whole result dict under the key `"dataset"` instead.

    So the run result is `result` or `dataset`, and the corpus stats are the
    `dataset` *inside* it either way. Reading only the first spelling made this
    return None for every backend-launched run — `untrained_marker_ids` then saw
    "unknown" and reset nothing, and the ~28 chat-scaffold markers the pretrain
    never contained entered SFT at random init. That is exactly the failure
    `llm/train/warm_start.py` exists to prevent, and it was silent: the only
    sign was `warm_started: 0` on a base that had plainly never seen a chat
    template.
    """
    path = storage.meta_path(base_model)
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    run = meta.get("result")
    if not isinstance(run, dict):
        run = meta.get("dataset")
    if not isinstance(run, dict):
        return None
    dataset = run.get("dataset")
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
    # Resolved before the key is derived, so "every set on disk" keys as the sets
    # that are actually there — a run asking for all of them after a new set was
    # fetched must not reuse a build that predates it.
    selection = sft_corpus.resolve(cfg.datasets)
    if not selection:
        raise FileNotFoundError(
            f"No SFT sets under {storage.sft_dir()} — fetch conversations first"
        )
    key = sft_corpus.dataset_key(selection, reasoning)
    named = ", ".join(f"{name}[:{cap}]" if cap else name for name, cap in selection)

    sft_corpus.clear_legacy()
    if cfg.rebuild_dataset or not sft_corpus.is_built(key, selection, reasoning):
        stats = sft_corpus.build(tokenizer, selection, reasoning)
        print(
            f"[2/5] {named} ({reasoning.value}) -> {stats.conversations:,} conversations, "
            f"{stats.total_tokens:,} tokens, {stats.trainable_tokens:,} trainable "
            f"({stats.trainable_fraction:.0%})"
            + (f", {stats.skipped} skipped" if stats.skipped else "")
        )
    else:
        stats = None
        print(f"[2/5] Reusing the built dataset for {named} ({reasoning.value})")

    corpus, mask = sft_corpus.load(key)
    eot_id, documents, info = sft_corpus.load_segments(key)

    # Markers this dataset uses that the base never saw, warm-started before the
    # first step — see llm/train/warm_start.py for why the caller decides this.
    marker_ids = {
        tokenizer.token_to_id(name)
        for name in special.token_list()
        if tokenizer.token_to_id(name) is not None
    }
    # The build records this, so the corpus is normally never read for it. Only a
    # dataset built before the field existed is scanned, and then in bounded
    # chunks — this is the step that OOM-killed the engine when it asked for the
    # whole corpus as a Python list.
    used = info.get("special_ids_seen")
    if used is None:
        used = sft_corpus.special_ids_in(corpus, marker_ids, eot_id)
    base_seen = _base_special_ids(cfg.base_model)
    to_reset = untrained_marker_ids(used, base_seen)
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
        # What this model was actually fine-tuned on, which is the only durable
        # way to tell two runs from the same base model apart afterwards.
        "dataset": {
            "sets": [{"name": name, "count": cap} for name, cap in selection],
            "reasoning": reasoning.value,
            "key": key,
            **(stats.__dict__ if stats else {"reused": True}),
            # The union of this corpus and the base's, so the record composes
            # down a pretrain -> SFT -> SFT chain. Recording only this run's
            # corpus loses the base's history at every hop: a thinking fine-tune
            # chained off a strip-thinking one would read <|thinking|> as never
            # trained and reset a row the *pretrain* had trained, throwing the
            # signal away. Written last so it wins over the `stats` copy above,
            # and derived from `used` rather than `stats` so a reused build —
            # which carries no stats at all — still records one.
            "special_ids_seen": sorted(set(used) | set(base_seen or [])),
        },
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
