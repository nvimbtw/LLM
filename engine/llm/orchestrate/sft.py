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

import numpy as np

from llm import storage
from llm.chat import template
from llm.chat.template import Reasoning
from llm.config import SftConfig, TrainConfig
from llm.corpus import bpe, pack, remote, sft as sft_corpus, special
from llm.eval import chat as chat_eval
from llm.model import checkpoint
from llm.orchestrate import train as train_pipeline
from llm.train import loop
from llm.train.diagnostics import DIAGNOSTIC_ROWS, Probes
from llm.train.plan import DataPlan
from llm.train.progress import ProgressCallback, Tokenize, log_progress
from llm.train.warm_start import untrained_marker_ids, warm_start_embeddings


def _base_run(base_model: str) -> dict:
    """What the run that produced the base model recorded about itself.

    Two meta.json shapes are in circulation and both are live, because two
    writers produce them. A run driven straight through the engine keeps what
    `run` wrote here: `{"config", "result"}`. A run driven by the backend queue
    has its meta *rewritten* on the way out by `training::write_meta`, which
    stores this whole result dict under the key `"dataset"` instead.

    Reading only the first spelling made `_base_special_ids` return None for
    every backend-launched run — `untrained_marker_ids` then saw "unknown" and
    reset nothing, and the ~28 chat-scaffold markers the pretrain never contained
    entered SFT at random init. That is exactly the failure
    `llm/train/warm_start.py` exists to prevent, and it was silent: the only
    sign was `warm_started: 0` on a base that had plainly never seen a chat
    template.
    """
    path = storage.meta_path(base_model)
    if not path.is_file():
        return {}
    meta = json.loads(path.read_text(encoding="utf-8"))
    run = meta.get("result")
    if not isinstance(run, dict):
        run = meta.get("dataset")
    return run if isinstance(run, dict) else {}


def _base_special_ids(base_model: str) -> list[int] | None:
    """Which markers the base model was trained on, or None when that was never
    recorded (a model trained before the field existed).

    The corpus stats are the `dataset` *inside* the run record either way — see
    `_base_run` for the two shapes that record comes in.
    """
    dataset = _base_run(base_model).get("dataset")
    if not isinstance(dataset, dict):
        return None
    return dataset.get("special_ids_seen")


def _pretrain_probe(base_model: str, ctx_window: int, vocab: int) -> np.ndarray | None:
    """A fixed batch of the base model's own pretraining text, or None.

    Reachable because a pretrain run records `corpus_key` in its meta.json and
    the key names a directory under `datasets/pretrain/` — so an SFT run that has
    already read that file for `special_ids_seen` can read the corpus the base
    model was built from without the backend, the data engine, or a second
    tokenizer being involved. The ids are the base tokenizer's on both sides,
    which is the property that makes this legitimate at all.

    It is held out from *this* run — SFT trains on conversations and never on
    prose — but it is not held out from pretraining, so the number is a drift
    signal and not an absolute measure of generalization: read whether it is
    climbing, not what it is.

    Rows are drawn at evenly spaced offsets rather than off the front, because
    `corpus.bin` is laid out in dataset and stage order and a probe taken from
    the head would measure whichever dataset happened to be packed first.
    """
    key = _base_run(base_model).get("corpus_key")
    if not isinstance(key, str) or not storage.pretrain_corpus_path(key).is_file():
        return None

    corpus = pack.load_corpus(key)
    window = ctx_window + 1
    rows = DIAGNOSTIC_ROWS
    if len(corpus) < rows * window:
        return None

    stride = (len(corpus) - window) // (rows - 1) if rows > 1 else 0
    probe = np.stack(
        [np.asarray(corpus[i * stride : i * stride + window], dtype=np.int64) for i in range(rows)]
    )
    # A corpus packed under a different vocabulary would index past the embedding
    # table and take the run down thousands of steps in. The key should make that
    # impossible; this is two thousand comparisons to be sure it is.
    return probe if int(probe.max()) < vocab else None


def _chat_probe(tokenizer, ctx_window: int, reasoning: Reasoning) -> np.ndarray | None:
    """A fixed batch of held-out conversations, rendered as SFT renders them.

    `eval/chat/` is the one chat corpus in the tree that no run may select, so
    the KL anchor is measured somewhere the fine-tune has genuinely not been.
    """
    ids: list[int] = []
    for conversation in chat_eval.load(storage.eval_dir() / "chat"):
        turns = _turns_of(conversation)
        if turns is None:
            continue
        rendered, _ = template.build_training_example(
            conversation.get("system"), turns, tokenizer, None, reasoning
        )
        ids.extend(rendered)

    rows = min(DIAGNOSTIC_ROWS, len(ids) // ctx_window)
    if rows < 1:
        return None
    return np.asarray(ids[: rows * ctx_window], dtype=np.int64).reshape(rows, ctx_window)


def _turns_of(conversation: dict) -> list[template.Turn] | None:
    """The conversation's turns, or None if any role is one SFT does not render.

    Refused rather than guessed at, for the reason `eval/chat.py` refuses it: an
    unknown role shifts the loss mask onto the wrong tokens.
    """
    turns = []
    for turn in conversation.get("turns", []):
        if turn.get("role") not in ("user", "assistant"):
            return None
        turns.append(template.Turn(role=template.Role(turn["role"]), text=turn.get("text", "")))
    return turns or None


def _probes(
    cfg: SftConfig,
    tokenizer,
    vocab: int,
    ctx_window: int,
    reasoning: Reasoning,
    device: str,
) -> Probes:
    """The fixed batches this fine-tune will be measured against.

    Built before the first step and never rebuilt: a probe that moved between
    reports would be a different question asked each time, and the whole value of
    these is that the same batch is scored against changing weights.

    Each part is independently optional, and a part that could not be resolved is
    left absent so it reaches the panel as "not measured" rather than as a zero —
    the base model's corpus may have been repacked or deleted, and `eval/chat/`
    may be empty. Both cases say so out loud: a silently missing forgetting
    signal is worse than none, because a flat chart reads as good news.
    """
    pretrain = _pretrain_probe(cfg.base_model, ctx_window, vocab)
    if pretrain is None:
        print("      no held-out pretrain loss: the base model names no corpus on disk")

    chat = _chat_probe(tokenizer, ctx_window, reasoning) if cfg.measure_base_kl else None
    if cfg.measure_base_kl and chat is None:
        print(f"      no base KL: eval/chat/ holds under {ctx_window} tokens of conversation")
        return Probes(pretrain=pretrain)

    reference = None
    if chat is not None:
        # A second full copy of the weights, which is why this is opt-in. Frozen
        # and in eval mode: it is the fixed thing the tune is measured against.
        reference, _ = checkpoint.load(storage.weights_path(cfg.base_model), device)
        reference.eval()
        reference.requires_grad_(False)

    return Probes(pretrain=pretrain, chat=chat, reference=reference)


def run(
    cfg: SftConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
    owner: str | None = None,
) -> dict:
    started = time.time()
    # Open across the build and closed before training: the conversations are
    # read while the dataset is rendered and never afterwards.
    with remote.Client(owner) as client:
        return _finetune(cfg, client, on_progress, cancel, started)


def _finetune(cfg, client, on_progress, cancel, started) -> dict:
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
    # Resolved before the key is derived, so a dataset pulled again since the last
    # build keys apart from the smaller one it replaced.
    selection = sft_corpus.resolve(client, cfg.datasets)
    key = sft_corpus.dataset_key(selection, reasoning)
    named = ", ".join(str(selected) for selected in selection)

    sft_corpus.clear_legacy()
    if on_progress is not None:
        on_progress(Tokenize(progress=0.05))
    if cfg.rebuild_dataset or not sft_corpus.is_built(key, selection, reasoning):
        stats = sft_corpus.build(client, tokenizer, selection, reasoning)
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

    if on_progress is not None:
        # A reused dataset reports what the files on disk say — the counts the
        # build would have returned are not recorded in the sidecar, and summing
        # the mask to recover the trainable share means reading the whole corpus
        # for one number, so it stays unmeasured rather than paid for.
        on_progress(
            Tokenize(
                conversations=stats.conversations if stats else len(documents),
                total_tokens=stats.total_tokens if stats else len(corpus),
                trainable_tokens=stats.trainable_tokens if stats else None,
                skipped_too_long=stats.skipped if stats else None,
                progress=1.0,
                done=True,
            )
        )

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
    probes = _probes(cfg, tokenizer, model.vocab, arch["ctx_window"], reasoning, device)
    provenance = {
        "base_model": cfg.base_model,
        "parameters": model.parameter_count(),
        "device": device,
        "warm_started": reset,
        # The one field both families answer "what shape is this" with, whatever
        # their config happens to say: SFT states no architecture at all — it
        # comes from the checkpoint by construction — and a *resumed* pretrain
        # deliberately states none either. `train.py` records the same key from
        # its own config for the same reason.
        "architecture": {name: arch[name] for name in train_pipeline.ARCHITECTURE_SUMMARY_FIELDS},
        # What this model was actually fine-tuned on, which is the only durable
        # way to tell two runs from the same base model apart afterwards.
        "dataset": {
            "sets": [one.record() for one in selection],
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
    }
    base = _base_run(cfg.base_model)
    summary = loop.run(
        model,
        corpus,
        data_plan,
        train_cfg,
        device,
        on_progress,
        cancel,
        mask,
        token_counts=info.get("token_counts"),
        probes=probes,
        # A fine-tune's checkpoints are models exactly as a pretrain's are, and
        # the writer is the pretrain's — what differs is only what each run knows
        # to record about itself.
        on_checkpoint=train_pipeline.CheckpointWriter(
            tokenizer=tokenizer,
            config=cfg.model_dump(),
            provenance=provenance,
            state_extra={
                "corpus_key": key,
                "tokenizer_id": base.get("tokenizer_id"),
                "tokenizer_fingerprint": base.get("tokenizer_fingerprint"),
            },
        ),
    )
    # The reference model is a second full copy of the weights, and this is the
    # last frame holding one. `release_device` (from `api/runner.py`'s finally)
    # hands back the *allocator's* pool, not memory something still points at, so
    # dropping it here is what makes the opt-in KL anchor releasable at all.
    probes.reference = None

    print("[5/5] Saving")
    model_dir = storage.ensure_dir(storage.model_dir(cfg.model_name))
    checkpoint.save(model, storage.weights_path(cfg.model_name))
    # The base tokenizer travels with the fine-tuned weights, so inference loads
    # both from one directory.
    tokenizer.save(str(storage.tokenizer_file(model_dir)))

    result = {
        "model_name": cfg.model_name,
        **provenance,
        **summary,
        "total_secs": round(time.time() - started, 1),
    }

    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(), "result": result}, indent=2),
        encoding="utf-8",
    )
    return result
