"""The chat pipeline: a conversation in, an assistant turn out.

Distinct from `infer`, which continues raw prose. This renders the whole
conversation through the canonical template — the same one SFT trains on — and
generates the next assistant turn under the marker grammar, so a reply is
well-formed markup rather than something the frontend has to guess at.
"""

import time

import torch

from llm import storage
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Role, ThinkBudget, Turn, build_prompt
from llm.config import ChatConfig
from llm.corpus import bpe
from llm.model import checkpoint, sample
from llm.orchestrate.infer import load

# Headroom left for the reply before the prompt starts evicting its own freshest
# tokens. The prompt is truncated oldest-first to leave this much room.
RESERVE_FOR_REPLY = 256


@torch.inference_mode()
def stream(cfg: ChatConfig):
    """Yield decoded pieces of the assistant turn, then return a summary."""
    started = time.time()
    model, tokenizer, device = load(cfg.model_name)

    history = [Turn(role=Role(t.role), text=t.text) for t in cfg.turns]
    budget = ThinkBudget.parse(cfg.budget) if cfg.budget else None

    reserve = min(RESERVE_FOR_REPLY, max(1, model.ctx_window // 2))
    prompt_ids = build_prompt(
        cfg.system, history, tokenizer, model.ctx_window, reserve, budget
    )
    if not prompt_ids:
        raise ValueError("Conversation rendered to zero tokens")

    rules = Rules.build(tokenizer)
    blocks = OpenBlocks(rules)

    generator = torch.Generator(device=device).manual_seed(cfg.seed) if cfg.seed else None

    logits, caches = model(
        torch.tensor([prompt_ids], device=device), caches=model.empty_caches(), pos_offset=0
    )

    produced: list[int] = []
    position = len(prompt_ids)
    stopped = "length"

    for _ in range(cfg.max_tokens):
        # The closers a drain would cost are reserved against the cap, so the
        # markup this turn still has to close is budgeted rather than spent past
        # the ceiling the caller set.
        if position >= model.ctx_window - blocks.depth:
            break

        step = logits[0, -1].clone()
        blocks.mask(step)
        next_id = sample.pick(step, cfg.temperature, cfg.top_k, cfg.top_p, generator)

        if next_id in (rules.end_of_assistant, rules.end_of_text):
            stopped = "marker"
            break

        blocks.observe(next_id)
        produced.append(next_id)
        # Decoded against what came before and diffed: a byte-level token can be
        # half a UTF-8 character, so decoding it alone yields a replacement char.
        before = bpe.decode(tokenizer, produced[:-1])
        after = bpe.decode(tokenizer, produced)
        yield {"token": after[len(before) :]}

        logits, caches = model(
            torch.tensor([[next_id]], device=device), caches=caches, pos_offset=position
        )
        position += 1

    # The token cap is the one stop the grammar cannot prevent, so anything still
    # open is closed here — a truncated reply is still well-formed markup.
    drained = blocks.closers()
    for closer in drained:
        produced.append(closer)
        piece = bpe.decode(tokenizer, [closer])
        yield {"token": piece}

    return {
        "model_name": cfg.model_name,
        "text": bpe.decode(tokenizer, produced),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(produced),
        "drained_blocks": len(drained),
        "stopped": stopped,
        "elapsed_secs": round(time.time() - started, 2),
    }


def run(cfg: ChatConfig) -> dict:
    """Drain `stream` and return only the finished turn."""
    generator = stream(cfg)
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


def measure(model_name: str, conversation: dict) -> dict:
    """Token cost of a conversation as SFT would render it.

    The generator uses this to reject an over-long conversation at generation
    time rather than discovering it later as a training window full of context
    with no assistant token in it. Rendered under KEEP and with no budget: that
    is the widest rendering, and measuring a narrower one would under-report a
    conversation a later unstripped run still has to fit.
    """
    from llm.chat.template import Reasoning, build_training_example

    tokenizer = bpe.load(storage.model_dir(model_name))
    turns = [
        Turn(role=Role(t["role"]), text=t.get("text", ""))
        for t in conversation.get("turns", [])
        if t.get("role") in ("user", "assistant")
    ]
    if len(turns) != len(conversation.get("turns", [])):
        raise ValueError("every turn must have role 'user' or 'assistant'")

    ids, mask = build_training_example(
        conversation.get("system"), turns, tokenizer, None, Reasoning.KEEP
    )
    return {
        "model_name": model_name,
        "tokens": len(ids),
        "trainable_tokens": int(sum(mask)),
    }


def markers() -> dict:
    """The reserved marker vocabulary, so a caller can derive its tag set from
    the real thing rather than keeping a second copy in sync."""
    from llm.corpus import special

    return {
        "markers": special.SPECIAL_TOKENS,
        "block_size": special.SPECIAL_BLOCK,
        "end_of_text": special.END_OF_TEXT,
    }


def count_tokens(model_name: str, text: str) -> dict:
    """Token length of `text` under a model's tokenizer.

    Loads only the tokenizer, never the weights, so this is cheap enough to call
    per message for usage accounting.
    """
    path = storage.tokenizer_file(storage.model_dir(model_name))
    if not path.is_file():
        raise FileNotFoundError(f"No tokenizer for model {model_name!r} at {path}")
    tokenizer = bpe.load(storage.model_dir(model_name))
    return {"model_name": model_name, "tokens": len(bpe.encode(tokenizer, text))}
