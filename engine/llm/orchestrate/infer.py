"""The inference pipeline.

    1. load the weights (architecture comes out of the checkpoint)
    2. load the tokenizer saved beside them
    3. encode the prompt
    4. generate, one token at a time, reusing the KV cache
    5. decode

The model and tokenizer are cached in process: loading weights per request would
dominate the cost of a short generation.
"""

import time

import torch

from llm import storage
from llm.config import InferConfig
from llm.corpus import bpe, special
from llm.model import checkpoint, sample
from llm.train.loop import pick_device

_CACHE: dict = {}


def load(model_name: str):
    """The model + tokenizer for `model_name`, loaded once per process."""
    if _CACHE.get("name") != model_name:
        device = pick_device()
        model, _ = checkpoint.load(storage.weights_path(model_name), device)
        tokenizer = bpe.load(storage.model_dir(model_name))
        _CACHE.update(name=model_name, model=model, tokenizer=tokenizer, device=device)
    return _CACHE["model"], _CACHE["tokenizer"], _CACHE["device"]


def release() -> None:
    """Drop the cached model — the weights are the largest thing this process
    holds, and a retrain of the same name must not serve the stale ones."""
    _CACHE.clear()


@torch.inference_mode()
def stream(cfg: InferConfig):
    """Generate token by token, yielding each decoded piece as it lands.

    A generator rather than a callback so the caller decides what backpressure
    means — an SSE route yields straight out of it, and `run` below just drains
    it. Terminates with a dict summarising the generation.
    """
    started = time.time()
    model, tokenizer, device = load(cfg.model_name)

    prompt_ids = bpe.encode(tokenizer, cfg.prompt)
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens")

    # The prompt cannot fill the window completely — there would be no room to
    # generate. Oldest-first truncation keeps the most recent context.
    room = model.ctx_window - 1
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(cfg.seed)

    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)
    caches = model.empty_caches()

    # Prefill: the whole prompt in one pass, populating the cache. Every step
    # after this feeds a single token.
    logits, caches = model(
        torch.tensor([prompt_ids], device=device), caches=caches, pos_offset=0
    )

    generated: list[int] = []
    position = len(prompt_ids)

    for _ in range(cfg.max_tokens):
        if position >= model.ctx_window:
            break

        next_id = sample.pick(
            logits[0, -1],
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            generator=generator,
        )
        if next_id == eot_id:
            break

        generated.append(next_id)
        # Decoded incrementally against what came before, then diffed: a
        # byte-level BPE token can be half a UTF-8 character or half a word, so
        # decoding it alone yields a replacement character where decoding it in
        # context yields the real text.
        before = bpe.decode(tokenizer, generated[:-1])
        after = bpe.decode(tokenizer, generated)
        yield {"token": after[len(before) :]}

        logits, caches = model(
            torch.tensor([[next_id]], device=device), caches=caches, pos_offset=position
        )
        position += 1

    return {
        "model_name": cfg.model_name,
        "text": bpe.decode(tokenizer, generated),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "elapsed_secs": round(time.time() - started, 2),
    }


def run(cfg: InferConfig) -> dict:
    """Drain `stream` and return only the finished generation."""
    generator = stream(cfg)
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value
