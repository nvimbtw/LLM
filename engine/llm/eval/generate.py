"""Generating text for the benchmark to measure.

**Deliberately unconstrained.** The markup grammar (`llm.chat.markup`) is a
serving-time constraint; generating under it here would measure the constraint
rather than the model, and score every model alike. What the suite reports is
what the model produces on its own.

Prompts are plain prose on purpose, so one suite works on a base model and an
SFT'd one alike.
"""

from dataclasses import dataclass

import torch

from llm.corpus import bpe, special
from llm.model import sample
from llm.model.transformer import Transformer

DEFAULT_PROMPTS = [
    "The old man walked down to the river and",
    "She opened the letter and read it slowly. It said",
    "In the morning the town was quiet, because",
    "There was once a small house at the edge of the",
    "He could not remember where he had left the",
    "The children ran across the field toward",
    "It had been raining for three days when",
    "The door opened and a woman came in. She was",
    "Nobody in the village had ever seen anything like",
    "After dinner they sat by the fire and talked about",
    "The book was old, and its pages were",
    "When the train finally arrived, the platform was",
]


@dataclass
class SamplingConfig:
    max_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    # Fixed so a benchmark is reproducible: two runs of the same model must not
    # differ because the sampler drew differently.
    seed: int = 1234


@dataclass
class Sample:
    prompt: str
    text: str


@torch.inference_mode()
def continue_tokens(
    model: Transformer,
    tokenizer,
    prompt: str,
    cfg: SamplingConfig,
    device: str = "cpu",
) -> str:
    """One continuation of `prompt`, stopping at end-of-text."""
    ids = bpe.encode(tokenizer, prompt)[: model.ctx_window - 1]
    if not ids:
        return ""

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)

    caches = model.empty_caches()
    logits, caches = model(torch.tensor([ids], device=device), caches=caches, pos_offset=0)

    produced: list[int] = []
    position = len(ids)
    for _ in range(cfg.max_tokens):
        if position >= model.ctx_window:
            break
        next_id = sample.pick(
            logits[0, -1], cfg.temperature, cfg.top_k, cfg.top_p, generator
        )
        if next_id == eot_id:
            break
        produced.append(next_id)
        logits, caches = model(
            torch.tensor([[next_id]], device=device), caches=caches, pos_offset=position
        )
        position += 1

    return bpe.decode(tokenizer, produced)


def samples(
    model: Transformer,
    tokenizer,
    cfg: SamplingConfig,
    prompts: list[str] | None = None,
    device: str = "cpu",
) -> list[Sample]:
    """One continuation per prompt. Each is seeded off the base seed plus its
    index, so prompts differ from each other but the suite as a whole repeats."""
    out: list[Sample] = []
    for i, prompt in enumerate(prompts or DEFAULT_PROMPTS):
        per_prompt = SamplingConfig(**{**vars(cfg), "seed": cfg.seed + i})
        out.append(Sample(prompt=prompt, text=continue_tokens(model, tokenizer, prompt, per_prompt, device)))
    return out
