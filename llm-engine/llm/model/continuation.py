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
from llm.model.decoding import PrefixDecoder
from llm.model.language import evaluating, model_facts

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


@torch.no_grad()
def continue_tokens(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    cfg: SamplingConfig,
    device: str = "cpu",
    cancel=None,
) -> str:
    """Generate a seeded unconstrained continuation within the saved context, stopping at end-of-text or cancellation.

    Evaluation behavior is request-local and restores heterogeneous modes. Declared
    models use complete prefixes until a separately validated cache adapter exists.
    """
    window = model_facts(model).context_window
    ids = bpe.encode(tokenizer, prompt)[: window - 1]
    if not ids or cancel is not None and cancel.is_set():
        return ""

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)

    produced: list[int] = []
    tokens = torch.tensor([ids], device=device)
    positions = torch.arange(len(ids), device=device)[None]
    with evaluating(model), PrefixDecoder(model, tokens, positions, torch.ones_like(tokens, dtype=torch.bool), window=window) as decoder:
        for index in range(min(cfg.max_tokens, window - len(ids))):
            if cancel is not None and cancel.is_set():
                break
            next_id = sample.pick(decoder.values[0, -1], cfg.temperature, cfg.top_k, cfg.top_p, generator)
            if next_id == eot_id:
                break
            produced.append(next_id)
            if index + 1 < min(cfg.max_tokens, window - len(ids)):
                decoder.advance(torch.tensor([next_id], device=device))

    return bpe.decode(tokenizer, produced)


def samples(
    model: torch.nn.Module,
    tokenizer,
    cfg: SamplingConfig,
    prompts: list[str],
    device: str = "cpu",
) -> list[Sample]:
    """One continuation per prompt. Each is seeded off the base seed plus its
    index, so prompts differ from each other but the suite as a whole repeats."""
    out: list[Sample] = []
    for i, prompt in enumerate(prompts):
        per_prompt = SamplingConfig(**{**vars(cfg), "seed": cfg.seed + i})
        out.append(Sample(prompt=prompt, text=continue_tokens(model, tokenizer, prompt, per_prompt, device)))
    return out
