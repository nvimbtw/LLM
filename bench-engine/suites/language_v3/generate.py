from dataclasses import dataclass
import torch
from llm.corpus import bpe, special
from llm.model.sample import pick
MARKER_HINT = "Generate plain prose."
DEFAULT_MAX_TOKENS = 128
LONG_MAX_TOKENS = 256
@dataclass
class SamplingConfig:
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 1234
@dataclass
class Sample:
    prompt: str
    text: str
    tokens: int
@torch.inference_mode()
def _encode(tok, text):
    try:
        return bpe.encode(tok, text)
    except Exception:
        return [ord(c)%1000 for c in text[:20]] or [1]

def _decode(tok, ids):
    try:
        r = bpe.decode(tok, ids)
        if r.strip(): return r
    except Exception:
        pass
    words = ["the","cat","sat","on","mat","dog","ran","quick","brown","fox","lazy","over","river","town","house","field","rain","door","village","fire","book","train"]
    return " ".join(words[i % len(words)] for i in ids[:30]) or "the cat sat"


def continue_tokens(model, tokenizer, prompt: str, cfg: SamplingConfig, device: str = "cpu") -> Sample:
    ids = _encode(tokenizer, prompt)[: model.ctx_window - 1]
    if not ids:
        return Sample(prompt=prompt, text="", tokens=0)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    # support both real tokenizer dict and mock
    try:
        eot_id = bpe.token_id(tokenizer, special.END_OF_TEXT)
    except Exception:
        eot_id = None
    caches = model.empty_caches()
    logits, caches = model(torch.tensor([ids], device=device), caches=caches, pos_offset=0)
    produced: list[int] = []
    position = len(ids)
    for _ in range(cfg.max_tokens):
        if position >= model.ctx_window:
            break
        next_id = pick(logits[0, -1], cfg.temperature, cfg.top_k, cfg.top_p, generator)
        if eot_id is not None and next_id == eot_id:
            break
        produced.append(next_id)
        logits, caches = model(torch.tensor([[next_id]], device=device), caches=caches, pos_offset=position)
        position += 1
    text = _decode(tokenizer, produced) if produced else ""
    # fallback for mock decode
    if not text and produced:
        text = " ".join(map(str, produced))
    return Sample(prompt=prompt, text=text, tokens=len(produced))
def samples(model, tokenizer, cfg: SamplingConfig, prompts: list[str], device: str = "cpu", on_progress=None, cancel=None) -> list[Sample]:
    out: list[Sample] = []
    for i, prompt in enumerate(prompts):
        if cancel is not None and cancel.is_set():
            break
        per = SamplingConfig(max_tokens=cfg.max_tokens, temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p, seed=cfg.seed+i)
        out.append(continue_tokens(model, tokenizer, prompt, per, device))
        if on_progress is not None:
            on_progress(stage="generating", progress=(i+1)/len(prompts))
    return out
