"""The chat pipeline: a conversation in, an assistant turn out.

Renders the whole conversation through the canonical template — the same one
SFT trains on — and generates the next assistant turn under the marker grammar,
so a reply is well-formed markup rather than something the frontend has to
guess at.
"""

import torch

from llm import storage
from llm.chat.context import MemoryContext, ProjectContext
from llm.chat.template import ThinkBudget, context_floor
from llm.config import ChatConfig, ChatTurn
from llm.corpus import bpe, relabel
from llm.model import checkpoint, device
from llm.train.loop import pick_device, release_device

from llm.model.generation import (
    MAX_TOOL_CALLS_PER_TURN, RESERVE_FOR_REPLY, ToolCalls as _ToolCalls,
    history as _history, stream_model,
)

_CACHE: dict = {}


def evict() -> None:
    """Drop the cached model and give its device memory back.

    There is one GPU and one process, so a model parked here is memory a training
    run cannot have — and a cached model is never asked to leave on its own: it
    sits there until some later chat happens to name a different one. On an 8 GB
    card that is the difference between a run that fits and a run that does not,
    which is why `api/runner.py` calls this before it starts one.

    The cost is a reload on the next chat, and it is worth paying in this
    direction only: a run is minutes to hours, a reload is under a second.
    """
    if not _CACHE:
        return
    _CACHE.clear()
    release_device()


def load(model_name: str):
    """The model + tokenizer for `model_name`, loaded once per process.

    Cached because loading weights per request would dominate the cost of a
    short generation.
    """
    if _CACHE.get("name") != model_name:
        evict()
        device = pick_device()
        model, _ = checkpoint.load(storage.weights_path(model_name), device)
        tokenizer = bpe.load(storage.model_dir(model_name))
        _CACHE.update(name=model_name, model=model, tokenizer=tokenizer, device=device)
    return _CACHE["model"], _CACHE["tokenizer"], _CACHE["device"]


@torch.no_grad()
def stream(cfg: ChatConfig):
    """Own parent device admission for one legacy model generation and release it when the iterator closes."""
    with device.admit():
        return (yield from _stream(cfg))


def _stream(cfg: ChatConfig):
    """Render/generate through the canonical model adapter after legacy cache loading."""
    model, tokenizer, device_name = load(cfg.model_name)
    return (yield from stream_model(cfg, model, tokenizer, device_name))


def run(cfg: ChatConfig) -> dict:
    """Drain `stream` and return only the finished turn."""
    generator = stream(cfg)
    while True:
        try:
            next(generator)
        except StopIteration as done:
            return done.value


def history_floor(
    model_name: str, turns: list[ChatTurn], system: str | None = None,
    budget: str | None = None, project: ProjectContext | None = None,
    memories: list[MemoryContext] | None = None,
) -> int:
    """Index into `turns` where `/prune` should cut — see
    `llm.chat.template.context_floor` for what that means and why. Loads the
    model like `stream` does (cached the same way), because `ctx_window` is
    architecture and lives on the checkpoint, not in the tokenizer alone."""
    model, tokenizer, _ = load(model_name)
    history = _history(turns)
    reserve = min(RESERVE_FOR_REPLY, max(1, model.ctx_window // 2))
    return context_floor(history, tokenizer, model.ctx_window, reserve,
                         system, ThinkBudget.parse(budget) if budget else None,
                         project, memories)


def validate_context_tokenizer(model_name: str, memories: list[MemoryContext]) -> None:
    """Check reserved-slot compatibility without model loading before Rust persists input."""
    if memories:
        tokenizer = bpe.load(storage.model_dir(model_name))
        relabel.adopt_memory_slots(tokenizer)


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
