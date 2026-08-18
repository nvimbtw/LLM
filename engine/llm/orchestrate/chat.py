"""The chat pipeline: a conversation in, an assistant turn out.

Renders the whole conversation through the canonical template — the same one
SFT trains on — and generates the next assistant turn under the marker grammar,
so a reply is well-formed markup rather than something the frontend has to
guess at.
"""

import time
from dataclasses import dataclass, field

import torch
from tokenizers import Tokenizer

from llm import storage
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Role, ThinkBudget, Turn, build_prompt
from llm.config import ChatConfig
from llm.corpus import bpe
from llm.model import checkpoint, sample
from llm.train.loop import pick_device

# Headroom left for the reply before the prompt starts evicting its own freshest
# tokens. The prompt is truncated oldest-first to leave this much room.
RESERVE_FOR_REPLY = 256

# How many tools one turn may call.
#
# Not a limit anyone has hit deliberately: it bounds a degenerate model that has
# learned to open a search and open another instead of answering. Without it a
# reply can spend its whole budget on calls the backend then has to execute.
MAX_TOOL_CALLS_PER_TURN = 4


@dataclass
class _ToolCalls:
    """The tool calls a turn has made, recognised as they close.

    Watching ids rather than the decoded text is the whole reason detection
    lives here: a marker is one token, while the pieces this generator yields
    are byte-level fragments that a caller downstream would have to reassemble
    before it could even see one.
    """

    rules: Rules
    tokenizer: Tokenizer
    made: list[dict] = field(default_factory=list)
    _opened_at: int | None = None

    def observe(self, token_id: int, produced: list[int]) -> dict | None:
        """The call `token_id` just completed, or None. `produced` must already
        include `token_id`, so the argument is everything between the two."""
        if token_id in self.rules.tool_open_for:
            self._opened_at = len(produced)
            return None

        name = self.rules.tool_close_for.get(token_id)
        if name is None or self._opened_at is None:
            return None

        argument = bpe.decode(self.tokenizer, produced[self._opened_at : -1])
        self._opened_at = None
        call = {"tool": name, "argument": argument.strip()}
        self.made.append(call)
        return call

    def mask(self, logits: torch.Tensor) -> None:
        """Refuse every tool opener once the turn has called enough of them.

        Applied after the grammar's own mask rather than inside it, because this
        is a budget the caller set and not a statement about well-formedness.
        """
        if len(self.made) < MAX_TOOL_CALLS_PER_TURN:
            return
        for token_id in self.rules.tool_open_for:
            if token_id < logits.numel():
                logits[token_id] = float("-inf")

_CACHE: dict = {}


def load(model_name: str):
    """The model + tokenizer for `model_name`, loaded once per process.

    Cached because loading weights per request would dominate the cost of a
    short generation.
    """
    if _CACHE.get("name") != model_name:
        device = pick_device()
        model, _ = checkpoint.load(storage.weights_path(model_name), device)
        tokenizer = bpe.load(storage.model_dir(model_name))
        _CACHE.update(name=model_name, model=model, tokenizer=tokenizer, device=device)
    return _CACHE["model"], _CACHE["tokenizer"], _CACHE["device"]


@torch.inference_mode()
def stream(cfg: ChatConfig):
    """Yield `(kind, data)` frames of the assistant turn, then return a summary.

    Two kinds: `("token", {"token": piece})` for every decoded fragment, and
    `("tool_call", {"tool", "argument"})` the moment a call closes — emitted
    alongside the tokens that spell it, not instead of them, so a caller that
    only renders text still sees the whole reply.
    """
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
    calls = _ToolCalls(rules, tokenizer)

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
        calls.mask(step)
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
        yield "token", {"token": after[len(before) :]}

        if (call := calls.observe(next_id, produced)) is not None:
            yield "tool_call", call

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
        yield "token", {"token": piece}
        # A call the cap cut short still closes here, and the caller is told
        # about it: the argument is whatever the model got out before the
        # ceiling, which is a poor query but a real one.
        if (call := calls.observe(closer, produced)) is not None:
            yield "tool_call", call

    return {
        "model_name": cfg.model_name,
        "text": bpe.decode(tokenizer, produced),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(produced),
        "drained_blocks": len(drained),
        "tool_calls": calls.made,
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
