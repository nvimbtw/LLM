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
from llm.chat import tools
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Role, ThinkBudget, Turn, build_prompt, context_floor
from llm.config import ChatConfig, ChatTurn
from llm.corpus import bpe
from llm.model import checkpoint, sample, device
from llm.train.loop import pick_device, release_device

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
    #: How many calls this turn may open. `0` is a real value and not "unset":
    #: it refuses the opener from the first token, which is how a caller asks
    #: for an answer rather than another command.
    budget: int = MAX_TOOL_CALLS_PER_TURN
    made: list[dict] = field(default_factory=list)
    _opened_at: int | None = None

    def observe(self, token_id: int, produced: list[int]) -> dict | None:
        """The call `token_id` just completed, or None. `produced` must already
        include `token_id`, so the body is everything between the two.

        A body that is not a JSON object naming a tool this build has is not a
        call: it is reported as nothing and the reply carries on. That is the
        cost of one marker for every tool — the grammar guarantees the *shape*
        of a call and can no longer guarantee that a tool by that name exists —
        and it is why this returns None rather than raising. A model that
        mangles its JSON has written a sentence nobody acts on, which is a far
        better failure than a turn that dies.
        """
        if self.rules.tool_call is None:
            return None
        opener, closer = self.rules.tool_call

        if token_id == opener:
            self._opened_at = len(produced)
            return None
        if token_id != closer or self._opened_at is None:
            return None

        body = bpe.decode(self.tokenizer, produced[self._opened_at : -1])
        self._opened_at = None
        named = tools.parse(body)
        if named is None:
            return None

        name, argument = named
        call = {"tool": name, "argument": argument}
        self.made.append(call)
        return call

    def mask(self, logits: torch.Tensor) -> None:
        """Refuse every tool opener once the turn has called enough of them.

        Applied after the grammar's own mask rather than inside it, because this
        is a budget the caller set and not a statement about well-formedness.

        At a budget of zero this fires before the first token, which is the
        whole mechanism behind a gym episode's terminal round: masking the
        opener is the only way to make a model *stop searching and answer*, and
        it is a mask rather than an instruction because a model this size does
        not reliably follow one.
        """
        if len(self.made) < self.budget or self.rules.tool_call is None:
            return
        opener = self.rules.tool_call[0]
        if opener < logits.numel():
            logits[opener] = float("-inf")

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
        # Dropped *before* the replacement is built, not overwritten by it: for
        # the moment both are referenced, the card holds two models, and the one
        # being replaced is exactly the size of the one arriving.
        evict()
        device = pick_device()
        model, _ = checkpoint.load(storage.weights_path(model_name), device)
        tokenizer = bpe.load(storage.model_dir(model_name))
        _CACHE.update(name=model_name, model=model, tokenizer=tokenizer, device=device)
    return _CACHE["model"], _CACHE["tokenizer"], _CACHE["device"]


def _history(turns: list[ChatTurn]) -> list[Turn]:
    """The conversation as the template reads it.

    A tool turn that named its call is rendered here rather than by whoever sent
    it — see `ChatTurn` for why there is exactly one renderer of that block. An
    unnamed one is text that was already rendered, which is how a stored chat's
    tool message comes back, and it passes through untouched.

    There is no role branch here, and there is one in `corpus.sft._turns`, which
    is worth stating because the two are otherwise mirrors. An **assistant**
    turn can also name a call — the one it made rather than the one it answers,
    with the marker after its text — and that is how a gym demonstration is
    stored. It cannot reach here: `ChatTurn` refuses the pairing on anything but
    a tool turn, and a live assistant reply needs no naming anyway, because the
    engine stores it as the raw text the model produced with its markers already
    in it. A corpus row has no validator to refuse it, which is why the other
    function has to decide by role and this one does not.
    """
    rendered: list[Turn] = []
    for turn in turns:
        text = turn.text
        if turn.tool is not None and turn.argument is not None:
            if tools.by_name(turn.tool) is None:
                raise ValueError(f"unknown tool {turn.tool!r}")
            text = tools.render_result(turn.tool, turn.argument, turn.text)
        rendered.append(Turn(role=Role(turn.role), text=text))
    return rendered


@torch.inference_mode()
def stream(cfg: ChatConfig):
    with device.admit():
        return (yield from _stream(cfg))


def _stream(cfg: ChatConfig):
    """Yield `(kind, data)` frames of the assistant turn, then return a summary.

    Two kinds: `("token", {"token": piece})` for every decoded fragment, and
    `("tool_call", {"tool", "argument"})` the moment a call closes — emitted
    alongside the tokens that spell it, not instead of them, so a caller that
    only renders text still sees the whole reply.
    """
    started = time.time()
    model, tokenizer, device = load(cfg.model_name)

    history = _history(cfg.turns)
    budget = ThinkBudget.parse(cfg.budget) if cfg.budget else None

    # NTK-scaled tables computed fresh per turn, never written into the cached
    # model's own buffers — see `Transformer.extended_rope`. `context_window` is
    # this turn's ceiling everywhere `model.ctx_window` used to be it.
    context_window = round(model.ctx_window * cfg.context_multiplier)
    rope = model.extended_rope(context_window) if context_window > model.ctx_window else None

    reserve = min(RESERVE_FOR_REPLY, max(1, context_window // 2))
    prompt_ids = build_prompt(
        cfg.system, history, tokenizer, context_window, reserve, budget
    )
    if not prompt_ids:
        raise ValueError("Conversation rendered to zero tokens")

    rules = Rules.build(tokenizer)
    blocks = OpenBlocks(rules)
    budget = MAX_TOOL_CALLS_PER_TURN if cfg.max_tool_calls is None else cfg.max_tool_calls
    calls = _ToolCalls(rules, tokenizer, budget)

    generator = torch.Generator(device=device).manual_seed(cfg.seed) if cfg.seed else None

    logits, caches = model(
        torch.tensor([prompt_ids], device=device),
        caches=model.empty_caches(),
        pos_offset=0,
        rope=rope,
    )

    produced: list[int] = []
    position = len(prompt_ids)
    stopped = "length"

    for _ in range(cfg.max_tokens):
        # The closers a drain would cost are reserved against the cap, so the
        # markup this turn still has to close is budgeted rather than spent past
        # the ceiling the caller set.
        if position >= context_window - blocks.depth:
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
            torch.tensor([[next_id]], device=device), caches=caches, pos_offset=position, rope=rope
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


def history_floor(model_name: str, turns: list[ChatTurn]) -> int:
    """Index into `turns` where `/prune` should cut — see
    `llm.chat.template.context_floor` for what that means and why. Loads the
    model like `stream` does (cached the same way), because `ctx_window` is
    architecture and lives on the checkpoint, not in the tokenizer alone."""
    model, tokenizer, _ = load(model_name)
    history = _history(turns)
    reserve = min(RESERVE_FOR_REPLY, max(1, model.ctx_window // 2))
    return context_floor(history, tokenizer, model.ctx_window, reserve)


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
