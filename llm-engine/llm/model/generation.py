"""Canonical model generation, chat prompt/grammar and token rendering shared by legacy and supervised execution."""

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
import time

import torch
from tokenizers import Tokenizer

from llm.chat import tools
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Role, ThinkBudget, Turn, build_prompt
from llm.config import ChatConfig, ChatTurn
from llm.corpus import bpe, relabel
from llm.model import sample
from llm.model.decoding import PrefixDecoder
from llm.model.language import evaluating, model_facts
from llm.model.transformer import Transformer

RESERVE_FOR_REPLY = 256
MAX_TOOL_CALLS_PER_TURN = 4


@dataclass
class ToolCalls:
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

def history(turns: list[ChatTurn]) -> list[Turn]:
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


def prepare_prompt(cfg: ChatConfig, tokenizer: Tokenizer, context_window: int) -> list[int]:
    """Render canonical bounded prompt tokens using authoritative trained/request context, with private memory-label adoption."""
    if cfg.memories:
        tokenizer = relabel.adopt_memory_slots(tokenizer)
    reserve = min(RESERVE_FOR_REPLY, max(1, context_window // 2))
    ids = build_prompt(cfg.system, history(cfg.turns), tokenizer, context_window, reserve,
        ThinkBudget.parse(cfg.budget) if cfg.budget else None, cfg.project, cfg.memories)
    if not ids:
        raise ValueError("Conversation rendered to zero tokens")
    return ids


class TokenRenderer:
    """Own incremental UTF-8 text and closing tool-call recognition for one bounded token stream."""

    def __init__(self, cfg: ChatConfig, tokenizer: Tokenizer):
        """Keep request-local decoder/grammar observation without touching model state or a cached tokenizer."""
        self.tokenizer = relabel.adopt_memory_slots(tokenizer) if cfg.memories else tokenizer
        self.ids: list[int] = []
        self.decoded = ""
        self.incremental = bpe.stream_decoder(self.tokenizer)
        self.calls = ToolCalls(Rules.build(self.tokenizer), self.tokenizer,
            MAX_TOOL_CALLS_PER_TURN if cfg.max_tool_calls is None else cfg.max_tool_calls)

    def push(self, token_id: int) -> list[tuple[str, dict]]:
        """Interpret one verified token into the existing text/tool event sequence, preserving incomplete UTF-8 fragments."""
        self.ids.append(token_id)
        if self.incremental is None:
            current = bpe.decode(self.tokenizer, self.ids)
            piece = current[len(self.decoded):]
            self.decoded = current
        else:
            piece = self.incremental.push(token_id)
        frames = [("token", {"token": piece})]
        call = self.calls.observe(token_id, self.ids)
        if call is not None:
            frames.append(("tool_call", call))
        return frames

    def finish(self) -> list[tuple[str, dict]]:
        """Flush a final incomplete UTF-8 byte sequence so concatenated deltas match whole-reply decoding without inventing a token ID."""
        tail = self.incremental.finish() if self.incremental is not None else ""
        return [("token", {"token": tail})] if tail else []


@dataclass(frozen=True)
class DrawSummary:
    """Record grammar closure and stop semantics after a complete token iterator."""

    drained_blocks: int
    stopped: str


@torch.no_grad()
def draw_tokens(cfg: ChatConfig, model: torch.nn.Module, tokenizer: Tokenizer, device: str,
                prompt_ids: list[int], check_cancelled: Callable[[], None]) -> Generator[int, None, DrawSummary]:
    """Sample canonical grammar-constrained IDs inside one eval/no-grad cache lifetime; forced closures obey the same context bound."""
    check_cancelled()
    trained = model_facts(model).context_window
    window = round(trained * cfg.context_multiplier)
    if not isinstance(model, Transformer) and window != trained:
        raise ValueError("context extension is unsupported for this saved model's validated profile")
    rope = model.extended_rope(window) if window > trained else None
    rules = Rules.build(tokenizer)
    blocks = OpenBlocks(rules)
    calls = ToolCalls(rules, tokenizer, MAX_TOOL_CALLS_PER_TURN if cfg.max_tool_calls is None else cfg.max_tool_calls)
    produced: list[int] = []
    generator = torch.Generator(device=device).manual_seed(cfg.seed) if cfg.seed else None
    tokens = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
    positions = torch.arange(len(prompt_ids), device=device)[None]
    stopped = "length"
    with evaluating(model), PrefixDecoder(model, tokens, positions, torch.ones_like(tokens, dtype=torch.bool), window=window, rope=rope) as decoder:
        for _ in range(cfg.max_tokens):
            check_cancelled()
            if len(prompt_ids) + len(produced) >= window - blocks.depth:
                break
            step = decoder.values[0, -1].clone()
            blocks.mask(step)
            calls.mask(step)
            chosen = sample.pick(step, cfg.temperature, cfg.top_k, cfg.top_p, generator)
            if chosen in (rules.end_of_assistant, rules.end_of_text):
                stopped = "marker"
                break
            blocks.observe(chosen)
            produced.append(chosen)
            calls.observe(chosen, produced)
            yield chosen
            decoder.advance(torch.tensor([chosen], device=device))
        drained = blocks.closers()
        for closer in drained:
            check_cancelled()
            yield closer
    return DrawSummary(len(drained), stopped)


def stream_model(cfg: ChatConfig, model: torch.nn.Module, tokenizer: Tokenizer, device: str,
                 *, prompt_ids: list[int] | None = None, check_cancelled: Callable[[], None] = lambda: None,
                 on_token: Callable[[int], None] = lambda token: None):
    """Yield the existing chat frames/summary from an already admitted resident model, with explicit cancellation/token transport seams."""
    started = time.monotonic()
    trained = model_facts(model).context_window
    window = round(trained * cfg.context_multiplier)
    if cfg.memories:
        tokenizer = relabel.adopt_memory_slots(tokenizer)
    if prompt_ids is None:
        prompt_ids = prepare_prompt(cfg, tokenizer, window)
    renderer = TokenRenderer(cfg, tokenizer)
    draws = draw_tokens(cfg, model, tokenizer, device, prompt_ids, check_cancelled)
    try:
        while True:
            try:
                token_id = next(draws)
            except StopIteration as done:
                outcome = done.value
                break
            on_token(token_id)
            yield from renderer.push(token_id)
    finally:
        draws.close()
    yield from renderer.finish()
    return {"model_name": cfg.model_name, "text": bpe.decode(tokenizer, renderer.ids),
        "prompt_tokens": len(prompt_ids), "generated_tokens": len(renderer.ids),
        "drained_blocks": outcome.drained_blocks, "tool_calls": renderer.calls.made,
        "stopped": outcome.stopped, "elapsed_secs": round(time.monotonic() - started, 2)}
