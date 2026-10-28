"""A batch of gym episodes, carried in lockstep.

The reason this exists rather than reusing the browser's loop is one sentence: a
policy gradient is computed from token ids and log-probabilities, and no HTTP
frame carries them. So the rollout happens where the sampler is, and what comes
back from the gym is a *rendered* observation the LLM engine appends without ever
having laid it out — see `llm.gym.client`.

**The mask is written down as it is drawn.** Each episode holds one growing list
of ids and one growing list of flags saying which of them the model produced. It
is never reconstructed by re-rendering the conversation and looking for the
assistant's spans, because a re-tokenized rollout is a *guess* at what the
sampler drew — under a masked grammar not even a well-defined guess. Appending
is exact here for one reason worth stating: every seam between what the model
drew and what was appended around it is a marker, and a marker is an added token
the BPE never merges across.

**Lockstep is what keeps the card busy.** Every live episode generates its round
in one batched forward, and then all of their commands go out at once. The
alternative is `G x T` sequential waits on a sandbox with the GPU idle through
every one of them.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from llm.chat import tools
from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import ASSISTANT_OPEN, Role, Turn, build_prompt, tool_block
from llm.corpus import bpe
from llm.backend import RemoteError
from llm.gym.client import Episode
from llm.gym.interfaces import GymClient
from llm.model import sample
from llm.model.language import evaluating
from llm.model.generation import MAX_TOOL_CALLS_PER_TURN, ToolCalls as _ToolCalls
from llm.model.decoding import PrefixDecoder

#: What a round may spend on one reply. The browser's driver uses the same
#: number (`MAX_TOKENS` in `gym.rs`), and the two have to agree or a batch
#: practises a length its evaluation never gets.
MAX_REPLY_TOKENS = 256

#: An episode needs at least this much window left to say anything worth
#: grading. Below it the episode ends on what it last said rather than being
#: truncated mid-prompt — a model answering a question it was shown part of
#: reads afterwards as a model that scored zero.
MIN_REPLY_TOKENS = 32

#: How many sandbox calls a round may have in flight. The gyms engine and
#: just-bash behind it are the bound, not this process.
STEP_CONCURRENCY = 16


@dataclass
class Rollout:
    """One episode: its tokens, what the model drew, and what it earned."""

    episode: Episode
    ids: list[int]
    #: True at every index the model itself produced. The loss mask, and the
    #: only thing here that could not be recovered afterwards.
    policy: list[bool]
    #: Which rollouts this one's advantage is measured against. Opaque here:
    #: siblings are decided when the gym run is created, and pooling batches
    #: from several runs is what makes it a key rather than an index.
    group: str = ""
    reply: str = ""
    calls: list[dict] = field(default_factory=list)
    rounds: int = 0
    shaping: float = 0.0
    score: float = 0.0
    answer: str = ""
    truncated: bool = False
    #: False once the episode has been graded or abandoned.
    live: bool = True
    #: Why it stopped, for the frame a run publishes. Not a reward term.
    stopped: str = ""

    def took(self, ids: list[int], *, drawn: bool) -> None:
        self.ids.extend(ids)
        self.policy.extend([drawn] * len(ids))


def padded(rows: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-padded ids, each row's true positions, and which entries are real.

    Left rather than right because every row's *next* token then sits at the same
    index, which is what makes the decode step one gather instead of a loop.
    """
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    positions = torch.zeros_like(ids)
    live = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        pad = width - len(row)
        ids[index, pad:] = torch.tensor(row, dtype=torch.long, device=device)
        positions[index, pad:] = torch.arange(len(row), device=device)
        live[index, pad:] = True
    return ids, positions, live


@dataclass
class _Round:
    """The per-row grammar of one round of generation.

    One `OpenBlocks` and one `_ToolCalls` per row rather than a row-wise rewrite
    of either: both already mask a 1-D logit vector, and a Python loop over a
    batch this size costs less than the machinery to avoid it would.
    """

    blocks: list[OpenBlocks]
    calls: list[_ToolCalls]
    produced: list[list[int]]
    open: list[bool]

    @classmethod
    def of(cls, rules: Rules, tokenizer, width: int, budget: int) -> "_Round":
        return cls(
            blocks=[OpenBlocks(rules) for _ in range(width)],
            calls=[_ToolCalls(rules, tokenizer, budget) for _ in range(width)],
            produced=[[] for _ in range(width)],
            open=[True] * width,
        )

    def mask(self, logits: torch.Tensor) -> None:
        for index, (blocks, calls) in enumerate(zip(self.blocks, self.calls)):
            blocks.mask(logits[index])
            calls.mask(logits[index])

    def observe(self, index: int, token: int) -> None:
        self.blocks[index].observe(token)
        self.produced[index].append(token)
        self.calls[index].observe(token, self.produced[index])

    def depth(self) -> int:
        """The most closers any one row still owes."""
        return max((len(blocks.stack) for blocks in self.blocks), default=0)

    def drain(self, index: int) -> None:
        """Close whatever markup the token cap cut short, so a truncated reply is
        still well-formed — the same debt `chat.stream` settles at its own cap."""
        for closer in self.blocks[index].closers():
            self.produced[index].append(closer)
            self.calls[index].observe(closer, self.produced[index])


@torch.no_grad()
def _generate(
    model, rules, batch: list[Rollout], round_state: _Round, cfg, rope, generator, window: int
):
    """One round of every live episode's reply, in one batched decode.

    Under `no_grad`, and that is not an optimization detail. **The gradient is
    not taken over these forwards** — `policy.accumulate` does its own pass over
    the finished sequences once the returns are known — so a graph built here is
    retained activations for a whole batch of episodes across every round of
    generation, kept for a backward pass that never comes. It is the largest
    thing in the process and it is pure waste.

    Rows ending early remain in their own cache/full prefix until the round ends;
    attention never crosses rows. Legacy cache and declared full-prefix fallback
    preserve the same explicit padding/position semantics in evaluation mode.

    `window` bounds the decode as well as `cfg.max_tokens`, and it is not
    optional: the longest row sets the batch's positions, and a position past the
    RoPE tables raises rather than degrading — one long episode would take the
    whole run down with it. The closers a drain would cost are reserved against
    it, exactly as `chat.stream` reserves them, so the markup this round still
    has to close is budgeted rather than spent past the ceiling.
    """
    device = next(model.parameters()).device
    ids, positions, live = padded([roll.ids for roll in batch], device)
    with evaluating(model), PrefixDecoder(model, ids, positions, live, window=window, rope=rope) as decoder:
        for _ in range(cfg.max_tokens):
            if int(decoder.positions[:, -1:].max()) + 1 + round_state.depth() >= window:
                break
            step = decoder.values[:, -1].float()
            round_state.mask(step)
            drawn = sample.pick_rows(step, cfg.temperature, cfg.top_k, cfg.top_p, generator)

            for index, token in enumerate(drawn.tolist()):
                if not round_state.open[index]:
                    continue
                if token in (rules.end_of_assistant, rules.end_of_text):
                    round_state.open[index] = False
                    continue
                round_state.observe(index, token)

            if not any(round_state.open):
                break
            decoder.advance(drawn)

    for index in range(len(batch)):
        round_state.drain(index)


def _observed(client: GymClient, roll: Rollout) -> tuple[Rollout, str | None, float]:
    """The first call of a round, run. Only the first: that is the rule the
    engine sets for a chat turn, and an episode's conversation has to have the
    shape serving produces.

    **A step this side could not complete abandons the episode**, and the
    observation comes back as None. It is tempting to hand the model a caveat
    and carry on, and that is what the browser's driver does — but the caveat is
    a *rendered observation*, and rendering one here is the single thing this
    module may not do. So the episode leaves the batch the same way a closed
    episode leaves a run's mean: absent rather than zero, because it measures
    this side giving up and not the model.
    """
    call = roll.calls[0]
    try:
        seen = client.step(roll.episode.id, call["tool"], call["argument"], roll.reply)
    except RemoteError:
        return roll, None, 0.0
    return roll, seen.observation, seen.reward


@dataclass(frozen=True)
class Scaffold:
    """The marker ids a round is stitched together with.

    Built once per batch from the tokenizer, because appending them is what
    keeps the token sequence exact: `build_prompt` encodes a whole rendered
    conversation in one pass to avoid cross-turn BPE merges, and appending is
    only equivalent to that because every seam here is a marker.
    """

    end_of_assistant: int
    assistant_open: list[int]

    @classmethod
    def of(cls, rules: Rules, tokenizer) -> "Scaffold":
        return cls(
            end_of_assistant=rules.end_of_assistant,
            assistant_open=bpe.encode(tokenizer, ASSISTANT_OPEN),
        )


def _continue(
    roll: Rollout, scaffold: Scaffold, tokenizer, observation: str, window: int | None = None
) -> bool:
    """Close the assistant turn, append what the shell said, and open the next.

    The tool block is `template.tool_block`'s layout and `tools.render_result`'s
    content — the one renderer of that pair, which SFT trained against. What is
    *inside* the body was rendered by the backend and is appended verbatim; the
    engine has not looked at it.
    """
    call = roll.calls[0]
    answered = tools.render_result(call["tool"], call["argument"], observation)
    continuation = (
        [scaffold.end_of_assistant]
        + bpe.encode(tokenizer, tool_block(Turn(Role.TOOL, answered)))
        + scaffold.assistant_open
    )
    if window is not None and len(roll.ids) + len(continuation) + MIN_REPLY_TOKENS > window:
        return False
    roll.took(continuation, drawn=False)
    return True


def _grade(client: GymClient, roll: Rollout, why: str) -> None:
    """Finish the sampled reply through the gym seam; only legacy transport failures become absent episodes."""
    roll.live = False
    roll.stopped = why
    try:
        graded = client.finish(roll.episode.id, roll.reply)
    except RemoteError:
        roll.stopped = "ungraded"
        return
    roll.score = graded.score
    roll.answer = graded.answer


def _abandon(client: Client, roll: Rollout, why: str) -> None:
    roll.live = False
    roll.stopped = why
    roll.score = 0.0
    client.abandon(roll.episode.id)


def start(episodes: list[Episode], tokenizer, window: int) -> list[Rollout]:
    """A rollout per episode, each holding its prompt and nothing drawn yet."""
    started = []
    for episode in episodes:
        if not episode.tools:
            # Math must never be graded on a question silently cut to fit.
            unbounded = max(window, len(episode.prompt.encode()) + len(episode.system.encode()) + 64)
            full = build_prompt(episode.system, [Turn(Role.USER, episode.prompt)], tokenizer,
                                unbounded + MIN_REPLY_TOKENS, MIN_REPLY_TOKENS)
            if len(full) + MIN_REPLY_TOKENS > window:
                raise ValueError("math task does not fit the model context with room for an answer")
            started.append(Rollout(episode=episode, ids=full, policy=[False] * len(full)))
            continue
        ids = build_prompt(
            episode.system,
            [Turn(Role.USER, episode.prompt)],
            tokenizer,
            window,
            min(MAX_REPLY_TOKENS, max(1, window // 2)),
        )
        started.append(Rollout(episode=episode, ids=ids, policy=[False] * len(ids)))
    return started


def run(
    model,
    tokenizer,
    client: GymClient,
    episodes: list[Episode],
    cfg,
    *,
    window: int,
    rope=None,
    max_rounds: int,
    generator: torch.Generator | None = None,
) -> list[Rollout]:
    """Roll a whole batch of episodes out and hand back what they drew.

    Nothing is written down here. The gyms engine owns the record — runs,
    episodes, steps, rewards — exactly as it does for a browser-driven run; this
    holds the batch in memory, takes its step, and drops it.
    The final round masks tool calls so the model must answer in prose; tool-free
    episodes keep that same mask on every round.
    """
    rules = Rules.build(tokenizer)
    scaffold = Scaffold.of(rules, tokenizer)
    batch = start(episodes, tokenizer, window)

    for round_index in range(max_rounds):
        live = [roll for roll in batch if roll.live]
        if not live:
            break

        terminal = round_index + 1 == max_rounds
        state = _Round.of(rules, tokenizer, len(live), 0 if terminal else MAX_TOOL_CALLS_PER_TURN)
        for index, roll in enumerate(live):
            if not roll.episode.tools:
                state.calls[index] = _ToolCalls(rules, tokenizer, 0)
        _generate(model, rules, live, state, cfg, rope, generator, window)

        for index, roll in enumerate(live):
            roll.truncated |= state.open[index]
            roll.took(state.produced[index], drawn=True)
            roll.reply = bpe.decode(tokenizer, state.produced[index])
            roll.calls = list(state.calls[index].made)
            roll.rounds += 1

        _advance(client, live, scaffold, tokenizer, window, terminal)

    return batch


def _advance(client, live: list[Rollout], scaffold, tokenizer, window: int, terminal: bool) -> None:
    """Grade whoever answered, step whoever called, and open the next round."""
    calling = [roll for roll in live if roll.calls and not terminal]
    for roll in live:
        if roll not in calling:
            _grade(client, roll, "answered" if not terminal else "terminal")

    if not calling:
        return

    with ThreadPoolExecutor(max_workers=STEP_CONCURRENCY) as pool:
        seen = list(pool.map(lambda roll: _observed(client, roll), calling))

    for roll, observation, reward in seen:
        if observation is None:
            _abandon(client, roll, "unreachable")
            continue
        roll.shaping += reward
        # An episode with no room left to answer ends on what it last said,
        # rather than being handed an overlong context that would fail during
        # the policy pass after rollout.
        if not _continue(roll, scaffold, tokenizer, observation, window):
            _grade(client, roll, "window")
