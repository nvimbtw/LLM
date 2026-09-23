"""The canonical chat template — the single definition of how a conversation is
rendered into tokens for the model.

A conversation is laid out as::

    <|system|>S<|/system|><|user|>U1<|/user|><|assistant|>A1<|/assistant|><|user|>U2<|/user|><|assistant|>

an optional system block once at the front, alternating user/assistant turns,
always ending on an *open* `<|assistant|>` for the model to continue. Generation
stops when it emits `<|/assistant|>`.

A tool's answer is a third kind of turn, sitting between the assistant turn that
called for it and the one that uses it — the same *outside every turn* position
the system block occupies at the front::

    …<|assistant|>A1<|/assistant|><|tool-result|>R<|/tool-result|><|assistant|>

It is scaffold in both directions: never a training target, and forbidden to the
model by the grammar in `chat.markup`. See `chat.tools` for how R is laid out.

Inference builds prompts through here, and SFT loss-masking renders the exact
same shape, so the format the model is *trained* on and the format it is
*prompted* with cannot silently drift apart.

Ported from backend/crates/llm/src/chat_template.rs.
"""

from dataclasses import dataclass
from enum import Enum

from tokenizers import Tokenizer

from llm.chat.tools import TOOL_RESULT_CLOSE, TOOL_RESULT_OPEN
from llm.corpus import bpe

# The trailing marker every prompt ends on; the model generates from right here.
ASSISTANT_OPEN = "<|assistant|>"
ASSISTANT_CLOSE = "<|/assistant|>"
THINKING_OPEN = "<|thinking|>"
THINKING_CLOSE = "<|/thinking|>"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    #: A tool's answer, handed back between the assistant turn that asked for it
    #: and the one that uses it. Scaffold the model reads and never writes, so a
    #: turn with this role is never a training target — see `training_segments`.
    TOOL = "tool"


class ThinkBudget(str, Enum):
    """How much reasoning the assistant should show before answering.

    Three levels rather than a token count, because the label has to be
    something a prompt can ask a teacher for, a person can pick off a menu, and
    a model can condition on — and a model asked for "173 thinking tokens"
    cannot count its own tokens.
    """

    LOW = "low"
    MED = "med"
    HIGH = "high"

    @property
    def marker(self) -> str:
        """The reserved marker this budget renders as. One token — it has its own
        id in the special block, so the model conditions on a single embedding row
        rather than on a handful of ordinary word tokens that mean something else
        everywhere else in the corpus."""
        return f"<|think:{self.value}|>"

    @classmethod
    def parse(cls, text: str) -> "ThinkBudget | None":
        """Parse the wire/on-disk spelling. None for anything else, including
        "auto" — the absence of a budget is a real state (render nothing), which
        the caller expresses as None rather than as a fourth variant."""
        match text.strip().lower():
            case "low":
                return cls.LOW
            case "med" | "medium":
                return cls.MED
            case "high":
                return cls.HIGH
        return None


class Reasoning(str, Enum):
    """Whether a rendered assistant turn keeps its reasoning.

    STRIP drops the `<|thinking|>` block *and the budget marker describing it*,
    because the marker's meaning is "the assistant turn after me shows its
    reasoning at this budget" — keeping it over a stripped reply would claim a
    block that is not there.
    """

    KEEP = "keep"
    STRIP = "strip"


@dataclass
class Turn:
    role: Role
    text: str


def _budget_marker(budget: ThinkBudget | None) -> str:
    """The budget marker, or "" when there is none. Written once here so the
    three renderers cannot disagree about whether an absent budget emits
    anything."""
    return budget.marker if budget else ""


def strip_thinking(text: str) -> str:
    """Drop every `<|thinking|>` block from an assistant turn, keeping the reply.

    All of them, not just the first: a reply can open several blocks (it may
    think again after it has started answering), and removing only the leading
    one leaves the rest as literal markers in what is supposed to be
    reasoning-free text. An unclosed block is left alone rather than guessed at.
    """
    out: list[str] = []
    rest = text
    # Bounded: each iteration drops at least THINKING_OPEN, so `rest` shrinks.
    while (open_at := rest.find(THINKING_OPEN)) != -1:
        # Searched from the opener, so a stray closer *before* it is not treated
        # as this block's end (which would delete the text between them).
        close_at = rest.find(THINKING_CLOSE, open_at)
        if close_at == -1:
            break
        out.append(rest[:open_at])
        rest = rest[close_at + len(THINKING_CLOSE) :]
    out.append(rest)
    return "".join(out).strip()


def tool_block(turn: Turn) -> str:
    """A tool's answer. Its text is the already-rendered inner content — the
    call it answers plus the answer — which `chat.tools.render_result` builds
    for both this path and the SFT data, so the two cannot lay it out
    differently."""
    return f"{TOOL_RESULT_OPEN}{turn.text.strip()}{TOOL_RESULT_CLOSE}"


def _format_turn(turn: Turn) -> str:
    """A *history* turn, as `build_prompt` renders it.

    Deliberately carries no budget marker, even on a conversation that has one:
    a history turn has had its reasoning stripped, so marking it would claim a
    thinking block that is not there. Only the trailing open `<|assistant|>` —
    the one about to be generated — is marked.

    An assistant turn keeps its tool calls. `strip_thinking` takes the reasoning
    and nothing else, because the tool result that follows names the call it
    answers, and a result answering a call no longer in the prompt is a
    non-sequitur.
    """
    if turn.role is Role.USER:
        return f"<|user|>{turn.text.strip()}<|/user|>"
    if turn.role is Role.TOOL:
        return tool_block(turn)
    return f"{ASSISTANT_OPEN}{strip_thinking(turn.text)}{ASSISTANT_CLOSE}"


def _system_block(system: str | None) -> str:
    if system and system.strip():
        return f"<|system|>{system.strip()}<|/system|>"
    return ""


def render_conversation(
    system: str | None,
    turns: list[Turn],
    budget: ThinkBudget | None = None,
    reasoning: Reasoning = Reasoning.KEEP,
) -> str:
    """Render a whole conversation to text, in the shape
    `build_training_example` tokenizes.

    For seeding conversations into a *pretrain* corpus: the point of putting
    them there is that the model meets the chat scaffold during pretraining
    instead of first meeting it at SFT, so the text carries the same markers in
    the same places.
    """
    out = [_system_block(system)]

    for turn in turns:
        if turn.role is Role.USER:
            out.append(f"<|user|>{turn.text.strip()}<|/user|>")
            continue

        if turn.role is Role.TOOL:
            out.append(tool_block(turn))
            continue

        # Kept, the marker describes real content — and this is the path that
        # puts it into the pretrain corpus, which is what stops its embedding row
        # arriving at SFT at random init. Stripped, there is no block for it to
        # describe, so it goes too.
        if reasoning is Reasoning.KEEP:
            out.append(_budget_marker(budget))
            out.append(ASSISTANT_OPEN)
            out.append(turn.text.strip())
        else:
            out.append(ASSISTANT_OPEN)
            out.append(strip_thinking(turn.text))
        out.append(ASSISTANT_CLOSE)

    return "".join(out)


def _fit_history(
    history: list[Turn], tokenizer: Tokenizer, scaffold: int, allowance: int
) -> int:
    """Index of the oldest turn that survives once `history` is fitted under
    `allowance` tokens with `scaffold` already spent — newest first, whole
    turns only, the most recent kept even if it alone busts the budget.

    The walk `build_prompt` renders from, and the one `context_floor` reuses to
    ask where a cut would land under a *different* budget, without rendering
    anything.
    """
    used = scaffold
    start = len(history)
    for i in range(len(history) - 1, -1, -1):
        turn_cost = len(bpe.encode(tokenizer, _format_turn(history[i])))
        is_latest = i == len(history) - 1
        if not is_latest and used + turn_cost > allowance:
            break
        used += turn_cost
        start = i
    return start


def build_prompt(
    system: str | None,
    history: list[Turn],
    tokenizer: Tokenizer,
    ctx_window: int,
    reserve_for_reply: int,
    budget: ThinkBudget | None = None,
) -> list[int]:
    """Render a conversation into token ids, dropping the oldest whole turns
    until it fits `ctx_window - reserve_for_reply` tokens.

    The system block and the trailing open `<|assistant|>` are mandatory
    scaffold and always kept; only history turns are dropped, always at whole-
    turn boundaries so the prompt never begins mid-sentence. The most recent
    turn is kept even if it alone busts the budget — better a hard-cut latest
    message than nothing — and `_hard_cut` is where that cut actually happens.
    """
    system_str = _system_block(system)
    budget_str = _budget_marker(budget)

    def cost(text: str) -> int:
        return len(bpe.encode(tokenizer, text)) if text else 0

    # The always-present scaffold. The budget marker is part of it — mandatory
    # once asked for, exactly like the system block — so its cost comes out of
    # the budget before history is fitted around it.
    scaffold = cost(ASSISTANT_OPEN) + cost(budget_str) + cost(system_str)
    allowance = max(0, ctx_window - reserve_for_reply)
    start = _fit_history(history, tokenizer, scaffold, allowance)

    # Assemble the kept turns chronologically and encode once, so the returned
    # ids exactly match a single pass (no cross-turn BPE-merge surprises).
    rendered = system_str + "".join(_format_turn(t) for t in history[start:])
    rendered += budget_str + ASSISTANT_OPEN
    ids = bpe.encode(tokenizer, rendered)
    if len(ids) <= allowance:
        return ids
    return _hard_cut(ids, bpe.encode(tokenizer, system_str),
                     bpe.encode(tokenizer, budget_str + ASSISTANT_OPEN), allowance)


def _hard_cut(ids: list[int], head: list[int], tail: list[int], allowance: int) -> list[int]:
    """`ids` brought under `allowance`, keeping both ends of the scaffold.

    Reached only when a single turn is longer than the whole window — dropping
    whole turns cannot help, because the one that busts it is the one that must
    be kept. Until this existed the oversized prompt was returned as it stood
    and `Transformer.forward` raised `position N exceeds context window M`,
    which is a crash where the docstring above promises a truncation. A model
    with a 64-token window and a long question is not an error; it is a model
    that will answer badly, and that is the caller's to see.

    **The trailing `<|assistant|>` is never cut.** It is what asks the model to
    speak, and a prompt missing it is not a short prompt but a different one.
    The system block goes next, from its own front, and only once the body has
    nothing left to give.

    The freshest end of the body survives: it is nearest the question, and it is
    the direction every other eviction here already runs.
    """
    room = allowance - len(head) - len(tail)
    if room > 0:
        body = ids[len(head) : len(ids) - len(tail)]
        return head + body[-room:] + tail
    # No room for the body at all: the scaffold alone is the window. Keep the
    # tail whole and take what is left of the system block from its end, so the
    # part nearest the conversation is the part that survives.
    return head[len(head) - max(0, allowance - len(tail)) :] + tail


def context_floor(history: list[Turn], tokenizer: Tokenizer, ctx_window: int, reserve_for_reply: int) -> int:
    """Index of the oldest turn `build_prompt` would still keep if it were
    fitting the model's own `ctx_window` rather than some multiplier's
    extension of it.

    Read-only, and deliberately conservative rather than exact: `system` and
    the budget marker are ignored, both being a handful of tokens next to a
    real conversation, so the floor this returns is never later than the one a
    real turn would land on. This is the boundary `/prune` (the frontend
    command) snapshots — see `docs/frontend/README.md` — so a chat's usable
    range can be pulled back to what the model actually trained on while a
    raised `context_multiplier` keeps the room past it free for new growth,
    instead of that same old range being included again the moment it still
    fits the wider budget.

    Snapped past a lone `Role.TOOL` turn: dropping the assistant turn that
    called it but keeping the answer would leave a `<|tool-result|>` with
    nothing before it naming what it is an answer to — see
    `docs/llm-engine/README.md`'s note on where a tool's answer is stored.
    """
    allowance = max(0, ctx_window - reserve_for_reply)
    start = _fit_history(history, tokenizer, 0, allowance)
    if start < len(history) and history[start].role is Role.TOOL:
        start += 1
    return start


def training_segments(
    system: str | None,
    turns: list[Turn],
    budget: ThinkBudget | None = None,
    reasoning: Reasoning = Reasoning.KEEP,
) -> list[tuple[str, bool]]:
    """The conversation as `(text, is_target)` pieces, before any encoding.

    Split out from `build_training_example` so a caller with thousands of
    conversations can build every piece first and hand them to the tokenizer in
    one batch — see `encode_examples`. Nothing here touches a tokenizer, which is
    what makes that possible.

    The conversation is *cut at the markers* rather than encoded in one pass, and
    that is a correctness requirement, not a convenience. The pre-tokenizer only
    splits words on spaces, so a marker glues to the text next to it inside one
    "word" (`<|assistant|>Hello` is a single word) and there is no clean token
    boundary at a marker in a whole-string encoding. Splitting at the marker
    boundaries makes the reply tokens stand alone — exactly how the model sees
    them at generation time — and makes the mask exact. The cost is that the
    concatenated tokenization can differ by ~1 token at each seam.
    """
    segments: list[tuple[str, bool]] = []

    if block := _system_block(system):
        segments.append((block, False))

    for turn in turns:
        if turn.role is Role.USER:
            segments.append((f"<|user|>{turn.text.strip()}<|/user|>", False))
            continue

        # Untrained for the same reason the budget marker is: a tool result is
        # something the caller hands the model, not something the model decides.
        # Training it would teach the model to invent search results.
        if turn.role is Role.TOOL:
            segments.append((tool_block(turn), False))
            continue

        # The opening marker is the cue the model conditions on (not a target);
        # the reply body and the closing marker are the target — the model must
        # learn both to answer and to stop.
        #
        # The budget marker is part of that cue, and untrained for the same
        # reason: it is an *instruction* the caller supplies, never something the
        # model chooses. Training it as a target would teach the model to predict
        # how hard it is about to think, which is a fact about the request rather
        # than about the reply. Only its embedding row is ever learned.
        if reasoning is Reasoning.KEEP:
            if budget:
                segments.append((budget.marker, False))
            body = turn.text.strip()
        else:
            body = strip_thinking(turn.text)

        segments.append((ASSISTANT_OPEN, False))
        segments.append((body, True))
        segments.append((ASSISTANT_CLOSE, True))

    return segments


def assemble_example(
    encoded: list[list[int]], flags: list[bool]
) -> tuple[list[int], list[int]]:
    """One conversation's ids and loss mask, from its already-encoded segments.

    `mask[i] == 1` iff the token predicted from position `i` (that is, token
    `i+1`) is part of an assistant reply: its content through the closing
    `<|/assistant|>` marker, inclusive. Everything else — the system block, the
    user turns, the opening `<|assistant|>` cue, and the final position (which
    has no next token) — is 0.
    """
    ids: list[int] = []
    target: list[bool] = []
    for tokens, is_target in zip(encoded, flags):
        target.extend([is_target] * len(tokens))
        ids.extend(tokens)

    # Shift the per-token target flag onto prediction positions: position i is
    # trained iff the token it predicts (i+1) is a target. The last position has
    # no successor, so it is never trained.
    mask = [0] * len(ids)
    for i in range(len(ids) - 1):
        if target[i + 1]:
            mask[i] = 1

    return ids, mask


def build_training_example(
    system: str | None,
    turns: list[Turn],
    tokenizer: Tokenizer,
    budget: ThinkBudget | None = None,
    reasoning: Reasoning = Reasoning.KEEP,
) -> tuple[list[int], list[int]]:
    """Render a full training example — the conversation *including* the
    assistant replies — into token ids plus a parallel loss mask.

    One conversation at a time. `encode_examples` is the same thing for many, and
    is what the dataset build uses; this stays for the callers that genuinely
    have one (the benchmarks, and anything interactive).
    """
    segments = training_segments(system, turns, budget, reasoning)
    return assemble_example(
        [bpe.encode(tokenizer, text) for text, _ in segments],
        [flag for _, flag in segments],
    )


def encode_examples(
    batch: list[list[tuple[str, bool]]], tokenizer: Tokenizer
) -> list[tuple[list[int], list[int]]]:
    """`assemble_example` for several conversations, on every core.

    Every segment of every conversation goes to the tokenizer in one call, and
    the results are cut back apart in order. The library parallelizes a batch
    across cores; encoding segment by segment used one of this machine's six, and
    rendering the chat sets is minutes of a fine-tune during which the training
    figures have nothing to show.

    Identical output to calling `build_training_example` per conversation — the
    batch is a scheduling detail, and each segment is still encoded on its own,
    which is the property the mask depends on.
    """
    flat = [text for segments in batch for text, _ in segments]
    encoded = bpe.encode_batch(tokenizer, flat)

    examples = []
    at = 0
    for segments in batch:
        take = len(segments)
        examples.append(
            assemble_example(encoded[at : at + take], [flag for _, flag in segments])
        )
        at += take
    return examples
