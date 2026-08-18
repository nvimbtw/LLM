"""The marker grammar, enforced while the model decodes.

A reply is markup: `<|par|>`, `<|list|><|item|>`, `<|thinking|>`. A model this
size cannot keep it well-formed on its own, and served replies show it — closers
naming blocks that are not open, turn boundaries invented mid-reply, blocks left
unclosed. None of that is a knowledge problem the model could study its way out
of; it is a *representable* state, and the fix is to stop representing it.

`OpenBlocks` tracks what is open and masks the markers that would be illegal
next, so the sampler can only draw from well-formed continuations. Ordinary text
is never masked, so there is always something to draw.

**This is a serving-time constraint and must stay out of eval.** The markup suite
exists to measure whether the model has *learned* the grammar; generating under
the constraint would measure the constraint instead and score every model alike.

Ported from backend/crates/llm/src/markup.rs.
"""

from dataclasses import dataclass, field

import torch
from tokenizers import Tokenizer

from llm.chat import tools
from llm.corpus import special

# How deep blocks may nest before further openers are refused.
#
# Not a property of the data — the deepest rendering in the SFT corpus is
# list > item > bold, three — but a bound on a degenerate model that has learned
# to open and never close. Without it a reply can spend its whole token budget
# descending, and the drain at the end would be longer than the reply.
MAX_DEPTH = 6

# The document-markup pairs the model is allowed to use. Anything in
# SPECIAL_TOKENS not named here, not standalone and not a tool in `chat.tools`
# is forbidden, so adding a marker fails closed: the model cannot emit it until
# one of those tables says how it nests. `<|tool-result|>` is deliberately in
# none of them — the model reads results, it does not write them.
PAIRS: list[tuple[str, str]] = [
    ("<|par|>", "<|/par|>"),
    ("<|code|>", "<|/code|>"),
    ("<|codeblock|>", "<|/codeblock|>"),
    ("<|title|>", "<|/title|>"),
    ("<|heading|>", "<|/heading|>"),
    ("<|subheading|>", "<|/subheading|>"),
    ("<|link|>", "<|/link|>"),
    ("<|list|>", "<|/list|>"),
    ("<|item|>", "<|/item|>"),
    ("<|quote|>", "<|/quote|>"),
    ("<|italic|>", "<|/italic|>"),
    ("<|bold|>", "<|/bold|>"),
    ("<|warn|>", "<|/warn|>"),
    ("<|thinking|>", "<|/thinking|>"),
]

STANDALONE = ["<|divider|>"]


def _resolvable_tools(tokenizer: Tokenizer) -> list[tuple[tools.Tool, int, int]]:
    """Each tool this vocabulary can spell, with its opener and closer ids.

    A tool whose markers are absent is dropped rather than raised on, which is
    the one place tools differ from document markup. A model trained before a
    tool existed carries a tokenizer that predates it; that model cannot call
    the tool, but it can still serve — whereas a missing `<|par|>` means a
    vocabulary that was never built for this repo at all.
    """
    resolved = []
    for tool in tools.TOOLS:
        open_id = tokenizer.token_to_id(tool.open)
        close_id = tokenizer.token_to_id(tool.close)
        if open_id is not None and close_id is not None:
            resolved.append((tool, open_id, close_id))
    return resolved


@dataclass(frozen=True)
class Rules:
    """The marker classification, derived once from the tokenizer's vocabulary."""

    closer_for: dict[int, int]
    opener_for: dict[int, int]
    # Markers the model must never emit: the other roles' scaffold, the budget
    # cues it only ever conditions on, and <|reply|>, whose content is a message
    # id rather than prose.
    forbidden: frozenset[int]
    # An opener only legal directly inside a particular other block.
    requires_parent: dict[int, int]
    # An open block admitting only one kind of opener directly inside it.
    only_child: dict[int, int]
    # Blocks whose content is not markup — a tool call's argument is a query
    # string — so no opener is legal inside them.
    leaf: frozenset[int]
    # Openers legal only outside every block. The mirror of `requires_parent`:
    # that one demands a specific parent, this one demands none.
    top_level: frozenset[int]
    # Opener and closer id -> the tool whose call they bracket. The generation
    # loop needs both: the opener to start the span, the closer to end it and
    # name what was called.
    tool_open_for: dict[int, str]
    tool_close_for: dict[int, str]
    marker_ids: tuple[int, ...]
    end_of_assistant: int
    end_of_text: int

    def is_special(self, token_id: int) -> bool:
        return bool(self.marker_ids) and self.marker_ids[0] <= token_id <= self.marker_ids[-1]

    @classmethod
    def build(cls, tokenizer: Tokenizer) -> "Rules":
        def marker_id(marker: str) -> int:
            tid = tokenizer.token_to_id(marker)
            if tid is None:
                # A marker missing from the vocabulary would leave that block
                # unconstrained. There is nothing to recover to at this layer.
                raise ValueError(f"unknown marker {marker!r} — retrain the tokenizer")
            return tid

        closer_for: dict[int, int] = {}
        opener_for: dict[int, int] = {}
        for opener, closer in PAIRS:
            open_id, close_id = marker_id(opener), marker_id(closer)
            closer_for[open_id] = close_id
            opener_for[close_id] = open_id

        callable_tools = _resolvable_tools(tokenizer)
        for _, open_id, close_id in callable_tools:
            closer_for[open_id] = close_id
            opener_for[close_id] = open_id

        named = set(closer_for) | set(closer_for.values())
        named |= {marker_id(m) for m in STANDALONE}

        end_of_assistant = marker_id("<|/assistant|>")
        end_of_text = marker_id(special.END_OF_TEXT)

        # Both spellings of every slot — see `special.block_names`. A vocabulary
        # older than a marker still calls that slot `<|reserved_N|>`, and it has
        # to stay inside the masked range either way.
        marker_ids = sorted(
            tokenizer.token_to_id(name)
            for name in special.block_names()
            if tokenizer.token_to_id(name) is not None
        )

        # Everything the table does not account for. Fails closed by
        # construction: a marker added to SPECIAL_TOKENS and not to PAIRS lands
        # here and the model may never emit it.
        forbidden = frozenset(
            tid
            for tid in marker_ids
            if tid not in named and tid not in (end_of_assistant, end_of_text)
        )

        tool_openers = frozenset(open_id for _, open_id, _ in callable_tools)
        return cls(
            closer_for=closer_for,
            opener_for=opener_for,
            forbidden=forbidden,
            requires_parent={marker_id("<|item|>"): marker_id("<|list|>")},
            only_child={marker_id("<|list|>"): marker_id("<|item|>")},
            leaf=tool_openers,
            top_level=tool_openers,
            tool_open_for={open_id: tool.name for tool, open_id, _ in callable_tools},
            tool_close_for={close: tool.name for tool, _, close in callable_tools},
            marker_ids=tuple(marker_ids),
            end_of_assistant=end_of_assistant,
            end_of_text=end_of_text,
        )


@dataclass
class OpenBlocks:
    """Which marker blocks are open, and therefore which markers may come next.

    One per generated reply. Fed every sampled token through `observe`, asked
    about candidates through `allows`. An empty stack means "the reply may end
    here" — the `<|assistant|>` opener is part of the prompt, not on the stack.
    """

    rules: Rules
    stack: list[int] = field(default_factory=list)

    def allows(self, token_id: int) -> bool:
        """Whether `token_id` is a legal next token.

        True for every ordinary (non-marker) token, unconditionally — which is
        what guarantees `mask` can never empty the candidate set.
        """
        rules = self.rules
        if not rules.is_special(token_id):
            return True
        if token_id in rules.forbidden:
            return False

        # A turn may only end from the outside of every block it opened. This is
        # also what stops a reply stopping mid-block: the model cannot reach the
        # end-of-turn marker without emitting the closers first.
        if token_id in (rules.end_of_assistant, rules.end_of_text):
            return not self.stack

        if (opener := rules.opener_for.get(token_id)) is not None:
            return bool(self.stack) and self.stack[-1] == opener

        if token_id in rules.closer_for:
            return self._may_open(token_id)

        # Standalone markers (<|divider|>) open nothing and close nothing.
        return True

    def _may_open(self, token_id: int) -> bool:
        """An opener is legal when there is depth left for it, the enclosing
        block admits it, and it does not itself demand a different parent."""
        if len(self.stack) >= MAX_DEPTH:
            return False

        parent = self.stack[-1] if self.stack else None

        # A tool call is a statement the turn makes, not a phrase inside one:
        # nested in a <|par|> it would read as prose, and in a <|codeblock|> as
        # a sample. Either way the backend would still execute it.
        if token_id in self.rules.top_level and parent is not None:
            return False

        if parent is not None and parent in self.rules.leaf:
            return False

        # <|item|> outside a list, or anything but <|item|> directly inside one.
        # Enforced from both sides so neither has to trust the other.
        required = self.rules.requires_parent.get(token_id)
        if required is not None and parent != required:
            return False

        only = self.rules.only_child.get(parent) if parent is not None else None
        return token_id == only if only is not None else True

    def observe(self, token_id: int) -> None:
        """Record a token the sampler actually drew.

        A marker `allows` would have refused is ignored rather than applied — the
        stack must keep describing the text even when a caller samples without
        masking first, which is the only way the two could disagree.
        """
        if token_id in self.rules.closer_for:
            if len(self.stack) < MAX_DEPTH:
                self.stack.append(token_id)
            return
        opener = self.rules.opener_for.get(token_id)
        if opener is not None and self.stack and self.stack[-1] == opener:
            self.stack.pop()

    def mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Set every illegal marker's logit to -inf, leaving the rest untouched.

        Only the reserved marker block is scanned — a few dozen ids out of the
        vocabulary — so this costs nothing against the forward pass it follows.
        """
        for token_id in self.rules.marker_ids:
            if token_id < logits.numel() and not self.allows(token_id):
                logits[token_id] = float("-inf")
        return logits

    @property
    def is_complete(self) -> bool:
        """Whether the reply could legally stop here — nothing left open."""
        return not self.stack

    @property
    def depth(self) -> int:
        """How many blocks are open, which is exactly how many closers a drain
        costs. The generation loop reserves this against its token cap."""
        return len(self.stack)

    def closers(self) -> list[int]:
        """The closers that would drain everything still open, innermost first.

        For the token cap, the one stop the grammar cannot prevent: the loop
        appends these so a truncated reply is still well-formed markup rather
        than a half-open block the frontend has to guess at.
        """
        return [self.rules.closer_for[o] for o in reversed(self.stack) if o in self.rules.closer_for]
