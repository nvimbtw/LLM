"""The protected vocabulary: markers that must survive tokenization as single
tokens rather than being split into byte merges.

Ported from backend/crates/tokenize/src/pretokenize.rs. Only markers something
actually emits are listed — the chat scaffold, the document markup the frontend
renders, and the thinking block.
"""

# Order is load-bearing: it fixes each marker's id. Append only, never reorder,
# and take a reserved slot rather than inserting in the middle.
SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|par|>",
    "<|/par|>",
    "<|code|>",
    "<|/code|>",
    "<|user|>",
    "<|/user|>",
    "<|assistant|>",
    "<|/assistant|>",
    "<|system|>",
    "<|/system|>",
    "<|title|>",
    "<|/title|>",
    "<|heading|>",
    "<|/heading|>",
    "<|subheading|>",
    "<|/subheading|>",
    "<|codeblock|>",
    "<|/codeblock|>",
    "<|link|>",
    "<|/link|>",
    "<|list|>",
    "<|/list|>",
    "<|item|>",
    "<|/item|>",
    "<|quote|>",
    "<|/quote|>",
    "<|divider|>",
    "<|italic|>",
    "<|/italic|>",
    "<|bold|>",
    "<|/bold|>",
    # Paired thinking block: `<|thinking|>...reasoning...<|/thinking|>`. Both
    # delimiters are protected; the reasoning between them is ordinary text.
    "<|thinking|>",
    "<|/thinking|>",
    "<|warn|>",
    "<|/warn|>",
    # Reply-reference. Its content is a message id, not prose.
    "<|reply|>",
    "<|/reply|>",
    # How long the assistant should reason for, as one token sitting immediately
    # before the <|assistant|> it governs. Unpaired and input-only: the model
    # conditions on one of these and is never trained to emit it, so only the
    # embedding row has to be learned.
    "<|think:low|>",
    "<|think:med|>",
    "<|think:high|>",
    # Where a tool's answer is handed back, between the assistant turn that
    # asked and the one that uses it. Input-only: it is absent from
    # `markup.PAIRS`, so the grammar forbids the model from emitting one and
    # claiming to have been told something it was not.
    "<|tool-result|>",
    "<|/tool-result|>",
    # Retired, and deliberately still here. Web search was called through its
    # own marker pair before tool calls were standardised on `<|tool-call|>`
    # below. The slots stay occupied because an id is the one thing that may
    # never move: a vocabulary saved while these were named still carries them,
    # and `corpus.relabel` refuses a slot holding a marker it did not expect.
    # Nothing emits them — they are in no table in `chat.markup`, so the grammar
    # forbids the model from writing one.
    "<|web-search|>",
    "<|/web-search|>",
    # Every tool call, whatever the tool. The body is a JSON object naming the
    # tool and its argument (`chat.tools`), which is what makes the tool table
    # a property of the *build* rather than of the vocabulary: a model trained
    # today can be served a tool invented next month without its tokenizer
    # knowing anything about it.
    "<|tool-call|>",
    "<|/tool-call|>",
]

# The document separator, baked between documents in the packed corpus.
END_OF_TEXT = "<|endoftext|>"

# The block is FIXED and deliberately larger than the marker list.
#
# Special tokens take the lowest ids, so if the block were sized to the list then
# adding one marker would shift every id after it — silently invalidating every
# corpus and every model already trained against the old numbering. Padding to a
# constant means a new marker consumes a reserved slot and shifts nothing.
# Raising this number is itself a vocabulary change.
SPECIAL_BLOCK = 64

assert len(SPECIAL_TOKENS) <= SPECIAL_BLOCK, "more markers than reserved slots"


def placeholder(index: int) -> str:
    return f"<|reserved_{index}|>"


def block_names() -> list[str]:
    """Every name a slot in the reserved block might be carrying: the markers as
    they are spelled now, plus the placeholder spelling of every slot.

    Both, because a vocabulary trained before a marker was named still holds
    `<|reserved_N|>` in that slot. Anything asking "which ids are markers?" has
    to recognize either spelling — miss the placeholders and those slots fall
    out of the range the grammar masks, which lets a model emit a placeholder as
    though it were ordinary text.
    """
    return SPECIAL_TOKENS + [placeholder(i) for i in range(SPECIAL_BLOCK)]


def token_list() -> list[str]:
    """Every special token, padded with placeholders to exactly SPECIAL_BLOCK.

    This is what the tokenizer trains against, so the block is reserved whether
    or not the markers are used yet.
    """
    reserved = [placeholder(i) for i in range(len(SPECIAL_TOKENS), SPECIAL_BLOCK)]
    return SPECIAL_TOKENS + reserved
