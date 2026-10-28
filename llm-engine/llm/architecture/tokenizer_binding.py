"""Authoritative tokenizer facts from bounded data-only artifact bytes, without model construction."""

import hashlib
import re
from uuid import UUID

from tokenizers import Tokenizer

from llm.architecture.spec import FinalTokenizer, Recipe, Repeat, Stack

TOKENIZER_BYTES = 64 << 20


def reserved_names(recipes: tuple[Recipe, ...]) -> tuple[str, ...]:
    """Collect full reserved-ID names used by frozen constructor bindings without expanding repeated calls."""
    names = set()
    for recipe in recipes:
        declarations = list(recipe.modules.values())
        pending = list(recipe.flow)
        while pending:
            operation = pending.pop()
            if isinstance(operation, Repeat):
                pending.extend(operation.body)
            elif isinstance(operation, Stack):
                declarations.append(operation.template)
        for declaration in declarations:
            for value in declaration.parameters.values():
                if isinstance(value, str) and (match := re.fullmatch(r"\$\{tokenizer\.([a-z][a-z0-9_]*_id)\}", value)):
                    names.add(match[1])
    return tuple(sorted(names))


def binding_from_bytes(tokenizer_id: UUID, raw: bytes, reserved_names: tuple[str,...] = ()) -> FinalTokenizer:
    """Read exact vocabulary/hash and requested reserved marker IDs; unsupported names fail instead of accepting client guesses."""
    if len(raw) > TOKENIZER_BYTES:
        raise ValueError("tokenizer artifact exceeds byte budget")
    try:
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    except Exception as error:
        raise ValueError("invalid tokenizer artifact") from error
    vocabulary_size = tokenizer.get_vocab_size()
    token_ids = set(tokenizer.get_vocab().values())
    if not token_ids or len(token_ids) != vocabulary_size or min(token_ids) != 0 or max(token_ids) != vocabulary_size-1:
        raise ValueError("tokenizer IDs must form the owned contiguous vocabulary")
    reserved = {}
    for name in reserved_names:
        if not name.endswith("_id"):
            raise ValueError("reserved binding names must identify an engine marker with an _id suffix")
        marker = "<|endoftext|>" if name == "pad_id" else f"<|{name[:-3]}|>"
        token_id = tokenizer.token_to_id(marker)
        if token_id is None:
            raise ValueError("reserved marker is unavailable in the selected tokenizer")
        reserved[name] = token_id
    return FinalTokenizer(kind="final",tokenizer_id=tokenizer_id,fingerprint=hashlib.sha256(raw).hexdigest(),
                          vocabulary_size=vocabulary_size,reserved_token_ids=reserved)
