"""Deterministic byte-complete vocabulary for concrete builtin architecture examples, with no learned corpus merges."""

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import ByteLevel

from llm.corpus import special


def byte_alphabet() -> Tokenizer:
    """Construct a real byte-level tokenizer with fixed reserved IDs and all 256 bytes; no data acquisition or fitting occurs."""
    markers = special.token_list()
    vocabulary = {marker:index for index,marker in enumerate(markers)}
    vocabulary.update({symbol:index+len(markers) for index,symbol in enumerate(sorted(ByteLevel.alphabet()))})
    tokenizer = ByteLevelBPETokenizer(vocab=vocabulary,merges=[])
    tokenizer.add_special_tokens(markers)
    return Tokenizer.from_str(tokenizer.to_str())
