"""Actual private tokenizer-byte derivation and unchanged fixed token IDs for saved-definition SFT."""

import json
from uuid import uuid4

import pytest
from tokenizers import Tokenizer

from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.corpus import relabel, special
from llm.corpus.baseline import byte_alphabet
from llm.train.sft_tokenizer import prepare_tokenizer


def legacy_bytes():
    """Construct a real byte tokenizer with both memory positions still spelled as their original placeholders."""
    document = json.loads(byte_alphabet().to_str())
    renames = [(special.SPECIAL_TOKENS.index(marker), marker,
                special.placeholder(special.SPECIAL_TOKENS.index(marker)))
               for marker in (special.MEMORY_OPEN, special.MEMORY_CLOSE)]
    relabel.apply_renames(document, renames)
    return Tokenizer.from_str(json.dumps(document)).to_str().encode()


def test_private_memory_derivation_preserves_every_other_id_and_exact_source_bytes():
    """A deterministic derived binding changes only the two allowed labels and never the base tokenizer or file."""
    raw = legacy_bytes()
    original = Tokenizer.from_str(raw.decode())
    binding = binding_from_bytes(uuid4(), raw, ("pad_id",))
    derived_raw, derived = prepare_tokenizer(raw, binding, [47, 48])
    assert prepare_tokenizer(raw, binding, [48, 47, 47]) == (derived_raw, derived)
    assert derived.tokenizer_id != binding.tokenizer_id and derived.fingerprint != binding.fingerprint
    assert derived.derived_from == binding.fingerprint
    assert derived.vocabulary_size == binding.vocabulary_size and derived.reserved_token_ids == binding.reserved_token_ids
    after = Tokenizer.from_str(derived_raw.decode())
    for name, token_id in original.get_vocab().items():
        expected = special.SPECIAL_TOKENS[token_id] if token_id in (47, 48) else name
        assert after.token_to_id(expected) == token_id
    assert original.to_str().encode() == raw
    assert prepare_tokenizer(derived_raw, derived, [47, 48]) == (derived_raw, derived)


def test_unused_memory_slots_and_already_named_vocabularies_retain_byte_exact_identity():
    """Ordinary SFT and an already adopted vocabulary keep the saved bytes even if formatting is noncanonical."""
    raw = json.dumps(json.loads(legacy_bytes()), indent=2).encode()
    binding = binding_from_bytes(uuid4(), raw)
    assert prepare_tokenizer(raw, binding, [0, 1, 2]) == (raw, binding)
    raw = json.dumps(json.loads(byte_alphabet().to_str()), indent=2).encode()
    binding = binding_from_bytes(uuid4(), raw)
    assert prepare_tokenizer(raw, binding, [47, 48]) == (raw, binding)


def test_bad_binding_or_slot_collision_cannot_authorize_a_new_tokenizer():
    """Saved hash mismatches, invalid used IDs and foreign names fail without renumbering anything."""
    raw = legacy_bytes()
    binding = binding_from_bytes(uuid4(), raw)
    for invalid in ([True], ["47"], [-1], [binding.vocabulary_size], [0] * 65537):
        with pytest.raises(ValueError, match="bounded integers"):
            prepare_tokenizer(raw, binding, invalid)
    with pytest.raises(ValueError, match="saved definition"):
        prepare_tokenizer(raw, binding.model_copy(update={"fingerprint": "a" * 64}), [47])
    document = json.loads(raw)
    relabel.apply_renames(document, [(47, special.placeholder(47), "foreign-memory-name")])
    collision = Tokenizer.from_str(json.dumps(document)).to_str().encode()
    with pytest.raises(ValueError, match="slot 47"):
        prepare_tokenizer(collision, binding_from_bytes(uuid4(), collision), [47])
