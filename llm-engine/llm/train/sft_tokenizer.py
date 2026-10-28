"""Data-only SFT tokenizer derivation with exact saved identity and fixed reserved-token IDs."""

import hashlib
from uuid import uuid5

from tokenizers import Tokenizer

from llm.architecture.spec import FinalTokenizer
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.corpus import relabel, special


def marker_history(metadata: dict, vocabulary_size: int) -> list[int] | None:
    """Read complete known marker provenance; incomplete/unknown history cannot authorize resetting old token rows."""
    result = metadata.get("result")
    dataset = result.get("dataset") if isinstance(result, dict) else None
    if not isinstance(dataset, dict) or dataset.get("marker_history_complete") is False:
        return None
    seen = dataset.get("special_ids_seen")
    if seen is None:
        return None
    if (not isinstance(seen, list) or len(seen) > 65536
            or any(type(value) is not int or not 0 <= value < vocabulary_size for value in seen)):
        raise ValueError("saved SFT marker history has invalid token IDs")
    return sorted(set(seen))


def prepare_tokenizer(raw: bytes, binding: FinalTokenizer, used_ids: list[int]) -> tuple[bytes, FinalTokenizer]:
    """Verify saved bytes and privately adopt memory labels only when their fixed IDs occur in SFT data.

    Unchanged vocabularies retain their exact bytes and identity. A derived
    vocabulary has a deterministic identity and records the exact source hash;
    existing token IDs, merges and caller-owned artifacts remain unchanged.
    This performs no filesystem/catalog mutation or model execution.
    """
    actual = binding_from_bytes(binding.tokenizer_id, raw, tuple(binding.reserved_token_ids))
    if actual.model_copy(update={"derived_from": binding.derived_from}) != binding:
        raise ValueError("SFT tokenizer bytes differ from the saved definition")
    if (len(used_ids) > 65536 or any(type(token_id) is not int or not 0 <= token_id < binding.vocabulary_size
                                    for token_id in used_ids)):
        raise ValueError("SFT used-token IDs must be bounded integers within the saved vocabulary")
    memory_slots = {special.SPECIAL_TOKENS.index(marker) for marker in (special.MEMORY_OPEN, special.MEMORY_CLOSE)}
    if not memory_slots.intersection(used_ids):
        return raw, binding
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    adopted = relabel.adopt_memory_slots(tokenizer)
    if adopted is tokenizer:
        return raw, binding
    derived_raw = adopted.to_str().encode("utf-8")
    digest = hashlib.sha256(derived_raw).hexdigest()
    identity = uuid5(binding.tokenizer_id, f"slate-sft-memory-labels-v1:{digest}")
    derived = binding_from_bytes(identity, derived_raw, tuple(binding.reserved_token_ids))
    if derived.vocabulary_size != binding.vocabulary_size or derived.reserved_token_ids != binding.reserved_token_ids:
        raise ValueError("SFT marker adoption changed immutable vocabulary facts")
    return derived_raw, derived.model_copy(update={"derived_from": binding.fingerprint})
