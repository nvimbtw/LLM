"""Renaming reserved slots in a vocabulary that was saved before they were used.

The special block is a fixed 64 slots, padded with `<|reserved_N|>` placeholders
so that naming a new marker consumes a slot instead of shifting every id after
it (see `special.SPECIAL_BLOCK`). The consequence is that a vocabulary saved
before a marker existed already *has* that marker's id — it is just still
spelled as the placeholder.

So adopting a new marker into an old vocabulary is a rename, not a retrain.
Nothing about the numbering moves: a packed `corpus.bin` stays valid, a
checkpoint still loads, and the embedding row is the one that was already there
(untrained, which is what `train.warm_start` exists to handle).

What this deliberately does *not* do is touch the merges, the byte alphabet, or
any slot whose current name is not a placeholder. A slot already carrying a real
marker under a different name is a vocabulary from somewhere else, and renaming
it would silently repoint tokens in every corpus keyed on it.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

from llm.corpus import special, vocab


def adopt_memory_slots(tokenizer: Tokenizer) -> Tokenizer:
    """Return a private copy adopting memory slots when needed, preserving every ID.

    The shared tokenizer and its file remain untouched. Existing non-placeholder
    collisions raise rather than changing the meaning of a trained embedding row.
    """
    markers = (special.MEMORY_OPEN, special.MEMORY_CLOSE)
    slots = [special.SPECIAL_TOKENS.index(marker) for marker in markers]
    if all(tokenizer.token_to_id(marker) == slot for marker, slot in zip(markers, slots)):
        return tokenizer
    document = json.loads(tokenizer.to_str())
    renames = [rename for rename in slot_renames(document) if rename[2] in markers]
    apply_renames(document, renames)
    adopted = Tokenizer.from_str(json.dumps(document, ensure_ascii=False))
    if any(adopted.token_to_id(marker) != slot for marker, slot in zip(markers, slots)):
        raise ValueError("memory context requires the fixed64 reserved-token vocabulary")
    return adopted


def slot_renames(document: dict) -> list[tuple[int, str, str]]:
    """The `(id, from, to)` renames bringing `document` up to the current table.

    Empty when the vocabulary is already current. Raises when a slot holds
    something that is neither the placeholder nor the marker now expected there,
    because that is a vocabulary this repo did not build and guessing at it is
    how a corpus gets silently repointed.
    """
    by_id = {entry["id"]: entry["content"] for entry in document.get("added_tokens", [])}

    renames: list[tuple[int, str, str]] = []
    for slot, marker in enumerate(special.SPECIAL_TOKENS):
        current = by_id.get(slot)
        if current is None or current == marker:
            continue
        if current != special.placeholder(slot):
            raise ValueError(
                f"slot {slot} holds {current!r}, expected {marker!r} or "
                f"{special.placeholder(slot)!r} — not a vocabulary built here"
            )
        renames.append((slot, current, marker))
    return renames


def apply_renames(document: dict, renames: list[tuple[int, str, str]]) -> None:
    """Rewrite `document` in place, in both places a name is recorded.

    `added_tokens` is what makes a marker survive as one token; `model.vocab` is
    what maps the string to the id. Rewriting one and not the other yields a
    tokenizer that encodes the marker but cannot decode it back.
    """
    vocab = document.get("model", {}).get("vocab", {})
    for entry in document.get("added_tokens", []):
        for slot, was, now in renames:
            if entry["id"] == slot:
                entry["content"] = now
                if vocab.pop(was, None) is not None:
                    vocab[now] = slot


def relabel_file(path: Path, dry_run: bool) -> list[tuple[int, str, str]]:
    """Bring one saved `tokenizer.json` up to the current marker table."""
    document = json.loads(path.read_text(encoding="utf-8"))
    renames = slot_renames(document)
    if renames and not dry_run:
        apply_renames(document, renames)
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return renames


def refresh_build_sha(path: Path, dry_run: bool) -> bool:
    """Restamp a tokenizer build's `special_tokens_sha`, if it has gone stale.

    Without this the relabelled vocabulary still fails `vocab.matches()` and is
    rebuilt on next use — which re-keys every corpus built from it, for a table
    the file on disk now satisfies exactly.
    """
    if not path.is_file():
        return False

    record = json.loads(path.read_text(encoding="utf-8"))
    current = vocab.special_tokens_sha()
    if record.get("special_tokens_sha") == current:
        return False
    if not dry_run:
        record["special_tokens_sha"] = current
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return True
