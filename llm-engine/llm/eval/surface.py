"""Lexical extraction retained for historical exploration scripts."""

def words_in(text: str) -> list[str]:
    """Lowercased alphabetic words, keeping mid-word apostrophes.

    Deliberately marker-unaware — callers strip markers first.
    """
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalpha():
            current.append(ch.lower())
        elif ch in ("'", "’") and current:
            # Only mid-word; a leading quote starts no word and a trailing one is
            # trimmed on flush.
            current.append("'")
        elif current:
            words.append("".join(current).rstrip("'"))
            current = []
    if current:
        words.append("".join(current).rstrip("'"))
    return [w for w in words if w]
