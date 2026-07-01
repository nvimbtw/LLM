#!/usr/bin/env python3
"""
Normalize legacy document markup to the backend tokenizer's protected
vocabulary (SPECIAL_TOKENS in backend/src/tokenize/src/pretokenize.rs).

Legacy/frontend markup used names and spaced/attributed tags that the
pre-tokenizer does not protect (e.g. <|paragraph|>, <|heading 1|>,
<|code block | lang='..'|>), so they got shredded into characters during
tokenization. This rewrites them in place to the protected names.

Usage:
    python3 normalize_markup.py <file_or_dir> [<file_or_dir> ...]
"""

import os
import re
import sys

# Simple 1:1 string swaps. The leading '/' in close tags means an open-tag
# substring never appears inside its close tag, so order is irrelevant here.
SWAPS = [
    ("<|paragraph|>", "<|par|>"),
    ("<|/paragraph|>", "<|/par|>"),
    ("<|heading 1|>", "<|title|>"),
    ("<|/heading 1|>", "<|/title|>"),
    ("<|heading 2|>", "<|heading|>"),
    ("<|/heading 2|>", "<|/heading|>"),
    ("<|code inline|>", "<|code|>"),
    ("<|/code inline|>", "<|/code|>"),
    ("<|/code block|>", "<|/codeblock|>"),
]

# <|code block|> may carry attributes (e.g. | lang='text'); collapse any
# variant to the bare protected token.
CODEBLOCK_OPEN_RE = re.compile(r"<\|code block\b[^>]*\|>")

# Tokens that are already valid backend tokens and pass through unchanged:
# <|endoftext|>, <|quote|>/<|/quote|>, <|divider|>, <|italic|>/<|/italic|>,
# <|bold|>/<|/bold|>.


def normalize(text: str) -> str:
    text = CODEBLOCK_OPEN_RE.sub("<|codeblock|>", text)
    for old, new in SWAPS:
        text = text.replace(old, new)
    return text


def iter_files(targets):
    for t in targets:
        if os.path.isdir(t):
            for name in sorted(os.listdir(t)):
                if name.endswith(".txt"):
                    yield os.path.join(t, name)
        elif os.path.isfile(t) and t.endswith(".txt"):
            yield t


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    count = 0
    for path in iter_files(sys.argv[1:]):
        with open(path, encoding="utf-8") as f:
            original = f.read()
        normalized = normalize(original)
        if normalized != original:
            with open(path, "w", encoding="utf-8") as f:
                f.write(normalized)
            count += 1
    print(f"normalized {count} file(s)")


if __name__ == "__main__":
    main()
