#!/usr/bin/env python3
"""
Clean PG-19 books and re-emit them with the project's document markup
(the same vocabulary the frontend parser / existing uploads use:
<|heading 1|>, <|heading 2|>, <|paragraph|>, <|quote|>, <|divider|>,
<|italic|>, <|endoftext|>), then drop them into the backend uploads dir
as plain .txt files "as if uploaded".

Usage:
    python3 prepare_pg19.py --split test --limit 100
"""

import argparse
import csv
import os
import re
import sys
import uuid
import json

SRC_ROOT = "/home/x_x/Programming/LLM/pg19"
METADATA = "/home/x_x/Programming/LLM/pg19_raw/metadata.csv"
OUT_DIR = "/home/x_x/Programming/LLM/backend/data/raw/uploads"

# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

# Lines that mark Gutenberg / proofreader credits and transcriber notes.
BOILERPLATE_LINE_RE = re.compile(
    r"(project\s+gutenberg|gutenberg\.(org|net)|pgdp\.net|distributed\s+proofread|"
    r"online\s+distributed|produced\s+by|e-?text\s+prepared|prepared\s+by|"
    r"transcrib|proofreading\s+team|html\s+version\s+of\s+this|file\s+was\s+produced)",
    re.IGNORECASE,
)

# A leading artifact like "ST. PAUL***" left over from a stripped header.
LEADING_STARS_RE = re.compile(r"\*{2,}\s*$")

# Scene/section dividers: "* * *", "*****", "-----", etc.
DIVIDER_RE = re.compile(r"^\s*([\*\-\=_#~]\s*){3,}$")

# Roman-numeral chapter markers: "I.", "XVIII. THE TITLE"
ROMAN_RE = re.compile(r"^[IVXLCDM]+\.?(\s|$)", re.IGNORECASE)

# Explicit structural keywords.
HEADING_KW_RE = re.compile(
    r"^(chapter|chap\.|book|part|section|volume|canto|act|scene|epilogue|"
    r"prologue|preface|introduction|contents|appendix|letter)\b",
    re.IGNORECASE,
)

# Gutenberg-style emphasis: _italic_
ITALIC_RE = re.compile(r"_([^_\n]+)_")


def strip_boilerplate(text: str) -> str:
    """Drop leading credit blocks, transcriber notes, and trailing footers."""
    lines = text.split("\n")

    # Remove any line that is purely a boilerplate credit, plus the line that
    # immediately continues it (these credits are usually 1-3 wrapped lines).
    cleaned = []
    skip_continuation = 0
    for line in lines:
        if skip_continuation > 0 and line.strip():
            skip_continuation -= 1
            continue
        skip_continuation = 0
        if BOILERPLATE_LINE_RE.search(line):
            # Skip this line and up to 2 following non-blank continuation lines.
            skip_continuation = 2
            continue
        cleaned.append(line)

    # Strip a leading "...***" artifact line if present near the very top.
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)
    if cleaned and LEADING_STARS_RE.search(cleaned[0]):
        cleaned.pop(0)

    return "\n".join(cleaned)


def split_blocks(text: str):
    """Split into paragraph blocks on blank lines."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip("\n") for b in blocks if b.strip()]


def is_heading(block: str) -> bool:
    """A short, standalone, caps-y or keyword/numeral line is a heading."""
    joined = " ".join(line.strip() for line in block.split("\n"))
    if not joined or len(joined) > 120:
        return False
    if HEADING_KW_RE.match(joined) or ROMAN_RE.match(joined):
        return True
    letters = [c for c in joined if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return True
    return False


def is_blockquote(block: str) -> bool:
    """Consistently indented blocks read as quoted/verse material."""
    lines = [l for l in block.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    return all(l.startswith("    ") or l.startswith("\t") for l in lines)


def unwrap(block: str) -> str:
    """Join hard-wrapped lines of a paragraph into a single line."""
    return " ".join(line.strip() for line in block.split("\n") if line.strip())


def inline_format(text: str) -> str:
    return ITALIC_RE.sub(r"<|italic|>\1<|/italic|>", text)


# Markup uses the backend tokenizer's protected vocabulary (SPECIAL_TOKENS in
# backend/src/tokenize/src/pretokenize.rs): the book title is <|title|>, chapter
# headings are <|heading|>, paragraphs are the paired <|par|>…<|/par|>, and
# <|divider|>/<|italic|> are emitted verbatim. Anything not in that list would be
# shredded by pre-tokenization, so these names must stay in sync with the crate.
def render_book(text: str, title: str) -> str:
    text = strip_boilerplate(text)
    out = []
    if title:
        out.append(f"<|title|>{title}<|/title|>")

    for block in split_blocks(text):
        if DIVIDER_RE.match(block.strip()):
            out.append("<|divider|>")
        elif is_heading(block):
            out.append(f"<|heading|>{unwrap(block)}<|/heading|>")
        elif is_blockquote(block):
            out.append(f"<|quote|>{inline_format(unwrap(block))}<|/quote|>")
        else:
            out.append(f"<|par|>{inline_format(unwrap(block))}<|/par|>")

    out.append("<|endoftext|>")
    return "\n\n".join(out) + "\n"


def load_titles() -> dict:
    titles = {}
    with open(METADATA, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                titles[row[0]] = row[1].strip()
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    titles = load_titles()
    src_dir = os.path.join(SRC_ROOT, args.split)
    files = sorted(os.listdir(src_dir))[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for name in files:
        book_id = os.path.splitext(name)[0]
        title = titles.get(book_id, "")
        with open(os.path.join(src_dir, name), encoding="utf-8") as f:
            raw = f.read()

        rendered = render_book(raw, title)
        out_name = f"{uuid.uuid4()}.txt"
        with open(os.path.join(args.out, out_name), "w", encoding="utf-8") as f:
            f.write(rendered)

        manifest.append({"file": out_name, "pg19_id": book_id, "title": title})

    manifest_path = os.path.join(args.out, f"_pg19_{args.split}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(manifest)} cleaned books -> {args.out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
