#!/usr/bin/env python3
"""
Clean PG-19 books into plain prose and drop them into the backend uploads dir
as .txt files "as if uploaded".

No document markup is injected — pretraining should see ordinary book text,
not tags. The only special token emitted is a single trailing
<|endoftext|> per file, which the Rust pretrain builder
(backend/src/training.rs: build_dataset) relies on as the document boundary
when it concatenates files; everything else (titles, headings, quotes,
italics, dividers) is rendered as plain text.

Cleaning also drops blocks that look like OCR/scan noise rather than real
prose: table-of-contents dot-leaders, running-head page numbers, and mangled
numbering like "ii..i.i.iii1.1.1.1.1." that would otherwise teach the model
to produce junk instead of words.

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

# Scene/section dividers: "* * *", "*****", "-----", etc. Dropped outright —
# the paragraph gap around them already reads as a break in plain text.
DIVIDER_RE = re.compile(r"^\s*([\*\-\=_#~]\s*){3,}$")

# Roman-numeral chapter markers: "I.", "XVIII. THE TITLE"
ROMAN_RE = re.compile(r"^[IVXLCDM]+\.?(\s|$)", re.IGNORECASE)

# Explicit structural keywords.
HEADING_KW_RE = re.compile(
    r"^(chapter|chap\.|book|part|section|volume|canto|act|scene|epilogue|"
    r"prologue|preface|introduction|contents|appendix|letter)\b",
    re.IGNORECASE,
)

# Gutenberg-style emphasis markup: _italic_. We keep the word, drop the
# underscores — there's no <|italic|> tag in plain pretrain text.
ITALIC_RE = re.compile(r"_([^_\n]+)_")

# ---------------------------------------------------------------------------
# Noise / low-quality block filtering
# ---------------------------------------------------------------------------

# A short block made up of nothing but roman numerals, digits, dots, dashes,
# and whitespace: OCR'd running heads and page numbers, or mangled numbering
# like "ii..i.i.iii1.1.1.1.1.".
NOISE_ONLY_RE = re.compile(r"^[ivxlcdmIVXLCDM0-9.\-\s]+$")

# Table-of-contents dot-leader lines: "The Beginning . . . . . . . 12"
TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d*\s*$")

# Scan placeholder markers, standalone ("[Illustration]", "[Illustration: a
# map of the coast]") or fused with a paragraph's OCR'd drop-cap ("[Illustration:
# L]ITTLE girls..." really means the paragraph starts with "LITTLE"). Captures
# the drop-cap letter, if any, so it isn't lost when the marker is dropped.
ILLUSTRATION_RE = re.compile(r"\[\s*illustration\s*:?\s*([A-Za-z])?[^\]]*\]", re.IGNORECASE)

# ASCII box-drawn ads/tables ("+----+ | PATENT BINDERS | +----+"), common in
# periodical issues (Punchinello, magazines): a high fraction of border
# characters means it's layout art, not prose.
BORDER_CHARS = set("+-|=_~")
BORDER_RATIO_THRESHOLD = 0.15


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


def is_toc_block(block: str) -> bool:
    """A block where most lines end in a dot-leader is a contents listing."""
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for l in lines if TOC_LEADER_RE.search(l))
    return hits / len(lines) >= 0.5


def is_noise_block(block: str) -> bool:
    """OCR/scan garbage: page numbers, running heads, mangled numbering."""
    stripped = block.strip()
    if len(stripped) <= 40 and NOISE_ONLY_RE.match(stripped):
        return True
    non_space = len(re.sub(r"\s", "", stripped))
    if non_space == 0:
        return True
    border = sum(1 for c in stripped if c in BORDER_CHARS)
    if border / non_space >= BORDER_RATIO_THRESHOLD:
        return True
    letters = sum(c.isalpha() for c in stripped)
    if len(stripped) <= 60 and letters / non_space < 0.5:
        return True
    words = stripped.split()
    if len(words) >= 3:
        wordish = sum(1 for w in words if re.search(r"[A-Za-z]{2,}", w))
        if wordish / len(words) < 0.4:
            return True
    return False


def collapse_spaces(s: str) -> str:
    return re.sub(r" {2,}", " ", s)


def unwrap(block: str) -> str:
    """Join hard-wrapped lines of a paragraph into a single clean line."""
    joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
    return collapse_spaces(joined)


def clean_verse(block: str) -> str:
    """Trim a quoted/verse block line-by-line but keep its line breaks."""
    lines = [collapse_spaces(l.strip()) for l in block.split("\n") if l.strip()]
    return "\n".join(lines)


def inline_format(text: str) -> str:
    text = ITALIC_RE.sub(r"\1", text)
    text = ILLUSTRATION_RE.sub(lambda m: m.group(1) or "", text)
    return collapse_spaces(text).strip()


def render_book(text: str, title: str) -> str:
    text = strip_boilerplate(text)
    out = []
    if title:
        out.append(title.strip())

    for block in split_blocks(text):
        if DIVIDER_RE.match(block.strip()):
            continue
        if is_heading(block):
            formatted = inline_format(unwrap(block))
        elif is_toc_block(block) or is_noise_block(block):
            continue
        elif is_blockquote(block):
            formatted = inline_format(clean_verse(block))
        else:
            formatted = inline_format(unwrap(block))
        if formatted:
            out.append(formatted)

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
