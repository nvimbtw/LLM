#!/usr/bin/env python3
"""
Migrate existing SFT conversations to the bare-tag markup convention.

`<|codeblock|>` and `<|link|>` used to carry attributes (`lang='python'`,
`href='...'`), but those attribute forms don't tokenize to a single reserved id
— they byte-shred into raw markup (see SPECIAL_TOKENS in
backend/src/tokenize/src/pretokenize.rs). The convention is now BARE tags:

    <|codeblock|lang='python'|>…<|/codeblock|>   ->  <|codeblock|>…<|/codeblock|>
    <|link|href='URL'|>text<|/link|>             ->  <|link|>URL<|/link|>

For links the destination moves into the visible text (the frontend renders the
text as the href now — see rich-message.tsx). If a link had no href, its existing
text is kept as-is.

Only assistant-turn text is touched; each line is parsed as JSON, transformed,
and re-serialised so structure and user turns are preserved exactly.

Usage:
    python3 scripts/migrate_sft_bare_tags.py            # dry run: report only
    python3 scripts/migrate_sft_bare_tags.py --apply    # rewrite in place (+ .bak)
    python3 scripts/migrate_sft_bare_tags.py --dir path/to/sft
"""

import paths
import argparse
import glob
import json
import os
import re
import sys

DEFAULT_DIR = paths.sft_dir()

# An opening codeblock tag with any attributes -> bare. `\|+>` also swallows the
# occasional malformed `||>` seen in older data. Bare `<|codeblock|>` won't match
# (nothing between `codeblock|` and the closing `|>`), so it's left untouched.
CODEBLOCK_ATTR_RE = re.compile(r"<\|codeblock\|[^>]*?\|+>")

# A full link element carrying attributes: capture the attrs and the inner text.
LINK_ATTR_RE = re.compile(r"<\|link\|([^>]*?)\|+>(.*?)<\|/link\|>", re.S)
HREF_RE = re.compile(r"href='([^']*)'")


def to_bare(text: str) -> str:
    """Rewrite attributed codeblock/link tags in one assistant reply to bare tags."""
    text = CODEBLOCK_ATTR_RE.sub("<|codeblock|>", text)

    def link_sub(m: re.Match) -> str:
        attrs, inner = m.group(1), m.group(2)
        href = HREF_RE.search(attrs)
        # The destination becomes the visible text; fall back to the old text.
        dest = href.group(1) if href else inner
        return f"<|link|>{dest}<|/link|>"

    return LINK_ATTR_RE.sub(link_sub, text)


def migrate_file(path: str, apply: bool) -> tuple[int, int]:
    """Returns (lines_changed, turns_changed) for one .jsonl file."""
    lines_changed = 0
    turns_changed = 0
    out_lines: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                out_lines.append(line)
                continue
            try:
                conv = json.loads(stripped)
            except json.JSONDecodeError:
                # Leave unparseable lines exactly as-is rather than dropping them.
                out_lines.append(line)
                continue

            changed = False
            for turn in conv.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                new_text = to_bare(turn["text"])
                if new_text != turn["text"]:
                    turn["text"] = new_text
                    turns_changed += 1
                    changed = True
            if changed:
                lines_changed += 1
            out_lines.append(json.dumps(conv, ensure_ascii=False) + "\n")

    if apply and lines_changed:
        os.replace(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

    return lines_changed, turns_changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite SFT data to bare codeblock/link tags.")
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"SFT jsonl dir (default {DEFAULT_DIR})")
    ap.add_argument("--apply", action="store_true", help="rewrite in place (keeps a .bak); default is a dry run")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    total_lines = 0
    total_turns = 0
    for path in files:
        lines, turns = migrate_file(path, args.apply)
        total_lines += lines
        total_turns += turns
        print(f"  {os.path.basename(path)}: {lines} conversations, {turns} assistant turns changed")

    verb = "Rewrote" if args.apply else "Would rewrite"
    print(f"{verb} {total_turns} assistant turns across {total_lines} conversations.")
    if not args.apply and total_turns:
        print("Dry run — re-run with --apply to write the changes (.bak kept).")


if __name__ == "__main__":
    main()
