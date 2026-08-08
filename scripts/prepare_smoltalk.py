#!/usr/bin/env python3
"""
Rebuild data/raw/sft/ from HuggingFaceTB/smol-smoltalk.

Source: 484,570 conversations (460,341 train + 24,229 test), the variant of
SmolTalk that HuggingFace tuned for their 135M/360M models — shorter turns and
plainer language than the full set, which is the right register for a model this
size. Schema is `messages: [{role, content}]` plus a `source` tag naming the
sub-corpus. 15% carry a system prompt; 99.94% strictly alternate user/assistant
and end on an assistant turn.

This replaces a 1,912-conversation set that was largely self-generated: the model
was being fine-tuned on text produced by models in the same family, which caps how
far the fine-tune can pull it. Real instruction-following data at 150x the volume
is the point of the swap.

## What is lost, and it is not small

**smol-smoltalk has no reasoning traces.** Every `<|thinking|>` block and every
`<|think:low|>`/`<|think:med|>`/`<|think:high|>` budget marker disappears from SFT
with this data. The budget stays renderable — `chat_template` is untouched — but
nothing trains it, so the composer's dropdown becomes a lever attached to nothing
and `untrained_marker_ids` will reset those three embedding rows on the next
fine-tune. That is the direct cost of "replace", and `--keep-thinking` re-mixes the
old `think-*.jsonl` alongside if it is not worth paying.

## Markdown becomes markup

The corpus is markdown; the models here speak the marker vocabulary in
`tokenize::pretokenize::SPECIAL_TOKENS`. Block level:

    blank-line separated  ->  <|par|>...<|/par|>
    ```lang ... ```       ->  <|codeblock|>...<|/codeblock|>
    -, *, +, "1." lines   ->  <|list|><|item|>...<|/item|>...<|/list|>
    # H / ## H            ->  <|heading|> / <|subheading|>
    > quoted              ->  <|quote|>...<|/quote|>
    --- / *** / ___       ->  <|divider|>

Inline, inside everything but a code block:

    `x`                   ->  <|code|>x<|/code|>
    **x**                 ->  <|bold|>x<|/bold|>
    *x*                   ->  <|italic|>x<|/italic|>
    [text](url)           ->  text <|link|>url<|/link|>

Underscore emphasis (`_x_`, `__x__`) is deliberately **not** converted: this corpus
is 11% code and `sort_tuples` or `__init__` would be mangled into italics. Bare URLs
are left alone for the same reason — a URL inside a code span is common and the
inline pass would have to unpick which is which.

Nested lists are flattened to one level. The rendering has no nesting convention
below `list > item`, and a model at this size has no chance of keeping two levels
balanced anyway.

Code spans are lifted out before the emphasis pass and put back after, so `**` and
`*` inside code are left exactly as written.

## Fitting the context window

The whole rendered example — system, every turn, the scaffold — has to fit the
model's window. `build_sft_dataset` refuses one that does not and counts it as
`skipped_too_long`, so nothing downstream breaks; the cut happens here as well to
keep that count near zero, because a conversation refused there is one this script
imported, converted and wrote for nothing.

Conversations are truncated to the longest whole-pair prefix that fits, which keeps
a 6-turn conversation as its first 2 or 4 turns rather than dropping it. A pair is
kept or dropped whole: cutting inside an assistant reply would teach the model to
stop mid-sentence, which is the exact failure the reply-cap investigation spent a
week on (see backend/CLAUDE.md). A conversation whose *first* pair alone overruns
the budget is dropped entirely.

The budget is an *estimate* of tokens, because the tokenizer is Rust and this is
not. Two rates, because one is badly wrong:

    ASCII bytes      ~3.7 bytes per token   (the vocabulary was trained on English)
    non-ASCII bytes  ~1.0 bytes per token   (no merge covers them; byte fallback)

Characters are the wrong unit and bytes alone are barely better. This corpus has
Chinese, Russian and Korean conversations in it — openhermes-50k especially — and
one of them measured 2,572 characters, 7,108 UTF-8 bytes and roughly 7,000 tokens
under an 8192-vocabulary that has never seen a Han character. A character budget
admitted it; a byte budget would have too.

That the two-rate estimate prices such conversations out is the right outcome and
not a workaround. A model with this vocabulary cannot represent them in less than
one token per byte, so they would spend a seventh of the window on a sentence while
teaching nothing an English model can use.

`cargo test -p llm --test sft_fits_the_window -- --ignored` is the exact check, in
real tokens against the real merges, and it is what says whether this estimate is
still close enough.

## Marker neutralization

Any `<|...|>`-shaped text already in the source is broken with a zero-width space
before it is used, exactly as prepare_fineweb.py does. This corpus contains
tokenizer and prompt-engineering discussions, so it genuinely does contain literal
`<|endoftext|>` — left alone it would become a real document boundary mid-reply.

Usage:
    python3 scripts/prepare_smoltalk.py                    # replace raw/sft/
    python3 scripts/prepare_smoltalk.py --keep-thinking    # keep think-*.jsonl too
    python3 scripts/prepare_smoltalk.py --limit 20000      # a subset, for a trial run
"""

import paths
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

REPO = "HuggingFaceTB/smol-smoltalk"

# Repo-relative, resolved against this file so the script runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_DIR = paths.sft_dir()

# The files this script owns. Everything else in raw/sft/ is left alone unless
# --replace-all is passed, so a hand-written set can sit beside this one.
OUT_PREFIX = "smoltalk"
OUT_SHARD_CONVERSATIONS = 25_000

# Estimated tokens of rendered conversation to allow, against a 1024-token window.
#
# Under the ceiling rather than at it: the estimate below does not count the
# scaffold (`<|user|>`, `<|assistant|>` and their closers, one token each) or the
# markers the markdown conversion adds, and it is an estimate in any case.
DEFAULT_MAX_TOKENS = 900

# Bytes per token for ASCII text under the current 8192-merge vocabulary.
#
# Held-out prose measures 3.79 and this corpus averages 3.71, but neither is the
# number to use: what overruns the window is the code-heavy tail, and code
# tokenizes far worse than the prose the merges were learned from. Tuned down
# against the exact check until the fraction `build_sft_dataset` has to refuse
# settled: 3.5 left 1.18% overrunning, 3.0 leaves 0.29% while still keeping 93.7%
# of the source. Lower it further and the loss is in what gets dropped, not in
# what gets through.
ASCII_BYTES_PER_TOKEN = 3.0

# The thinking-block set this replaces, kept only under --keep-thinking.
THINKING_FILES = ("think-low.jsonl", "think-med.jsonl", "think-high.jsonl")

# Anything matching the tokenizer's marker grammar. The zero-width break keeps the
# words readable while stopping `replace_special_tokens` matching.
MARKER = re.compile(r"<\|(/?[a-zA-Z0-9:]+)\|>")

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
QUOTE = re.compile(r"^\s{0,3}>\s?")
HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
BULLET = re.compile(r"^(\s*)(?:[-*+]|\d{1,3}[.)])\s+(.*)$")
FENCE = re.compile(r"^\s*```")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
BOLD = re.compile(r"\*\*(\S(?:[^*]*\S)?)\*\*")
ITALIC = re.compile(r"(?<![*\w])\*(\S(?:[^*]*\S)?)\*(?![*\w])")
MDLINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)\)")

# Placeholder for a lifted code span. Control characters cannot appear in the
# source text, so this cannot collide with anything real.
HOLE = "\x00%d\x00"
HOLE_RE = re.compile(r"\x00(\d+)\x00")


def neutralize(text: str) -> str:
    """Break any literal marker so it cannot be tokenized as a real one."""
    return MARKER.sub("<|\\1|​>", text)


def inline(text: str) -> str:
    """Emphasis, code spans and links, in the one order that composes.

    Code spans are lifted first so the emphasis passes cannot see inside them, and
    restored last so a `**` inside code survives verbatim."""
    spans: list[str] = []

    def lift(m: re.Match) -> str:
        spans.append(m.group(1))
        return HOLE % (len(spans) - 1)

    text = CODE_SPAN.sub(lift, text)
    text = MDLINK.sub(
        lambda m: (f"{m.group(1)} " if m.group(1).strip() else "")
        + f"<|link|>{m.group(2)}<|/link|>",
        text,
    )
    text = BOLD.sub(lambda m: f"<|bold|>{m.group(1)}<|/bold|>", text)
    text = ITALIC.sub(lambda m: f"<|italic|>{m.group(1)}<|/italic|>", text)
    return HOLE_RE.sub(lambda m: f"<|code|>{spans[int(m.group(1))]}<|/code|>", text)


def starts_block(line: str) -> bool:
    """Whether `line` begins a block, and so ends the one being accumulated."""
    return bool(
        not line.strip()
        or FENCE.match(line)
        or HEADING.match(line)
        or QUOTE.match(line)
        or HR.match(line)
        or BULLET.match(line)
    )


def split_blocks(text: str):
    """Markdown to a flat sequence of (kind, payload) blocks."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if FENCE.match(line):
            body = []
            i += 1
            while i < len(lines) and not FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # the closing fence, or the end of the text
            yield "codeblock", "\n".join(body).strip("\n")
            continue

        if HR.match(line):
            yield "divider", ""
            i += 1
            continue

        m = HEADING.match(line)
        if m:
            kind = "heading" if len(m.group(1)) == 1 else "subheading"
            yield kind, m.group(2).strip()
            i += 1
            continue

        if QUOTE.match(line):
            body = []
            while i < len(lines) and QUOTE.match(lines[i]):
                body.append(QUOTE.sub("", lines[i]).strip())
                i += 1
            yield "quote", " ".join(x for x in body if x)
            continue

        if BULLET.match(line):
            items: list[str] = []
            while i < len(lines):
                m = BULLET.match(lines[i])
                if m:
                    # Nesting is flattened: an indented item joins the same list.
                    items.append(m.group(2).strip())
                    i += 1
                elif lines[i].strip() and not starts_block(lines[i]):
                    # A wrapped continuation line belongs to the item above it.
                    if items:
                        items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            yield "list", [x for x in items if x]
            continue

        body = []
        while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
            body.append(lines[i].strip())
            i += 1
        yield "par", " ".join(body)


def to_markup(text: str) -> str:
    """An assistant reply, rendered in the marker vocabulary."""
    out = []
    for kind, payload in split_blocks(text):
        if kind == "codeblock":
            # Never passed through `inline`: its content is code, and a `*` in it is
            # multiplication rather than emphasis.
            if payload:
                out.append(f"<|codeblock|>{payload}<|/codeblock|>")
        elif kind == "divider":
            out.append("<|divider|>")
        elif kind == "list":
            if payload:
                items = "".join(f"<|item|>{inline(x)}<|/item|>" for x in payload)
                out.append(f"<|list|>{items}<|/list|>")
        elif payload:
            out.append(f"<|{kind}|>{inline(payload)}<|/{kind}|>")
    return "".join(out)


def est_tokens(text: str) -> int:
    """Roughly how many tokens `text` costs under the current vocabulary.

    Non-ASCII bytes are charged one token each because no merge covers them — the
    encoder falls back to raw bytes, so a Han character is three tokens and a
    paragraph of Chinese is its own byte length. Getting this wrong in the cheap
    direction (characters, or bytes at one flat rate) admits conversations that
    overrun the window by 6x."""
    raw = text.encode("utf-8")
    non_ascii = sum(1 for b in raw if b >= 0x80)
    return int((len(raw) - non_ascii) / ASCII_BYTES_PER_TOKEN + non_ascii) + 1


def build(messages, max_tokens: int):
    """One source conversation as an `SftConversation`, or None if unusable."""
    msgs = list(messages)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = neutralize(msgs[0]["content"].strip())
        msgs = msgs[1:]

    if not msgs or len(msgs) % 2 != 0:
        # Not a whole number of exchanges; the tail would be a user turn with no
        # reply, which `build_training_example` would render with nothing to train on.
        msgs = msgs[: len(msgs) - (len(msgs) % 2)]
    if not msgs:
        return None

    turns = []
    for index, m in enumerate(msgs):
        expected = "user" if index % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None
        text = neutralize(m["content"])
        turns.append(
            {"role": expected, "text": to_markup(text) if expected == "assistant" else text.strip()}
        )
    if any(not t["text"] for t in turns):
        return None

    budget = max_tokens - (est_tokens(system) if system else 0)
    kept: list[dict] = []
    used = 0
    for i in range(0, len(turns), 2):
        pair = turns[i : i + 2]
        cost = sum(est_tokens(t["text"]) for t in pair)
        if used + cost > budget:
            break
        kept.extend(pair)
        used += cost
    if not kept:
        # Even the opening exchange overruns the window. Truncating inside a reply
        # would teach the model to stop mid-sentence, so this one is dropped.
        return None

    conversation = {"turns": kept}
    if system:
        conversation["system"] = system
    return conversation


def load_shards(cache_dir: str):
    """Every parquet shard, train and test, downloading if needed."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, repo_type="dataset", local_dir=cache_dir)
    shards = sorted(glob.glob(os.path.join(path, "data", "*.parquet")))
    if not shards:
        sys.exit(f"no parquet shards under {path}/data")
    return shards


def clear_previous(replace_all: bool) -> int:
    """Remove what a previous run of this script wrote.

    Scoped to `smoltalk-*.jsonl` by default so a hand-written set beside it
    survives; --replace-all empties the directory."""
    pattern = "*.jsonl" if replace_all else f"{OUT_PREFIX}-*.jsonl"
    removed = 0
    for path in glob.glob(os.path.join(SFT_DIR, pattern)):
        os.remove(path)
        removed += 1
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--limit", type=int, default=0, help="stop after N conversations (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="estimated tokens of rendered conversation to allow")
    ap.add_argument(
        "--keep-thinking",
        action="store_true",
        help="leave think-*.jsonl in place, so the budget markers keep training data",
    )
    ap.add_argument(
        "--replace-all",
        action="store_true",
        help="empty raw/sft/ rather than only removing this script's own output",
    )
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache", "smoltalk"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    os.makedirs(SFT_DIR, exist_ok=True)
    shards = load_shards(args.cache_dir)

    if args.keep_thinking and args.replace_all:
        sys.exit("--keep-thinking and --replace-all contradict each other")

    kept_thinking = []
    if args.keep_thinking:
        kept_thinking = [f for f in THINKING_FILES if os.path.exists(os.path.join(SFT_DIR, f))]
    else:
        for name in THINKING_FILES:
            path = os.path.join(SFT_DIR, name)
            if os.path.exists(path):
                os.remove(path)

    removed = clear_previous(args.replace_all)
    print(f"removed {removed} previously-prepared file(s) from {SFT_DIR}")

    written = dropped_shape = dropped_long = 0
    shard_index = 0
    buffer: list[str] = []
    started = time.time()

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        name = f"{OUT_PREFIX}-{shard_index:03d}.jsonl"
        with open(os.path.join(SFT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(buffer) + "\n")
        shard_index += 1
        buffer = []

    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=2000, columns=["messages", "source"]):
            sources = batch.column("source").to_pylist()
            for messages, source in zip(batch.column("messages").to_pylist(), sources):
                before = len(messages)
                conversation = build(messages, args.max_tokens)
                if conversation is None:
                    if before < 2:
                        dropped_shape += 1
                    else:
                        dropped_long += 1
                    continue
                # Carried for provenance and later filtering; `SftConversation`
                # ignores fields it does not declare.
                conversation["source"] = source
                buffer.append(json.dumps(conversation, ensure_ascii=False))
                written += 1
                if len(buffer) >= OUT_SHARD_CONVERSATIONS:
                    flush()
                if args.limit and written >= args.limit:
                    flush()
                    report(written, dropped_shape, dropped_long, shard_index, kept_thinking, started)
                    return
    flush()
    report(written, dropped_shape, dropped_long, shard_index, kept_thinking, started)


def report(written, dropped_shape, dropped_long, shards, kept_thinking, started) -> None:
    total = written + dropped_shape + dropped_long
    print(f"wrote {written:,} conversations across {shards} file(s) in {SFT_DIR}")
    print(f"  dropped {dropped_long:,} that could not fit the window even as one exchange")
    print(f"  dropped {dropped_shape:,} with an unusable turn structure")
    print(f"  kept {100.0 * written / max(total, 1):.1f}% of {total:,} source conversations")
    if kept_thinking:
        print(f"  kept alongside: {', '.join(kept_thinking)}")
    else:
        print("  no thinking blocks in this data: the <|think:*|> markers now train on nothing")
    print(f"  {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
