#!/usr/bin/env python3
"""
Rebuild the `smoltalk` SFT set (data/sft/smoltalk/) from HuggingFaceTB/smol-smoltalk.

This writes one *named set*, not the whole SFT directory: a fine-tune selects
sets the way a pretrain selects corpora, so this corpus sits beside the thinking
sets rather than being blended into them. See scripts/prepare_smoltalk2.py for
the reasoning-trace sets and engine/llm/corpus/sft.py for how they are chosen.

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

## Conversion

The markdown-to-markup conversion, the token estimate and the window fitting all
live in chat_markup.py, which every SFT fetcher shares — see its docstring for
the rules and why each one is the way it is. What stays here is smol-smoltalk's
own schema and its drop accounting.

Usage:
    python3 scripts/prepare_smoltalk.py                    # replace raw/sft/
    python3 scripts/prepare_smoltalk.py --keep-thinking    # keep think-*.jsonl too
    python3 scripts/prepare_smoltalk.py --limit 20000      # a subset, for a trial run
"""

import argparse
import glob
import json
import os
import sys
import time

import chat_markup as cm
import paths

REPO = "HuggingFaceTB/smol-smoltalk"

# Repo-relative, resolved against this file so the script runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This script owns one set folder and never writes outside it, so every other
# SFT set — the thinking ones especially — is safe from --replace-all.
OUT_SET = "smoltalk"
SFT_DIR = paths.sft_set_dir(OUT_SET)

# Where the pre-set-layout data lived. Only the legacy think-*.jsonl handling
# below still looks here; everything this script writes goes to SFT_DIR.
LEGACY_ROOT = paths.sft_dir()

# The files this script owns within its set folder. Everything else there is left
# alone unless --replace-all is passed.
OUT_PREFIX = "smoltalk"
OUT_SHARD_CONVERSATIONS = 25_000

# Estimated tokens of rendered conversation to allow, against a 1024-token window.
#
# Under the ceiling rather than at it: the estimate below does not count the
# scaffold (`<|user|>`, `<|assistant|>` and their closers, one token each) or the
# markers the markdown conversion adds, and it is an estimate in any case.
DEFAULT_MAX_TOKENS = 900

# The thinking-block set this replaces, kept only under --keep-thinking.
THINKING_FILES = ("think-low.jsonl", "think-med.jsonl", "think-high.jsonl")


def build(messages, max_tokens: int):
    """One source conversation as an `SftConversation`, or None if unusable."""
    msgs = list(messages)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = cm.neutralize(msgs[0]["content"].strip())
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
        text = cm.neutralize(m["content"])
        turns.append(
            {"role": expected,
             "text": cm.to_markup(text) if expected == "assistant" else text.strip()}
        )
    if any(not t["text"] for t in turns):
        return None

    kept = cm.fit_pairs(turns, max_tokens, system)
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

    Scoped to `smoltalk-*.jsonl` inside this script's own set folder by default,
    so a hand-written file beside it survives; --replace-all empties that folder.
    Neither touches a sibling set."""
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
        help="empty this script's set folder rather than only removing its own output",
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
        kept_thinking = [f for f in THINKING_FILES if os.path.exists(os.path.join(LEGACY_ROOT, f))]
    else:
        for name in THINKING_FILES:
            path = os.path.join(LEGACY_ROOT, name)
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
