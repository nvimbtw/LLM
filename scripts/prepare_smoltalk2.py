#!/usr/bin/env python3
"""
Build the smoltalk2 SFT sets — thinking and non-thinking — as two named sets.

    data/sft/smoltalk2-nothink/*.jsonl
    data/sft/smoltalk2-think/*.jsonl

Two sets, not one blend, because they are meant to be two *runs*: fine-tune the
same base model on each and you get two comparable models rather than one that
sometimes reasons. Select them with `{"datasets": ["smoltalk2-think"]}`.

Source: HuggingFaceTB/smoltalk2, `SFT/` config. Verified schema:

    messages              [{role, content}]
    chat_template_kwargs  {custom_instructions, enable_thinking, python_tools, xml_tools}
    source                which sub-corpus the row came from

The repo already separates the two modes at the file level — shard groups end in
`_think` or `_no_think` — so the split needs no filtering or heuristics. That
separation is the reason this source was chosen over OpenThoughts or the raw
Nemotron set.

`chat_template_kwargs.custom_instructions` is the system prompt, and is usually
empty.

## Reasoning blocks

Thinking rows carry a literal `<think>…</think>` at the start of the assistant's
content. It becomes `<|thinking|>…<|/thinking|>`, the marker pair the engine's
template defines and the frontend renders.

A block that opens and never closes causes the conversation to be **dropped**
rather than emitted. The engine's `strip_thinking()` removes blocks by exact
marker match, so a malformed one survives into what is supposed to be
reasoning-free output and shows up as literal `<think>` in a reply. Dropping is
cheap; a corrupted reply is not.

Reasoning content goes through the same markdown-to-markup conversion as the
reply. The frontend renders the thinking block, so unconverted markdown is as
wrong in there as anywhere else.

## Budget markers

Think conversations carry a `budget` of low/med/high, derived from the measured
length of the reasoning block:

    < 120 estimated tokens   low
    < 400                    med
    otherwise                high

These markers exist in the chat template and, since the smol-smoltalk swap,
train on nothing at all — prepare_smoltalk.py's docstring records that losing
them turned the composer's dropdown into "a lever attached to nothing". This is
what gives them data again.

No-think conversations carry **no** budget field. The absence is a real state,
not a fourth level: the marker's meaning is "the reply after me shows its
reasoning at this budget", and there is no such reply.

## Which shard groups

`--groups` selects them; the default is everything except four groups that are
wrong for a model this size:

    OpenThoughts3_1.2M*   113 shards of very long math/code traces — most would
                          be dropped on length, and what survives is skewed hard
                          toward competition problems
    *multilingual*        eight languages against an 8k English-trained vocab,
                          where non-ASCII costs a token per byte
    aya_dataset*          multilingual for the same reason
    LongAlign_64k*        built for a 131k context; this model has 1k

Pass `--groups all` to take everything, or a comma-separated list to be explicit.
`--list-groups` prints what the repo currently offers without fetching anything.

Usage:
    python3 prepare_smoltalk2.py                        # both sets, curated groups
    python3 prepare_smoltalk2.py --mode think           # only the thinking set
    python3 prepare_smoltalk2.py --limit 20000          # a subset, for a trial run
    python3 prepare_smoltalk2.py --list-groups
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import chat_markup as cm
import corpus_shards as cs
import paths

REPO = "HuggingFaceTB/smoltalk2"
CONFIG = "SFT/"

# One set folder per mode. These names are what a fine-tune's `datasets` asks for.
SETS = {"think": "smoltalk2-think", "nothink": "smoltalk2-nothink"}

OUT_SHARD_CONVERSATIONS = 25_000

# Estimated tokens of rendered conversation to allow, against a 1024-token window.
# Under the ceiling rather than at it: the estimate does not count the scaffold
# markers, and it is an estimate in any case.
DEFAULT_MAX_TOKENS = 900

# Reasoning length, in estimated tokens, at which each budget label starts.
BUDGET_THRESHOLDS = ((120, "low"), (400, "med"))
BUDGET_HIGH = "high"

# Shard groups excluded unless asked for by name — see the docstring.
EXCLUDED = ("openthoughts3", "multilingual", "aya_dataset", "longalign")

# `name-00000-of-00113.parquet` -> `name`. The shard suffix is what makes one
# group look like many files.
SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}\.parquet$")


def group_of(path: str) -> str:
    """The shard group a file belongs to, without its shard numbering."""
    return SHARD_SUFFIX.sub("", os.path.basename(path))


def mode_of(group: str) -> str | None:
    """`think`, `nothink`, or None when the group names neither.

    Order matters: `OpenThoughts3_1.2M_no_think_no_think` ends with both, and
    `_no_think` is the one that decides.
    """
    if group.endswith("_no_think"):
        return "nothink"
    if group.endswith("_think"):
        return "think"
    return None


def discover_groups() -> dict[str, list[str]]:
    """Every shard group in the SFT config, mapped to its shard paths."""
    groups: dict[str, list[str]] = {}
    for path in cs.list_parquet_shards(REPO, prefix=CONFIG):
        groups.setdefault(group_of(path), []).append(path)
    return groups


def wanted_groups(groups: dict[str, list[str]], selector: str, mode: str) -> list[str]:
    """Which groups to read for one mode, honouring `--groups`."""
    candidates = [g for g in sorted(groups) if mode_of(g) == mode]
    if selector == "all":
        return candidates
    if selector:
        asked = {g.strip() for g in selector.split(",") if g.strip()}
        unknown = asked - set(groups)
        if unknown:
            sys.exit(f"Unknown shard group(s): {', '.join(sorted(unknown))}\n"
                     f"Run --list-groups to see what the repo offers.")
        return [g for g in candidates if g in asked]
    return [g for g in candidates if not any(x in g.lower() for x in EXCLUDED)]


def budget_for(reasoning: str) -> str:
    """Which budget marker a reasoning block of this length describes."""
    tokens = cm.est_tokens(reasoning)
    for limit, label in BUDGET_THRESHOLDS:
        if tokens < limit:
            return label
    return BUDGET_HIGH


def build(messages, kwargs, max_tokens, want_thinking, tally):
    """One source row as a conversation, or None with `tally` incremented."""
    system = (kwargs or {}).get("custom_instructions") or None
    if system:
        system = cm.neutralize(system.strip())

    msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(msgs) % 2:
        # A trailing user turn has no reply to train on.
        msgs = msgs[: len(msgs) - 1]
    if not msgs:
        tally["shape"] += 1
        return None

    turns = []
    budget = None
    for index, m in enumerate(msgs):
        expected = "user" if index % 2 == 0 else "assistant"
        if m["role"] != expected:
            tally["shape"] += 1
            return None

        content = m.get("content") or ""
        # The template supplies the structural markers; data claiming to be
        # scaffold is rejected rather than escaped.
        if cm.has_structural_markers(content):
            tally["structural"] += 1
            return None

        if expected == "user":
            turns.append({"role": "user", "text": cm.neutralize(content).strip()})
            continue

        reasoning, reply = cm.split_reasoning(content)
        if want_thinking:
            # An unclosed block leaves `<think>` in the text. `split_reasoning`
            # returns nothing rather than guessing, and this is where that is
            # caught — emitting it would put a literal tag in a trained reply.
            if reasoning is None or "<think>" in reply.lower():
                tally["bad_think"] += 1
                return None
            budget = budget or budget_for(reasoning)
            body = (f"{cm.THINKING_OPEN}{cm.to_markup(cm.neutralize(reasoning))}"
                    f"{cm.THINKING_CLOSE}{cm.to_markup(cm.neutralize(reply))}")
        else:
            # A no-think row should carry no reasoning; if one slipped in, its
            # reply is still usable and the block is simply dropped.
            body = cm.to_markup(cm.neutralize(reply))

        turns.append({"role": "assistant", "text": body})

    if any(not t["text"] for t in turns):
        tally["empty"] += 1
        return None

    kept = cm.fit_pairs(turns, max_tokens, system)
    if not kept:
        tally["too_long"] += 1
        return None

    conversation = {"turns": kept}
    if system:
        conversation["system"] = system
    if want_thinking and budget:
        conversation["budget"] = budget
    return conversation


def clear_previous(out_dir: str, prefix: str) -> int:
    removed = 0
    for path in glob.glob(os.path.join(out_dir, f"{prefix}-*.jsonl")):
        os.remove(path)
        removed += 1
    return removed


def write_set(mode, groups, selector, args) -> None:
    """Fetch and convert one mode into its own set folder."""
    chosen = wanted_groups(groups, selector, mode)
    if not chosen:
        print(f"[{mode}] no shard groups selected; skipping")
        return

    shards = [s for g in chosen for s in sorted(groups[g])]
    out_dir = paths.sft_set_dir(SETS[mode])
    os.makedirs(out_dir, exist_ok=True)
    prefix = SETS[mode]

    print(f"\n[{mode}] {len(chosen)} group(s), {len(shards)} shard(s) -> {out_dir}")
    for g in chosen:
        print(f"    {g}")

    removed = clear_previous(out_dir, prefix)
    if removed:
        print(f"  cleared {removed} previously-prepared file(s)")

    tally = {"shape": 0, "structural": 0, "bad_think": 0, "empty": 0, "too_long": 0}
    written = shard_index = 0
    buffer: list[str] = []
    started = time.time()

    def flush():
        nonlocal shard_index, buffer
        if not buffer:
            return
        name = f"{prefix}-{shard_index:03d}.jsonl"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(buffer) + "\n")
        shard_index += 1
        buffer = []

    rows = cs.iter_parquet_rows(
        REPO, shards, ["messages", "chat_template_kwargs", "source"], batch_size=500
    )
    for row in rows:
        conversation = build(
            row["messages"], row["chat_template_kwargs"],
            args.max_tokens, mode == "think", tally,
        )
        if conversation is None:
            continue
        # Carried for provenance and later filtering; the engine ignores fields
        # it does not declare.
        conversation["source"] = row["source"]
        buffer.append(json.dumps(conversation, ensure_ascii=False))
        written += 1
        if len(buffer) >= OUT_SHARD_CONVERSATIONS:
            flush()
        if args.limit and written >= args.limit:
            break
    flush()

    dropped = sum(tally.values())
    total = written + dropped
    print(f"  wrote {written:,} conversations across {shard_index} file(s) "
          f"in {time.time() - started:.0f}s")
    print(f"  kept {100.0 * written / max(total, 1):.1f}% of {total:,} source rows")
    for reason, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        if count:
            print(f"    dropped {count:,} — {reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", default="both", choices=("both", "think", "nothink"),
                    help="which set(s) to build (default: both)")
    ap.add_argument("--groups", default="",
                    help="comma-separated shard groups, or 'all'; default is a curated subset")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N conversations per set (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="estimated tokens of rendered conversation to allow")
    ap.add_argument("--list-groups", action="store_true",
                    help="print the available shard groups and exit")
    args = ap.parse_args()

    groups = discover_groups()

    if args.list_groups:
        for name in sorted(groups):
            mode = mode_of(name) or "-"
            excluded = " (excluded by default)" if any(
                x in name.lower() for x in EXCLUDED) else ""
            print(f"  {mode:8s} {len(groups[name]):3d} shard(s)  {name}{excluded}")
        return

    for mode in ("nothink", "think"):
        if args.mode in ("both", mode):
            write_set(mode, groups, args.groups, args)


if __name__ == "__main__":
    main()
