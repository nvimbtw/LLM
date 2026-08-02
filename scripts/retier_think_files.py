#!/usr/bin/env python3
"""Re-files raw/sft/think-*.jsonl under new thinking-budget boundaries.

The tier boundaries used to be fixed constants (140/700) chosen without reference
to the student's context window, which made the top tier arithmetically
impossible: a record is filed `high` only when its thinking block exceeds
`med_max`, and a 700-token block plus a reply plus the user turn does not fit a
1024-token window. They are derived from the window now — see
`synth::teach::derived_bounds`.

Records written under the old boundaries carry old labels, so the corpus would
otherwise contain conversations labelled `low` under two different rules. That is
worse than either rule alone: the whole point of the marker is that one label
means one thing.

No regeneration is needed. Every record stores its own `think_tokens`, measured in
the student's tokenizer at the time it was written, so re-filing is a pure data
operation.

    python3 scripts/retier_think_files.py --ctx 1024            # dry run
    python3 scripts/retier_think_files.py --ctx 1024 --write

`--ctx` must be the context window of the student the records were *measured*
against; passing a different one silently re-labels against the wrong scale.
"""

import argparse
import collections
import json
import os

SFT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "data", "raw", "sft",
)
TIERS = ("low", "med", "high")


def derived_bounds(ctx):
    """Mirror of `synth::teach::derived_bounds`. Keep the two in step."""
    ceiling = (ctx * 9 // 10) * 45 // 100
    return ceiling * 15 // 100, ceiling * 60 // 100


def classify(think_tokens, low_max, med_max):
    if think_tokens <= low_max:
        return "low"
    if think_tokens <= med_max:
        return "med"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True,
                    help="context window of the student the records were measured against")
    ap.add_argument("--write", action="store_true", help="apply; otherwise dry run")
    args = ap.parse_args()

    low_max, med_max = derived_bounds(args.ctx)
    print(f"ctx {args.ctx}: low <= {low_max}, med <= {med_max}, high above {med_max}\n")

    records = []
    for tier in TIERS:
        path = os.path.join(SFT_DIR, f"think-{tier}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                records.append((tier, r))

    if not records:
        print("no think-*.jsonl records found")
        return

    moves = collections.Counter()
    out = {t: [] for t in TIERS}
    for was, r in records:
        # A record with no measurement was filed by its *requested* tier, and
        # re-deriving it from a 0 would move it to `low` on no evidence. Left where
        # it is.
        if "think_tokens" not in r:
            out[was].append(r)
            moves[(was, was)] += 1
            continue
        now = classify(r["think_tokens"], low_max, med_max)
        r["budget"] = now
        out[now].append(r)
        moves[(was, now)] += 1

    print(f"{'from':>6} -> {'to':<6} {'count':>6}")
    for (was, now), n in sorted(moves.items()):
        mark = "" if was == now else "   <-- moved"
        print(f"{was:>6} -> {now:<6} {n:>6}{mark}")
    print()
    for tier in TIERS:
        print(f"  think-{tier}.jsonl: {len(records and out[tier]):>4} records")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return

    for tier in TIERS:
        path = os.path.join(SFT_DIR, f"think-{tier}.jsonl")
        if not out[tier]:
            # An empty tier gets an empty file rather than a stale one: leaving the
            # previous contents in place is how a re-tier silently does nothing.
            if os.path.exists(path):
                os.remove(path)
                print(f"removed empty {os.path.basename(path)}")
            continue
        with open(path, "w", encoding="utf-8") as f:
            for r in out[tier]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(out[tier])} to {os.path.basename(path)}")


if __name__ == "__main__":
    main()
