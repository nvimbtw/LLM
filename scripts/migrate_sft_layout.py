#!/usr/bin/env python3
"""
Move loose `.jsonl` at the SFT root into a named set folder.

SFT data used to live flat — `data/sft/smoltalk-000.jsonl` and friends — because
a fine-tune trained on everything there was and had no way to ask for less. Now
it selects sets the way a pretrain selects corpora (`data/sft/<set>/*.jsonl`),
so the flat files need a name.

Loose files still train, collected as a set called `root`, so nothing breaks
before this runs. But `root` is a fallback, not a name anyone would knowingly
select, and leaving 777 MB of conversations in it means any run asking for
"everything" quietly mixes the legacy set into whatever else was fetched. Giving
it a real name is what makes the selection honest.

Moves rather than copies: a half-migrated state where a file exists at both the
root and inside the set folder would be collected twice, training the same
conversations twice with nothing to show for it.

Usage:
    python3 scripts/migrate_sft_layout.py                  # -> data/sft/smoltalk/
    python3 scripts/migrate_sft_layout.py --set legacy     # a different name
    python3 scripts/migrate_sft_layout.py --dry-run
"""

import argparse
import glob
import os
import shutil
import sys

import paths

DEFAULT_SET = "smoltalk"


def loose_files() -> list[str]:
    """Every `.jsonl` sitting directly at the SFT root, sorted."""
    return sorted(glob.glob(os.path.join(paths.sft_dir(), "*.jsonl")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", default=DEFAULT_SET, dest="set_name",
                    help=f"set folder to move them into (default: {DEFAULT_SET})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would move and exit")
    args = ap.parse_args()

    files = loose_files()
    if not files:
        print(f"Nothing to migrate: no loose .jsonl at {paths.sft_dir()}")
        return

    destination = paths.sft_set_dir(args.set_name)
    total = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} file(s), {total / 1e6:.0f} MB -> {destination}")

    # Checked before anything moves, so a collision cannot leave the set half
    # migrated. Refusing is right: the two files are not known to be the same,
    # and overwriting would silently discard one of them.
    clashes = [
        f for f in files if os.path.exists(os.path.join(destination, os.path.basename(f)))
    ]
    if clashes:
        names = ", ".join(os.path.basename(f) for f in clashes[:5])
        sys.exit(
            f"{len(clashes)} file(s) already exist in {destination}: {names}\n"
            f"Nothing moved. Remove them, or pass --set with a different name."
        )

    if args.dry_run:
        for f in files:
            print(f"  would move {os.path.basename(f)}")
        return

    os.makedirs(destination, exist_ok=True)
    for f in files:
        shutil.move(f, os.path.join(destination, os.path.basename(f)))

    print(f"Moved {len(files)} file(s). The SFT root now holds no loose .jsonl.")
    print(f"Select it with:  {{\"datasets\": [\"{args.set_name}\"]}}")


if __name__ == "__main__":
    main()
