"""Adopt newly named markers into every vocabulary already on disk.

    python -m scripts.relabel_reserved_slots [--apply]

Dry by default: it prints what it would rename and changes nothing. Run it after
appending to `corpus.special.SPECIAL_TOKENS`, so that models trained before the
marker existed can still be served under a grammar that knows about it.

Why this is a rename and not a retrain — and what it deliberately leaves alone —
is in `llm.corpus.relabel`. Idempotent: a vocabulary already current reports no
renames, so re-running costs nothing.
"""

import argparse
from pathlib import Path

from llm import storage
from llm.corpus import relabel


def vocabulary_files() -> list[tuple[Path, Path | None]]:
    """Every saved vocabulary, paired with the build record beside it.

    Two places hold one: named tokenizer builds, and the private copy each model
    keeps next to its weights — which is the one inference actually loads, so
    missing it would leave every trained model unable to spell the new marker.
    """
    found: list[tuple[Path, Path | None]] = []

    for build in sorted(storage.tokenizers_dir().glob("*/")):
        path = storage.tokenizer_file(build)
        if path.is_file():
            found.append((path, storage.tokenizer_meta_path(build.name)))

    for model in sorted(storage.models_dir().glob("*/")):
        path = storage.tokenizer_file(model)
        if path.is_file():
            found.append((path, None))

    return found


def main() -> int:
    args = _arguments()
    files = vocabulary_files()
    if not files:
        print(f"No vocabularies under {storage.root()}")
        return 0

    changed = 0
    for path, build in files:
        renames = relabel.relabel_file(path, dry_run=not args.apply)
        restamped = relabel.refresh_build_sha(build, dry_run=not args.apply) if build else False
        if not renames and not restamped:
            continue

        changed += 1
        print(path)
        for slot, was, now in renames:
            print(f"  {slot:>3}  {was}  ->  {now}")
        if restamped:
            print("  special_tokens_sha restamped")

    verb = "Relabelled" if args.apply else "Would relabel"
    print(f"\n{verb} {changed} of {len(files)} vocabularies.")
    if changed and not args.apply:
        print("Re-run with --apply to write.")
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write the changes instead of listing them"
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
