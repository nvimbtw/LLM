#!/usr/bin/env python3
"""
Populate a Code dataset folder with real source files.

Source: codeparrot/github-code — 1,126 parquet shards of permissively-licensed
GitHub files, ~102k rows each. Verified schema:

    repo_name  owner/repo
    path       the file's path within the repo
    size       bytes
    content    the file itself
    license    SPDX-ish licence id

Chosen over bigcode/the-stack-smol and bigcode/starcoderdata, which are the more
obvious candidates and the better corpora, because both are **gated**: they need
a licence agreement accepted on the Hub and an authenticated client. This one is
open, and at the volume wanted here the difference in curation matters less than
not having the fetch fail overnight on a 401.

## Why code is in the pretrain corpus at all

The assistant this model is fine-tuned toward writes code — the SFT markup has
`<|codeblock|>` and `<|code|>` for it, and prepare_smoltalk.py's converter
refuses underscore emphasis specifically because that corpus is 11% code. But
the pretrain corpus contained essentially none, so those markers were wrapping
text the base model had never modelled.

## Two filters that matter more than they look

**Language.** `--langs` selects by file extension, defaulting to what the
assistant is actually asked about. This config carries no `language` column, so
the extension is the signal available. Taking every language uniformly would
spend the budget on things nobody will ask about.

**Minified, generated and vendored files.** A bundle or a lockfile is bytes
without structure — one 400 KB line of transpiled JavaScript teaches the
tokenizer a great deal and the model nothing. Filtered on mean line length and
on the fraction of non-alphanumeric characters, which catches minified output
without catching dense but legitimate code.

Vendored dependencies need a separate filter, because they defeat the first one
completely: a file checked into `node_modules/` is ordinary well-formed source
and every content heuristic passes it. It is still near-worthless here — the
same handful of libraries copied into thousands of repositories, so the corpus
would be dominated by however many projects vendored `lodash`. The first sample
drawn while testing this script was exactly that. Filtered on path instead.

## A caveat worth stating

The BPE trains on whatever the corpus contains, at `max_vocab: 8192`. Code is
dense in punctuation and identifiers and competes hard for merges against prose,
so adding ~10 GB of it beside ~40 GB of text is a real shift in what the
tokenizer spends itself on. Whether that trade is worth it is a measurement, not
an assumption — this script just makes the data available to measure with.

Usage:
    python3 prepare_code.py                            # 10 GB, default languages
    python3 prepare_code.py --mb 2000
    python3 prepare_code.py --langs py,rs,ts
    python3 prepare_code.py --dry-run
"""

import corpus_shards as cs

REPO_ID = "codeparrot/github-code"
DATASET = "Code"
DEFAULT_MB = 10_000

# What the assistant is actually asked about, plus documentation. Extensions
# rather than language names: this config has no language column.
DEFAULT_LANGS = "py,ts,tsx,js,jsx,rs,go,sh,sql,md,toml,yaml,yml"

# Source files below this are stubs, `__init__.py` and one-line configs.
MIN_CHARS = 200

# Minification heuristics. A source file averaging more than this per line is
# either generated or one long data literal; real code wraps.
MAX_MEAN_LINE = 200

# Above this fraction of non-alphanumeric, non-whitespace characters the file is
# closer to encoded data than to code. Real code sits well under it — punctuation
# is frequent but identifiers and keywords dominate.
MAX_SYMBOL_RATIO = 0.5

# Files past this are generated bundles more often than not, and one of them can
# swallow a whole shard.
MAX_CHARS = 200_000

# Path segments marking code that was copied in rather than written. Matched as
# `/segment/` against the file's path, so a repository *named* `vendor` is not
# excluded — only files sitting inside such a directory.
VENDORED = (
    "node_modules", "bower_components", "site-packages", "vendor", "third_party",
    "thirdparty", "external", "dist", "build", "target", ".min", "generated",
)


def is_vendored(path: str) -> bool:
    """Whether a file was copied into the repository rather than written in it."""
    segments = path.lower().split("/")
    return any(
        segment == marker or (marker == ".min" and ".min." in segment)
        for segment in segments
        for marker in VENDORED
    )


def looks_generated(text: str) -> bool:
    """Whether a file is minified, generated or otherwise not worth training on."""
    lines = text.count("\n") + 1
    if len(text) / lines > MAX_MEAN_LINE:
        return True
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / max(len(text), 1) > MAX_SYMBOL_RATIO


def iter_files(shards, extensions, min_chars):
    """Source files as documents, one file per document.

    A source file is already a natural document boundary — concatenating
    unrelated files would teach the model that one ends where another begins.
    """
    for row in cs.iter_parquet_rows(REPO_ID, shards, ["path", "content"]):
        path = row["path"] or ""
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix not in extensions or is_vendored(path):
            continue
        body = (row["content"] or "").strip()
        if len(body) < min_chars or len(body) > MAX_CHARS:
            continue
        if looks_generated(body):
            continue
        yield cs.neutralize(body)


def main():
    ap = cs.base_parser(__doc__.split("\n")[1], default_mb=DEFAULT_MB, dataset=DATASET)
    ap.set_defaults(min_chars=MIN_CHARS)
    ap.add_argument("--langs", default=DEFAULT_LANGS,
                    help=f"comma-separated file extensions (default: {DEFAULT_LANGS})")
    args = ap.parse_args()

    extensions = {e.strip().lower().lstrip(".") for e in args.langs.split(",") if e.strip()}
    cs.projection(REPO_ID, args.mb, {"languages": ", ".join(sorted(extensions))})
    if args.dry_run:
        return

    shards = cs.list_parquet_shards(REPO_ID, prefix="data/")
    print(f"{len(shards)} shards available; fetching from the front as needed.")

    cs.run_fetch(
        args, "code",
        lambda reverse: iter_files(
            list(reversed(shards)) if reverse else shards, extensions, args.min_chars
        ),
        prefix="code",
        manifest={"repo": REPO_ID, "extensions": sorted(extensions),
                  "min_chars": args.min_chars},
    )


if __name__ == "__main__":
    main()
