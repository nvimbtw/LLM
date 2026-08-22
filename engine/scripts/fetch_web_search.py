"""Fetch and convert a web-search SFT set from HuggingFace.

    python -m scripts.fetch_web_search [--count 2000]

Downloads a small slice of the Search-R1 dataset (ThornZ/Search-R1-SFT — real
Google-search traces: questions, the query the model issued, the ranked
documents it got back, and the grounded answer) and rewrites it into this
engine's conversation format:

    {"turns": [{role, text}, ...], "system": "...", "source": "search-r1"}

An assistant turn that issued a call renders as a `<|thinking|>` block (if the
source carried one) followed by
`<|tool-call|>{"tool": "web-search", "argument": query}<|/tool-call|>`.  The
result comes back as a `tool` turn that restates the call it answers and lists
the returned documents as `<|list|><|item|>` entries.  The closing assistant
turn grounds the answer from the results.  That is exactly the shape
`llm.chat.template.training_segments` renders — the engine wraps the tool turn
in `<|tool-result|>` at build time and never trains on it.

The raw Search-R1 results run to kilobytes a round, and the models here train
at a 512-token context, so the converter truncates: a cap on items per round
and on characters per item, both overridable, then a fixed-seed shuffle and a
count cap keep the output deterministic and reproducible.  The source revision
is pinned for the same reason a capped corpus selection is: the same request
must produce the same rows every time.
"""

import argparse
import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from llm import storage

SOURCE_REPO = "ThornZ/Search-R1-SFT"
SOURCE_FILE = "qwen2.5-7b-instruct-sft.json"
# Pinned so re-running fetches byte-identical input: the same ask must select
# the same conversations, the same way a capped corpus selection does.
SOURCE_REVISION = "fb5002ba90900bc896e82efb060ee93c7d9949fa"
OUTPUT_SET = "websearch"
SHARD_SIZE = 1000

THINK_OPEN = "<|thinking|>"
THINK_CLOSE = "<|/thinking|>"
PAR_OPEN = "<|par|>"
PAR_CLOSE = "<|/par|>"
LIST_OPEN = "<|list|>"
ITEM_OPEN = "<|item|>"
ITEM_CLOSE = "<|/item|>"
LIST_CLOSE = "<|/list|>"

# Where this corpus adds a web-search call and its answer. Every tool is called
# through the one marker, with the tool named inside the JSON — see
# `llm/chat/tools.py`, which is the source of truth for the layout. Spelled out
# here rather than imported because pulling the whole llm package into a fetch
# script makes the script depend on the tokenizer settings it exists to feed;
# the assertion at the bottom of this block is what keeps the two honest.
TOOL_CALL_OPEN = "<|tool-call|>"
TOOL_CALL_CLOSE = "<|/tool-call|>"


def _call(query: str) -> str:
    """One web-search call, exactly as `llm.chat.tools.call` renders it.

    `json.dumps` and not an f-string: a Search-R1 query is a natural-language
    question and several of them contain quotation marks.
    """
    payload = json.dumps({"tool": "web-search", "argument": query.strip()}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"

# The Search-R1 source tags this script reads but does not emit. Distinct from
# the output markers above on purpose — `<thinking>` is not `<|thinking|>`.
SOURCE_THINK_OPEN = "<thinking>"
SOURCE_THINK_CLOSE = "</thinking>"

# One ranked document per entry, e.g.
#     [Document 12] (ID: 9834215, Score: 0.912)
#     "Eiffel Tower" The Eiffel Tower was completed in March 1889...
DOC_RE = re.compile(
    r"\[Document\s+\d+\]\s*\(ID:\s*\d+,\s*Score:\s*[\d.]+\)\s*(.*?)(?=\n\[Document\s+\d+\]|\Z)",
    re.DOTALL,
)
QUOTED_TITLE = re.compile(r'^"(.*?)"\s*(.*)$', re.DOTALL)


def fetch(filename: str, revision: str) -> Path:
    """Download the source JSON, returning the local path."""
    print(f"fetching {SOURCE_REPO}@{revision[:7]} {filename}…")
    return Path(
        hf_hub_download(
            SOURCE_REPO,
            filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def _thinking(text: str) -> str:
    m = re.search(
        rf"{re.escape(SOURCE_THINK_OPEN)}(.*?){re.escape(SOURCE_THINK_CLOSE)}",
        text,
        re.DOTALL,
    )
    if not m:
        return ""
    body = _clean(m.group(1).strip())
    if not body:
        return ""
    return f"{THINK_OPEN}{PAR_OPEN}{body}{PAR_CLOSE}{THINK_CLOSE}"


def _clean(text: str) -> str:
    """Collapse whitespace and drop residual Search-R1 citation scaffolding."""
    text = re.sub(r"<cite(?:\s+[^>]*)?>|</cite\s*>", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def truncated(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters at a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    at = cut.rfind(" ")
    if at > 0:
        cut = cut[:at]
    return cut.rstrip(" ,.;:!?") + "…"


def _question(content: str) -> str:
    """The question behind the Search-R1 instruction preamble."""
    _, sep, rest = content.rpartition("Question:")
    if not sep:
        raise ValueError("user message without a Question: prefix")
    return _clean(rest.strip())


def _search_call(content: str) -> tuple[str, str]:
    """(thinking block, query) from an assistant turn that called search."""
    query = re.search(r"<search>\s*(.*?)\s*</search>", content, re.DOTALL)
    if not query:
        raise ValueError("assistant message without <search>…</search>")
    q = _clean(query.group(1))
    if not q:
        raise ValueError("empty search query")
    return _thinking(content), q


def _answer(content: str) -> tuple[str, str]:
    """(thinking block, answer) from the closing assistant turn."""
    ans = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
    if not ans:
        raise ValueError("assistant message without <answer>…</answer>")
    body = _clean(ans.group(1))
    if not body:
        raise ValueError("empty answer")
    return _thinking(content), body


def _documents(info: str, max_items: int, max_chars: int) -> str:
    """The `<|list|>` for one round of results, as `<|item|>` per document."""
    items: list[str] = []
    for match in DOC_RE.finditer(info):
        doc = _clean(match.group(1))
        if not doc:
            continue
        title = ""
        quoted = QUOTED_TITLE.match(doc)
        if quoted and len(quoted.group(1)) < len(quoted.group(2)) + 1:
            title, doc = quoted.group(1), quoted.group(2)
        item = f"{title} — {doc}" if title else doc
        items.append(truncated(item, max_chars))
        if len(items) >= max_items:
            break
    if not items:
        raise ValueError("information block with no parseable documents")
    body = "".join(
        f"{ITEM_OPEN}{item}{ITEM_CLOSE}" for item in items
    )
    return f"{LIST_OPEN}{body}{LIST_CLOSE}"


def convert(sample: dict, max_items: int, max_item_chars: int,
            max_thinking_chars: int) -> dict | None:
    """One Search-R1 sample to our conversation shape, or None if unusable.

    None when the sample never searched (it answers from memory, which teaches
    nothing about calling the tool) or when it does not close on an answer turn
    (the engine's `_turns` would reject it for an all-zero mask).
    """
    messages = sample.get("messages") or []
    system = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    ).strip()

    turns: list[dict] = []
    pending_query = None  # the search call the next <information> answers
    last_query = None     # the most recent call, for refinement rounds that
                          # feed more results for it without a fresh call
    searched = False
    closed = False
    asked = False  # the opening question, extracted once from the preamble

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        # A real result message *starts* with `<information>`; the instruction
        # preamble in the first user message mentions the marker in prose, so an
        # `in` test would mistake the question for a tool result.
        if role == "user" and content.strip().startswith("<information>"):
            # Some samples feed a second, re-ranked result set for the query the
            # model just answered provisionally — restate that query rather than
            # dropping the round.
            query = pending_query or last_query
            if query is None:
                return None
            _, _, info = content.partition("<information>")
            result = _call(query)
            result += _documents(info, max_items, max_item_chars)
            turns.append({"role": "tool", "text": result})
            pending_query = None
            continue
        if role == "user":
            if not asked:
                turns.append({"role": "user", "text": _question(content)})
                asked = True
            continue
        if role != "assistant":
            continue

        if "<search>" in content:
            thinking, query = _search_call(content)
            thinking = truncated(thinking, max_thinking_chars)
            turn = thinking + _call(query)
            turns.append({"role": "assistant", "text": turn})
            pending_query = query
            last_query = query
            searched = True
            continue

        if "<answer>" in content:
            thinking, answer = _answer(content)
            thinking = truncated(thinking, max_thinking_chars)
            if not answer:
                return None
            turns.append({
                "role": "assistant",
                "text": thinking + f"{PAR_OPEN}{answer}{PAR_CLOSE}",
            })
            closed = True
            continue

        return None

    if not searched or not closed or turns[-1]["role"] != "assistant":
        return None

    return {"turns": turns, "system": system, "source": "search-r1"}


def main() -> int:
    args = _arguments()
    source = fetch(args.file, args.revision)
    samples = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("data", samples.get("messages", []))
    print(f"{len(samples)} source samples loaded")

    picked = random.Random(args.seed).sample(samples, min(args.count, len(samples)))

    rows: list[dict] = []
    skipped = 0
    for sample in picked:
        try:
            row = convert(
                sample, args.max_items, args.max_item_chars, args.max_thinking_chars
            )
        except ValueError as err:
            print(f"  skip: {err}")
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    out_dir = storage.sft_dir() / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard, start in enumerate(range(0, len(rows), args.shard_size), start=0):
        chunk = rows[start : start + args.shard_size]
        if not chunk:
            break
        path = out_dir / f"{args.set}-{shard:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in chunk:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        print(f"{path.relative_to(storage.root())}: {len(chunk)} rows")

    print(f"{written} rows written to {out_dir.relative_to(storage.root())}"
          f" ({skipped} sampled but unusable)")
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000,
                        help="samples to convert (default 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="shuffle seed for reproducible selection (default 0)")
    parser.add_argument("--set", default=OUTPUT_SET,
                        help="output folder name under data/sft (default websearch)")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                        help="rows per output file (default 1000)")
    parser.add_argument("--revision", default=SOURCE_REVISION,
                        help="pinned commit of the source dataset (default embedded)")
    parser.add_argument("--file", default=SOURCE_FILE,
                        help="source file inside the dataset (default qwen2.5 SFT)")
    parser.add_argument("--max-items", type=int, default=3,
                        help="documents kept per search round (default 3)")
    parser.add_argument("--max-item-chars", type=int, default=240,
                        help="characters kept per document (default 240)")
    parser.add_argument("--max-thinking-chars", type=int, default=300,
                        help="characters kept per thinking block (default 300)")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())