#!/usr/bin/env python3
"""
Generate synthetic SFT chat data with a cheap LLM, via the `opencode` CLI.

Asks a cheap/free model for conversations, validates every line
against the backend's SFT schema AND the frontend's markup vocabulary, and
appends only the valid ones to `backend/data/raw/sft/synthetic.jsonl`. Bad
JSON, malformed conversations, or broken markup are dropped, so a flaky small
model can't corrupt the training set.

Since pretraining is plain prose, SFT is the only place the model learns the
document markup the frontend renders (lib/message-parser.ts + rich-message.tsx):

    <|par|>...<|/par|>                paragraph (every prose block)
    <|heading|> / <|subheading|>      section headings
    <|list|><|item|>...<|/item|>...   bulleted list
    <|codeblock|>...<|/codeblock|>    fenced code block
    <|code|>...<|/code|>              inline code
    <|bold|> / <|italic|>             emphasis
    <|quote|>...<|/quote|>            blockquote
    <|divider|>                       horizontal rule (void tag)
    <|link|>dest<|/link|>             hyperlink (the text is the destination)
    <|thinking|>...<|/thinking|>      hidden reasoning, start of a reply only

Assistant turns must be fully marked up (all prose inside <|par|> etc.); user
turns must be plain text — the chat template (backend/src/llm/src/chat_template.rs)
wraps turns in <|user|>/<|assistant|>/<|system|> itself, so those structural
tokens are rejected anywhere in the data.

Each model call gets a freshly sampled mix of topics, conversation shapes, and
markup features, so batches vary instead of repeating the same few Q&As, and
conversations duplicating an existing first user message (or an entire existing
conversation) are dropped.

Model output must be JSONL — one conversation per line — matching the schema the
Rust side deserializes (see `SftConversation` / `SftTurn` in
backend/src/training.rs):

    {"system"?: string, "turns": [{"role": "user"|"assistant", "text": string}, ...]}

No API key needed with the default free model. Requires the `opencode` CLI on
PATH (https://opencode.ai). Pick any model with `--model provider/name`.

Usage:
    python3 generate_sft_llm.py --count 200
    python3 generate_sft_llm.py --count 500 --batch 10 --think-frac 0.4
    python3 generate_sft_llm.py --model opencode/deepseek-v4-flash-free --count 100
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys

DEFAULT_OUT = "/home/x_x/Programming/LLM/backend/data/raw/sft/synthetic.jsonl"
DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"

# Strips ANSI escape codes (color, cursor moves) from opencode's TUI output so
# only plain text remains for JSON parsing.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# ---------------------------------------------------------------------------
# Markup vocabulary (must stay in sync with frontend/lib/message-parser.ts and
# SPECIAL_TOKENS in backend/src/tokenize/src/pretokenize.rs)
# ---------------------------------------------------------------------------

# Paired tags allowed in assistant replies. `title` is deliberately left out:
# it renders as a document title and chat replies shouldn't produce one.
PAIRED_TAGS = {
    "par", "heading", "subheading", "code", "codeblock",
    "list", "item", "bold", "italic", "quote", "link",
}
VOID_TAGS = {"divider"}

TAG_RE = re.compile(r"<\|(.*?)\|>")
THINKING_LEAD_RE = re.compile(r"^\s*<\|thinking\|>(.*?)<\|/thinking\|>", re.S)

MAX_TURN_CHARS = 1500

# ---------------------------------------------------------------------------
# Prompt ingredients — sampled per call so batches vary
# ---------------------------------------------------------------------------

TOPICS = [
    "cooking basics and fixing a dish that went wrong",
    "houseplant or garden care",
    "planning a short trip or packing",
    "starting an exercise habit",
    "a simple Python snippet or explanation",
    "a simple JavaScript snippet or explanation",
    "explaining a shell command or git basics",
    "writing or rewording a short email or message",
    "grammar, word choice, or spelling",
    "explaining a science concept simply",
    "a bit of practical history or geography",
    "budgeting or everyday money questions",
    "pet care and training",
    "sleep, focus, or everyday health habits",
    "board games, chess, or puzzle rules",
    "music practice or learning an instrument",
    "basic home or bike repair",
    "book, movie, or hobby recommendations",
    "simple arithmetic or a short word problem",
    "units, conversions, or everyday measurements",
    "explaining a computing concept (files, memory, networks) simply",
    "comparing two everyday options and their trade-offs",
    "study techniques or preparing for a test",
    "meal planning or grocery shopping",
    "weather, seasons, or nature questions",
    "organizing, decluttering, or small productivity systems",
]

SHAPES = [
    "a single question answered in one structured reply",
    "a question, then a follow-up asking for more detail or an example",
    "a question, then a follow-up asking to make the answer shorter or simpler",
    "a vague request where the assistant asks one clarifying question, the user "
    "clarifies, and the assistant then answers properly",
    "the user states something slightly wrong; the assistant gently corrects it",
    "a request for step-by-step instructions",
    "a request to compare two options",
    "a request that is best answered with a short code example",
]

FEATURES = [
    "a bulleted <|list|> of <|item|>s",
    "a <|codeblock|> code block (no language attribute — just the code inside)",
    "<|code|> inline code for a command, key, or term",
    "<|bold|> on the one or two key terms",
    "<|italic|> for light emphasis or a title",
    "a <|quote|> block quoting a saying or the user's own words back",
    "an answer in two parts under <|heading|> or <|subheading|> headings",
    "a <|divider|> between two distinct sections",
    "a <|link|>example.com/page<|/link|> whose text is the destination itself",
    "plain short <|par|> paragraphs only, no other markup",
]

SYSTEM_PROMPT_NOTE = (
    'Optionally include a "system" string key on some lines, e.g. '
    '{"system": "You are a helpful assistant. Answer concisely.", "turns": [...]}.'
)

PROMPT_TEMPLATE = """\
You are generating supervised training data for a small chat model. Output ONLY
JSONL: exactly @BATCH@ lines, each a single complete JSON object on its own
line, no prose, no markdown fences, no commentary before or after.

Each object is one chat conversation:
  {"turns": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}]}
@SYSTEM_NOTE@

Hard rules (a line breaking any of these is discarded):
- roles alternate user/assistant, start with "user", END on "assistant",
- 1 to 3 exchanges per conversation; keep every turn under ~150 words,
- user turns are PLAIN text: natural, casual, sometimes terse or with a typo —
  never any <|...|> tokens,
- assistant turns are written ENTIRELY in this markup — every piece of prose
  wrapped in a tag, no bare text between blocks:
    <|par|>a paragraph<|/par|>
    <|list|><|item|>bullet<|/item|><|item|>bullet<|/item|><|/list|>
    <|codeblock|>code here\\n<|/codeblock|>
    <|code|>inline code<|/code|>  <|bold|>bold<|/bold|>  <|italic|>italic<|/italic|>
    <|heading|>section heading<|/heading|>  <|subheading|>smaller heading<|/subheading|>
    <|quote|>a quoted line<|/quote|>  <|divider|>
    <|link|>example.com/page<|/link|>
  Never use <|user|>, <|assistant|>, <|system|>, <|endoftext|>, or any tag not
  listed above. Every opened tag must be closed. Escape newlines in JSON as \\n.
  Write every tag BARE, with no attributes: use <|codeblock|> (not
  <|codeblock|lang='python'|>) and <|link|>example.com/x<|/link|> (not
  <|link|href='...'|>). A tag containing '=' or a second '|' is rejected.

For about @THINK_PCT@% of conversations, the FINAL assistant reply must START
with a hidden reasoning block before the first <|par|>:
  <|thinking|>...two to five sentences of genuine reasoning about THIS question:
  what the user really needs, what to include or leave out, edge cases...<|/thinking|>
The reasoning must be specific to the question — never a generic filler phrase.
No <|...|> tags inside the thinking block. The other conversations have no
thinking block.

Content requirements for this batch:
- Spread the conversations across these topics: @TOPICS@.
- Use these conversation shapes across the batch: @SHAPES@.
- Across the batch, exercise these markup features where they genuinely fit:
  @FEATURES@. Do not force a feature where it reads unnaturally — a simple
  question still gets a simple <|par|> answer.
- Answers must be correct, concise, and self-contained; no real personal data,
  no current events.

Example of one fully valid line (yours must differ in topic and wording):
{"turns": [{"role": "user", "text": "my coffee always tastes bitter, what am i doing wrong"}, {"role": "assistant", "text": "<|thinking|>Bitterness is almost always over-extraction: grind too fine, water too hot, or steeping too long. The user sounds like a beginner, so I'll give the two easiest fixes first and skip ratios.<|/thinking|><|par|>Bitter coffee usually means it's over-extracted. Two easy fixes:<|/par|><|list|><|item|>Use a slightly coarser grind.<|/item|><|item|>Let boiled water sit ~30 seconds before pouring so it's around 93\\u00b0C.<|/item|><|/list|><|par|>If it's still bitter, cut the brew time by about 30 seconds.<|/par|>"}]}

Output the @BATCH@ JSONL lines now:"""


def build_prompt(batch: int, think_pct: int, rng: random.Random) -> str:
    topics = "; ".join(rng.sample(TOPICS, k=min(5, batch)))
    shapes = "; ".join(rng.sample(SHAPES, k=min(3, batch)))
    features = "; ".join(rng.sample(FEATURES, k=min(4, batch)))
    return (
        PROMPT_TEMPLATE
        .replace("@BATCH@", str(batch))
        .replace("@THINK_PCT@", str(think_pct))
        .replace("@SYSTEM_NOTE@", SYSTEM_PROMPT_NOTE)
        .replace("@TOPICS@", topics)
        .replace("@SHAPES@", shapes)
        .replace("@FEATURES@", features)
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def valid_assistant_markup(text: str) -> bool:
    """Well-formed reply markup: optional leading thinking block, then balanced
    tags from the allowed vocabulary, with at least one <|par|>."""
    # At most one thinking block, only at the very start, plain prose inside.
    m = THINKING_LEAD_RE.match(text)
    body = text[m.end():] if m else text
    if "<|thinking" in body or "/thinking" in body:
        return False
    if m and (not m.group(1).strip() or "<|" in m.group(1)):
        return False  # empty reasoning, or tags nested inside the block

    tags = TAG_RE.findall(body)
    # Every "<|" / "|>" must belong to a well-formed tag — catches danglers.
    if body.count("<|") != len(tags) or body.count("|>") != len(tags):
        return False

    stack = []
    for inner in tags:
        closing = inner.startswith("/")
        raw = inner[1:] if closing else inner
        # Bare tags only. An attribute form like `codeblock|lang='python'` or
        # `link|href='...'` no longer tokenizes to a single id (it byte-shreds
        # into the raw markup), so it's rejected outright.
        if "|" in raw or "=" in raw:
            return False
        name = raw.strip()
        if closing:
            if not stack or stack.pop() != name:
                return False
        elif name in VOID_TAGS:
            continue
        elif name in PAIRED_TAGS:
            stack.append(name)
        else:
            return False  # unknown or structural token (<|user|>, <|endoftext|>, …)
    if stack:
        return False

    return "par" in (t.split("|")[0] for t in tags)


def valid_conversation(obj) -> bool:
    """True if obj matches the SFT schema, the turn rules, and the markup rules."""
    if not isinstance(obj, dict) or "turns" not in obj:
        return False
    system = obj.get("system")
    if "system" in obj and (not isinstance(system, str) or "<|" in system):
        return False
    turns = obj["turns"]
    if not isinstance(turns, list) or not turns:
        return False
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            return False
        role, text = t.get("role"), t.get("text")
        if role not in ("user", "assistant"):
            return False
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TURN_CHARS:
            return False
        if role != ("user" if i % 2 == 0 else "assistant"):
            return False
        if role == "user":
            if "<|" in text or "|>" in text:
                return False
        elif not valid_assistant_markup(text):
            return False
    return turns[-1]["role"] == "assistant"


def parse_conversations(raw):
    """Pull every valid conversation object out of model output, line by line.

    Tolerant by design: the model may emit a banner, blank lines, or stray
    prose. We try to JSON-parse each cleaned line and keep only the ones that
    both parse and pass validation.
    """
    out = []
    for line in ANSI_RE.sub("", raw).splitlines():
        line = line.strip().rstrip(",")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if valid_conversation(obj):
            # Normalize: drop any keys beyond system/turns/role/text.
            conv = {"turns": [{"role": t["role"], "text": t["text"]}
                              for t in obj["turns"]]}
            if isinstance(obj.get("system"), str):
                conv = {"system": obj["system"], **conv}
            out.append(conv)
    return out


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def opener_key(conv) -> str:
    """Normalized first user message — near-duplicate conversations almost
    always share their opening question verbatim."""
    return re.sub(r"\s+", " ", conv["turns"][0]["text"].strip().lower())


def load_seen(path) -> set:
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                conv = json.loads(line)
                seen.add(opener_key(conv))
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
    return seen


def run_model(model, prompt, timeout):
    """Invoke `opencode run` and return its stdout, or '' on failure."""
    try:
        proc = subprocess.run(
            ["opencode", "run", "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  model call failed: {e}", file=sys.stderr)
        return ""
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description="Generate SFT data via a cheap LLM (opencode).")
    ap.add_argument("--count", type=int, default=200,
                    help="target number of valid conversations to add (default 200)")
    ap.add_argument("--batch", type=int, default=10,
                    help="conversations requested per model call (default 10; "
                         "marked-up replies are long, keep batches small)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"opencode model id (default {DEFAULT_MODEL})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output .jsonl path (default {DEFAULT_OUT})")
    ap.add_argument("--think-frac", type=float, default=0.35,
                    help="fraction of replies asked to include a thinking block (default 0.35)")
    ap.add_argument("--timeout", type=int, default=240,
                    help="per-call timeout in seconds (default 240)")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="safety cap on model calls; 0 = count/batch * 3")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for topic/shape sampling (default: random)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    think_pct = max(0, min(100, round(args.think_frac * 100)))
    max_calls = args.max_calls or max(1, (args.count // args.batch + 1) * 3)

    seen = load_seen(args.out)
    added = 0
    calls = 0
    dupes = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while added < args.count and calls < max_calls:
            calls += 1
            prompt = build_prompt(args.batch, think_pct, rng)
            convs = parse_conversations(run_model(args.model, prompt, args.timeout))
            if not convs:
                print(f"  call {calls}: 0 valid lines (model gave nothing usable)",
                      file=sys.stderr)
                continue
            fresh = 0
            for conv in convs:
                if added >= args.count:
                    break
                key = opener_key(conv)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                added += 1
                fresh += 1
            f.flush()
            print(f"  call {calls}: +{fresh} valid  (total {added}/{args.count}, "
                  f"{dupes} dupes skipped)")

    print(f"Added {added} conversations to {args.out} in {calls} model call(s)")
    if added < args.count:
        print("Stopped short of target — raise --max-calls, or the model is "
              "returning too little valid JSONL (try a different --model).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
