#!/usr/bin/env python3
"""
Generate synthetic SFT chat data with a cheap LLM, via the `opencode` CLI.

The hand-curated conversations in `engine/data/sft/synthetic.jsonl` are the
gold set: real coding-agent sessions (long pasted context, terse expert replies,
tool-driven investigations). They are the *style target*, and they are never
written to by this script — generated output goes to a sibling file
(`synthetic_llm.jsonl` by default) so the gold set can never be corrupted or
diluted by a flaky small model. The Rust loader (`build_sft_dataset` in
backend/src/training.rs) reads every `.jsonl` in the SFT dir, so both files
train.

How the style is transferred: every model call embeds two or three real gold
conversations verbatim as few-shot exemplars, freshly sampled per call. Telling a
small model "write like a senior engineer in a coding CLI" does not work; showing
it does.

What this adds on top of the gold set: **large thinking blocks**. The gold
conversations barely reason out loud (3 of 47 assistant turns, ~25 words each).
Most generated replies are required to open with a substantial `<|thinking|>`
block — genuine multi-step reasoning about the specific request — and blocks
shorter than `--min-think-words` are discarded.

Every conversation is validated against the backend's SFT schema AND the
frontend's markup vocabulary before it is written; malformed conversations and
broken markup are dropped, with a per-call tally of why.

Markup vocabulary (frontend/lib/message-parser.ts + rich-message.tsx):

    <|par|>...<|/par|>                paragraph
    <|heading|> / <|subheading|>      section headings
    <|list|><|item|>...<|/item|>...   bulleted list
    <|codeblock|>...<|/codeblock|>    fenced code block
    <|code|>...<|/code|>              inline code
    <|bold|> / <|italic|>             emphasis
    <|quote|>...<|/quote|>            blockquote
    <|divider|>                       horizontal rule (void tag)
    <|link|>dest<|/link|>             hyperlink (the text is the destination)
    <|thinking|>...<|/thinking|>      hidden reasoning, start of a reply only

Note the markup is *sprinkled*, not total: the parser renders bare text at the
root fine, and the gold replies mix plain prose, markdown-ish lists and headings
with inline `<|code|>`/`<|bold|>`. So a reply is not required to wrap every
paragraph in `<|par|>` — it only has to use the vocabulary correctly and use at
least some of it. User turns must be plain text; the chat template
(backend/src/llm/src/chat_template.rs) supplies <|user|>/<|assistant|>/<|system|>
itself, so those structural tokens are rejected anywhere in the data.

**The model is asked for delimiters, not JSON** (see the `Wire format` section):

    <<<CONV>>>
    <<<USER>>>
    why is my run stalling?
    <<<ASSISTANT>>>
    <|thinking|>...<|/thinking|><|par|>...<|/par|>
    <<<END>>>

Turn text is written literally — no escaping — and this script assembles the
JSON the Rust side deserializes (`SftConversation` / `SftTurn` in
backend/src/training.rs):

    {"system"?: string, "turns": [{"role": "user"|"assistant", "text": string}, ...]}

**Two backends.** `--backend opencode` (default) shells out to the opencode CLI;
it needs no API key with the default free model, but must run a TOOL-FREE agent
(see `run_model`). `--backend http` posts to any OpenAI-compatible
`/chat/completions` endpoint — DeepSeek's own API, OpenRouter, Groq, Google's
compatibility layer, or a local Ollama — which avoids the CLI's agent machinery
and its rate limits entirely.

Usage:
    python3 generate_sft_llm.py --count 200
    python3 generate_sft_llm.py --count 500 --batch 2 --min-think-words 150

    export SFT_API_KEY=sk-...
    python3 generate_sft_llm.py --backend http \
        --base-url https://api.deepseek.com/v1 --model deepseek-chat --count 200

    # local, no key, no quota
    python3 generate_sft_llm.py --backend http \
        --base-url http://localhost:11434/v1 --model qwen3:8b --count 50
"""

import paths
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

SFT_DIR = paths.sft_dir()
DEFAULT_OUT = os.path.join(SFT_DIR, "synthetic_llm.jsonl")
DEFAULT_EXEMPLARS = os.path.join(SFT_DIR, "synthetic.jsonl")
DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"
# Tool-free: see `run_model` for why the default `build` agent cannot be used.
DEFAULT_AGENT = "summary"

# Seconds to wait after a 429 or a 5xx before the next call.
RATE_LIMIT_BACKOFF_SECS = 20

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

# Assistant replies in the gold set top out around 4.7k chars; user turns run to
# 109k because they carry pasted logs, file dumps and task briefs. Cap the
# assistant side tightly (a small model padding past this is rambling) and the
# user side loosely.
MAX_ASSISTANT_CHARS = 6000
MAX_USER_CHARS = 30000

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

# The model is asked for delimiter-separated turns, not JSON.
#
# JSON was the obvious choice and the wrong one. A conversation has to survive
# newlines, quotes and backslashes inside every `text` value, and a small model
# gets that escaping wrong often enough that most of a batch is lost — the failure
# tally on the previous format was dominated by "bad JSON", on output whose actual
# *content* was fine. Delimiters remove the problem rather than police it: turn
# text is written literally, and this script assembles the JSON itself.
#
# `<<<...>>>` because it is vanishingly unlikely inside a reply about code, and
# because a delimiter alone on its own line is trivial for a weak model to place
# correctly — far easier than balancing braces across a 3000-character string.
D_CONV = "<<<CONV>>>"
D_SYSTEM = "<<<SYSTEM>>>"
D_USER = "<<<USER>>>"
D_ASSISTANT = "<<<ASSISTANT>>>"
D_END = "<<<END>>>"
DELIMITERS = (D_CONV, D_SYSTEM, D_USER, D_ASSISTANT, D_END)

# ---------------------------------------------------------------------------
# Prompt ingredients — sampled per call so batches vary
# ---------------------------------------------------------------------------

# Drawn from the gold set's actual domain: an agentic coding CLI driving model
# training runs, codebase work, and tool use.
TOPICS = [
    "debugging a stalled or crashing model training run",
    "reading an unfamiliar codebase and explaining its layout",
    "a CUDA/GPU out-of-memory error and how to cut memory use",
    "tokenizer, vocabulary, or BPE merge behaviour",
    "benchmark or leaderboard numbers that moved unexpectedly",
    "a dataset pipeline producing wrong or truncated records",
    "an agent CLI misusing its tools or looping on a tool call",
    "a TUI/terminal rendering bug (scroll, layout, redraw)",
    "checkpointing, resuming, or merging model weights",
    "learning-rate, batch-size, or schedule tuning",
    "an audit for bugs in a file the user names, read-only",
    "a git problem: bad merge, lost commit, messy history",
    "a shell one-liner or script that does the wrong thing",
    "profiling something slow and finding the real bottleneck",
    "a flaky or wrong test and what it actually proves",
    "refactoring a module without changing behaviour",
    "an API/library call returning an error the user pasted",
    "explaining a model-architecture detail (attention, RoPE, MoE, LoRA)",
    "evaluating whether a paper's technique is worth trying here",
    "a data-format question: JSONL schema, encoding, escaping",
    "concurrency: a race, a deadlock, or a worker pool stalling",
    "deciding between two implementation approaches under a constraint",
    "a regex or parser that misses an edge case",
    "reading logs or a stack trace the user pasted and locating the cause",
    "distillation, SFT, or RL fine-tuning strategy questions",
    "disk/IO, caching, or a run directory filling up",
]

SHAPES = [
    "a long pasted task brief (file paths, constraints, logs) answered with a "
    "short plan and a first concrete step",
    "a terse one-line question ('where is the file?', 'why a G?') answered in "
    "two or three dense sentences, no preamble",
    "a question, then a terse follow-up ('but why?', 'and if it fails?') that "
    "the assistant answers without re-explaining what it already said",
    "the user pastes an error or log; the assistant names the cause and the fix",
    "the user's premise is wrong; the assistant says so plainly and corrects it",
    "a multi-turn investigation: assistant states what it will check, user says "
    "'continue', assistant reports what it found",
    "a request to compare two approaches, ending on a recommendation",
    "a request for a short code change, answered with a codeblock and one note",
    "a vague request; the assistant asks one sharp clarifying question, the user "
    "answers, the assistant then delivers",
]

FEATURES = [
    "a bulleted <|list|> of <|item|>s",
    "a <|codeblock|> code block (no language attribute — just the code inside)",
    "<|code|> inline code for a path, command, flag, or symbol",
    "<|bold|> on the one or two key terms",
    "<|italic|> for light emphasis",
    "an answer in two parts under <|heading|> or <|subheading|> headings",
    "a <|divider|> between two distinct sections",
    "a <|quote|> block quoting the user's own words or an error string back",
    "plain prose with only inline <|code|> and <|bold|>, no block structure",
]

# The gold set has no system prompts; these mirror the implicit persona in it.
SYSTEM_PROMPTS = [
    "You are a coding copilot embedded in a terminal CLI. Be terse and technical.",
    "You are an ML engineering assistant. Answer with specifics, no filler.",
    "You are a senior engineer pair-programming over a shared repo. Be direct.",
]

PROMPT_TEMPLATE = """\
You are generating supervised training data for a small chat model that acts as
a CODING AGENT in a terminal CLI: it reads codebases, drives model-training runs,
uses tools, and answers like a senior engineer — dense, specific, no filler.

Below are @N_EX@ REAL conversations from the target dataset. Study their voice,
length, density and markup style closely. Your output must look like it came
from the same corpus.

--- REAL EXAMPLES ---
@EXEMPLARS@
--- END EXAMPLES ---

Output exactly @BATCH@ conversations in the SAME delimiter format as the examples
above. No prose, no markdown fences, no commentary before or after.

Every delimiter sits ALONE on its own line, with nothing else on that line:

@CONV@
@USER@
the user's message, as many lines as it needs
@ASSISTANT@
the reply, as many lines as it needs
@END@
@SYSTEM_NOTE@
Write turn text literally — real newlines, real quotes, no escaping of any kind.
Nothing is JSON here.

Hard rules (a conversation breaking any of these is discarded):
- roles alternate user/assistant, start with "user", END on "assistant",
- 1 to 4 exchanges per conversation,
- user turns are PLAIN text and realistic: sometimes a long pasted brief with
  file paths, logs, or requirements; sometimes three terse words with a typo.
  Never any <|...|> tokens in a user turn,
- assistant turns are under 6000 characters, written in the voice of the
  examples: lead with the answer, no "Great question!", no restating the ask,
- assistant markup uses ONLY this vocabulary:
    <|par|>a paragraph<|/par|>
    <|list|><|item|>bullet<|/item|><|item|>bullet<|/item|><|/list|>
    <|codeblock|>code here\\n<|/codeblock|>
    <|code|>inline code<|/code|>  <|bold|>bold<|/bold|>  <|italic|>italic<|/italic|>
    <|heading|>section heading<|/heading|>  <|subheading|>smaller heading<|/subheading|>
    <|quote|>a quoted line<|/quote|>  <|divider|>
    <|link|>example.com/page<|/link|>
  As in the examples, markup is sprinkled rather than total — plain prose and
  markdown-style lines between tags are fine — but every reply must use at least
  some of it, and every opened tag must be closed.
  Never use <|user|>, <|assistant|>, <|system|>, <|endoftext|>, or any tag not
  listed above. Write every tag BARE, with no attributes: <|codeblock|> (not
  <|codeblock|lang='python'|>) and <|link|>example.com/x<|/link|> (not
  <|link|href='...'|>). A tag containing '=' or a second '|' is rejected.
  Closing tags are spelled <|/par|> — the slash goes INSIDE, after the bar.
  </|par|> is wrong.

THINKING BLOCKS — the most important part of this task.
For about @THINK_PCT@% of conversations, the FINAL assistant reply must START
with a hidden reasoning block, before any other tag:

  <|thinking|>...<|/thinking|>

That block must be LONG and SUBSTANTIVE: at least @MIN_WORDS@ words, typically
@TYP_WORDS@. It is the model's private scratchpad, so write it as real working
thought, not a summary of the answer. Work through, in order:
  1. what the user is actually asking for, and what they are not,
  2. what the concrete details (paths, numbers, error strings, versions) imply,
  3. two or three candidate explanations or approaches, weighed against each
     other, with at least one explicitly rejected and the reason why,
  4. edge cases, failure modes, or what could make the chosen answer wrong,
  5. a decision about what to actually say and what to leave out.
Reason about THIS specific question — cite the user's own numbers, paths and
symbols. Change your mind mid-block where it is natural to. Never write generic
filler ("Let me think about this carefully", "The user wants help with X").
No <|...|> tags inside the thinking block; plain prose and \\n only.
The reply AFTER the block must stand on its own and stay tight — the length
goes in the thinking, not the answer. Conversations without a thinking block
have no reasoning leakage in the visible reply either.

Content requirements for this batch:
- Spread the conversations across these topics: @TOPICS@.
- Use these conversation shapes across the batch: @SHAPES@.
- Where they genuinely fit, exercise: @FEATURES@. Never force a feature — a
  terse question still gets a terse answer.
- Technical content must be CORRECT and self-contained. Invent plausible file
  paths, run directories, and numbers; no real personal data, no current events.

Output the @BATCH@ conversations now, starting with @CONV@:"""


def render_exemplar(conv) -> str:
    """A gold conversation in the delimiter format the model must reproduce.

    The exemplars have to be shown in the *output* format, not as JSON — a
    demonstration in one format while asking for another is most of why the JSON
    version failed.
    """
    parts = [D_CONV]
    if isinstance(conv.get("system"), str) and conv["system"].strip():
        parts += [D_SYSTEM, conv["system"].strip()]
    for turn in conv["turns"]:
        parts.append(D_USER if turn["role"] == "user" else D_ASSISTANT)
        parts.append(turn["text"])
    parts.append(D_END)
    return "\n".join(parts)


def build_prompt(batch, think_pct, min_words, exemplars, rng, system_frac):
    topics = "; ".join(rng.sample(TOPICS, k=min(5, len(TOPICS))))
    shapes = "; ".join(rng.sample(SHAPES, k=min(3, len(SHAPES))))
    features = "; ".join(rng.sample(FEATURES, k=min(4, len(FEATURES))))
    system_note = ""
    if rng.random() < system_frac:
        system_note = (
            f"\nOn ONE conversation only, put a {D_SYSTEM} block immediately after "
            f"{D_CONV}, before the first {D_USER}, e.g. "
            f"\"{rng.choice(SYSTEM_PROMPTS)}\".\n"
        )
    ex_text = "\n\n".join(render_exemplar(e) for e in exemplars)
    return (
        PROMPT_TEMPLATE
        .replace("@EXEMPLARS@", ex_text)
        .replace("@N_EX@", str(len(exemplars)))
        .replace("@BATCH@", str(batch))
        .replace("@THINK_PCT@", str(think_pct))
        .replace("@MIN_WORDS@", str(min_words))
        .replace("@TYP_WORDS@", f"{min_words + 60}-{min_words + 220}")
        .replace("@SYSTEM_NOTE@", system_note)
        .replace("@TOPICS@", topics)
        .replace("@SHAPES@", shapes)
        .replace("@FEATURES@", features)
        # Delimiters last: they must not be rewritten by an earlier replacement.
        .replace("@CONV@", D_CONV)
        .replace("@SYSTEM@", D_SYSTEM)
        .replace("@USER@", D_USER)
        .replace("@ASSISTANT@", D_ASSISTANT)
        .replace("@END@", D_END)
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# The closer this vocabulary uses is `<|/par|>`. Models reliably reach for the
# XML-shaped `</|par|>` instead — it is the form every other markup language uses,
# and no amount of showing the right one in the prompt fully suppresses it. The
# rewrite is unambiguous and lossless, so repairing beats discarding an otherwise
# good conversation over a transposed slash.
BAD_CLOSER_RE = re.compile(r"</\|(\w+)\|>")


def repair_markup(text: str) -> str:
    return BAD_CLOSER_RE.sub(r"<|/\1|>", text)


def repair_conversation(obj):
    """Apply `repair_markup` to assistant turns, in place, and return obj.

    Only assistant turns: a user turn containing markup is a rule violation, not a
    typo, and repairing it would hide that.
    """
    if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
        for turn in obj["turns"]:
            if isinstance(turn, dict) and turn.get("role") == "assistant" \
                    and isinstance(turn.get("text"), str):
                turn["text"] = repair_markup(turn["text"])
    return obj


def thinking_words(text) -> int:
    """Word count of a leading thinking block, or 0 if there isn't one."""
    m = THINKING_LEAD_RE.match(text)
    return len(m.group(1).split()) if m else 0


def valid_assistant_markup(text: str, min_think_words: int = 0) -> bool:
    """Well-formed reply markup: an optional leading thinking block, then
    balanced tags drawn from the allowed vocabulary.

    Bare prose between tags is allowed — the parser renders root-level text, and
    the gold conversations rely on that — but a reply with no markup at all is
    rejected, since SFT is the only place the model learns the vocabulary.
    """
    # At most one thinking block, only at the very start, plain prose inside.
    m = THINKING_LEAD_RE.match(text)
    body = text[m.end():] if m else text
    if "<|thinking" in body or "/thinking" in body:
        return False
    if m:
        inner = m.group(1)
        if not inner.strip() or "<|" in inner:
            return False  # empty reasoning, or tags nested inside the block
        if len(inner.split()) < min_think_words:
            return False  # a token gesture at reasoning, not a thinking block
        if not body.strip():
            return False  # reasoned, then said nothing

    tags = TAG_RE.findall(body)
    if not tags:
        return False
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
    return not stack


def valid_conversation(obj, min_think_words: int = 0) -> bool:
    """True if obj matches the SFT schema, the turn rules, and the markup rules.

    `min_think_words` is enforced only on turns that actually open with a
    thinking block — conversations are allowed to have none at all.
    """
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
        if not isinstance(text, str) or not text.strip():
            return False
        if role != ("user" if i % 2 == 0 else "assistant"):
            return False
        if role == "user":
            if len(text) > MAX_USER_CHARS or "<|" in text or "|>" in text:
                return False
        else:
            if len(text) > MAX_ASSISTANT_CHARS:
                return False
            if not valid_assistant_markup(text, min_think_words):
                return False
    return turns[-1]["role"] == "assistant"


def split_delimited(text):
    """Conversation dicts read out of delimiter-formatted output.

    Tolerant by design: a banner, a stray sentence, or a markdown fence between
    conversations is ignored, because only lines that are exactly a delimiter
    carry structure. A trailing conversation with no `<<<END>>>` (the response ran
    out of tokens) is still returned — it is only dropped later if it fails
    validation, which a half-written reply usually will.
    """
    convs = []
    current = None      # {"system": str|None, "turns": [...]}
    role = None         # which delimiter we are collecting text for
    buf = []

    def flush():
        if current is None or role is None:
            return
        body = "\n".join(buf).strip()
        if not body:
            return
        if role == "system":
            current["system"] = body
        else:
            current["turns"].append({"role": role, "text": body})

    for line in text.splitlines():
        marker = line.strip()
        if marker in DELIMITERS:
            flush()
            buf = []
            if marker == D_CONV:
                if current is not None and current["turns"]:
                    convs.append(current)
                current = {"system": None, "turns": []}
                role = None
            elif marker == D_END:
                if current is not None and current["turns"]:
                    convs.append(current)
                current, role = None, None
            elif current is None:
                # A role delimiter with no <<<CONV>>> before it: the model started
                # straight in. Treat it as an implicit conversation start rather
                # than discarding everything that follows.
                current = {"system": None, "turns": []}
                role = {D_SYSTEM: "system", D_USER: "user",
                        D_ASSISTANT: "assistant"}[marker]
            else:
                role = {D_SYSTEM: "system", D_USER: "user",
                        D_ASSISTANT: "assistant"}[marker]
            continue
        if role is not None:
            buf.append(line)

    flush()
    if current is not None and current["turns"]:
        convs.append(current)

    # Drop the `system: None` key so the shape matches what validation expects.
    return [
        ({"system": c["system"], "turns": c["turns"]} if c["system"]
         else {"turns": c["turns"]})
        for c in convs
    ]


def parse_conversations(raw, min_think_words: int = 0):
    """Pull every valid conversation out of model output; return (kept, reasons).

    Delimiter format first (what the prompt asks for); if the model ignored it and
    emitted JSON anyway, fall back to brace extraction rather than lose the batch.

    `reasons` is a Counter over why conversations were dropped. Without it every
    distinct failure — a timeout, a model answering in prose, a truncated reply, a
    rule violation — printed the same "0 valid lines (model gave nothing usable)",
    which is exactly the information needed to tell a broken harness from a weak
    model.
    """
    out = []
    reasons = Counter()
    text = ANSI_RE.sub("", raw)

    candidates = split_delimited(text)
    if not candidates:
        for blob in extract_json_objects(text):
            try:
                candidates.append(json.loads(blob))
            except json.JSONDecodeError:
                reasons["bad JSON"] += 1
    if not candidates and text.strip():
        reasons["no conversations in output"] += 1

    for obj in candidates:
        obj = repair_conversation(obj)
        if valid_conversation(obj, min_think_words):
            # Normalize: drop any keys beyond system/turns/role/text.
            conv = {"turns": [{"role": t["role"], "text": t["text"]}
                              for t in obj["turns"]]}
            if isinstance(obj.get("system"), str):
                conv = {"system": obj["system"], **conv}
            out.append(conv)
        else:
            reasons[reject_reason(obj, min_think_words)] += 1
    return out, reasons


def extract_json_objects(text):
    """Every balanced top-level `{...}` in `text`, in order.

    Replaces line-based parsing, which required each conversation to be exactly one
    line beginning `{` and ending `}`. Models do not reliably comply: they
    pretty-print across lines, wrap the batch in ```json fences, or prepend a
    sentence of commentary — and each of those silently discarded a conversation
    that was otherwise perfectly good. Scanning for balanced braces reads all three
    shapes, and is what makes this script portable to a provider other than
    opencode.

    String-aware, so a brace inside a `text` value cannot unbalance the scan; an
    unterminated final object (a truncated response) is simply not yielded.
    """
    out = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
                    start = None
    return out


def reject_reason(obj, min_think_words: int) -> str:
    """Why `valid_conversation` said no — for the run's failure tally."""
    if not isinstance(obj, dict) or not isinstance(obj.get("turns"), list) \
            or not obj["turns"]:
        return "no turns"
    turns = obj["turns"]
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            return "malformed turn"
        role, text = t.get("role"), t.get("text")
        if role not in ("user", "assistant") or not isinstance(text, str):
            return "malformed turn"
        if role != ("user" if i % 2 == 0 else "assistant"):
            return "roles out of order"
        if role == "user":
            if "<|" in text or "|>" in text:
                return "markup in a user turn"
        else:
            if len(text) > MAX_ASSISTANT_CHARS:
                return "assistant turn too long"
            words = thinking_words(text)
            if words and words < min_think_words:
                return f"thinking block under {min_think_words} words"
            if not valid_assistant_markup(text, min_think_words):
                return "bad markup"
    if turns[-1]["role"] != "assistant":
        return "does not end on assistant"
    return "rejected"


# ---------------------------------------------------------------------------
# Exemplars — the gold conversations shown to the model each call
# ---------------------------------------------------------------------------

# Gold lines that are transcript noise rather than training targets: bare API
# errors, model-fallback JSON blobs, empty tool-result echoes. Showing these to
# the generator would teach it to emit error strings as answers.
JUNK_RE = re.compile(
    r"API Error|does not support tools|may not exist or you may not have access"
    r'|"type":\s*"fallback"|\{"text":\s*""',
    re.I,
)


def load_exemplars(path, max_chars):
    """Gold conversations worth imitating, small enough to fit in a prompt."""
    pool = []
    if not os.path.exists(path):
        return pool
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                turns = conv["turns"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if len(json.dumps(conv, ensure_ascii=False)) > max_chars:
                continue  # a 100k pasted-log conversation won't fit a prompt
            asst = "".join(t["text"] for t in turns if t.get("role") == "assistant")
            if len(asst) < 150 or JUNK_RE.search(asst):
                continue
            pool.append(conv)
    return pool


def pick_exemplars(pool, rng, budget):
    """Sample two or three exemplars fitting a combined character budget."""
    if not pool:
        return []
    picks, used = [], 0
    for conv in rng.sample(pool, k=len(pool)):
        size = len(json.dumps(conv, ensure_ascii=False))
        if used + size > budget:
            continue
        picks.append(conv)
        used += size
        if len(picks) >= 3:
            break
    return picks or [min(pool, key=lambda c: len(json.dumps(c)))]


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def opener_key(conv) -> str:
    """Normalized first user message — near-duplicate conversations almost
    always share their opening question verbatim."""
    return re.sub(r"\s+", " ", conv["turns"][0]["text"].strip().lower())


def load_seen(paths) -> set:
    """Openers already present in any SFT file, so we never repeat the gold set."""
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(opener_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
    return seen


def run_model_http(base_url, api_key, model, prompt, timeout, temperature, max_tokens):
    """One completion from an OpenAI-compatible `/chat/completions` endpoint.

    That request shape is the de-facto standard — DeepSeek's own API, OpenRouter,
    Groq, Together, Google's OpenAI-compatibility layer and a local Ollama or
    llama.cpp server all speak it — so `--base-url` plus `--model` is enough to
    move this script between providers without touching the code.

    Deliberately `urllib` rather than `requests`: this script has no third-party
    dependencies and is not worth adding one for a single POST.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        print(f"  HTTP {e.code} from {url}: {detail}", file=sys.stderr)
        # 429 is a rate limit and 5xx is the provider having a moment; both are
        # worth waiting out rather than burning the remaining call budget.
        if e.code == 429 or e.code >= 500:
            time.sleep(RATE_LIMIT_BACKOFF_SECS)
        return ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  request to {url} failed: {e}", file=sys.stderr)
        return ""

    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        print(f"  unexpected response shape: {str(body)[:300]}", file=sys.stderr)
        return ""


def run_model(model, prompt, timeout, agent):
    """Invoke `opencode run` and return its stdout, or '' on failure.

    `agent` matters more than it looks. `opencode run` defaults to the `build`
    agent, which has full tool access — and the exemplars embedded in the prompt
    are transcripts of a coding agent being given instructions ("Check background
    task ... and ralph/tasks.md", "Want me implement some set?"). `build` reads
    those as work to do and disappears into a tool loop: measured, the same prompt
    that returns in 22s with a harmless exemplar times out past 300s with the real
    ones, while a 10KB block of filler text returns in 3.5s. It is not prompt size
    and not the model. `summary` is tool-free and answers the prompt as text.
    """
    try:
        proc = subprocess.run(
            ["opencode", "run", "--agent", agent, "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Deliberately not printing the exception: it stringifies the whole
        # command, i.e. the entire multi-kilobyte prompt, once per failed call.
        print(f"  model call timed out after {timeout}s", file=sys.stderr)
        return ""
    except FileNotFoundError:
        sys.exit("the `opencode` CLI is not on PATH — see https://opencode.ai")
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description="Generate SFT data via a cheap LLM (opencode).")
    ap.add_argument("--count", type=int, default=200,
                    help="target number of valid conversations to add (default 200)")
    ap.add_argument("--batch", type=int, default=3,
                    help="conversations requested per model call (default 3; "
                         "big thinking blocks are long, keep batches small)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"opencode model id (default {DEFAULT_MODEL})")
    ap.add_argument("--backend", choices=("opencode", "http"), default="opencode",
                    help="how to reach the model: the opencode CLI (default), or "
                         "any OpenAI-compatible /chat/completions endpoint")
    ap.add_argument("--base-url", default=None,
                    help="http backend: API base, e.g. https://api.deepseek.com/v1 "
                         "or http://localhost:11434/v1 for a local Ollama")
    ap.add_argument("--api-key-env", default="SFT_API_KEY",
                    help="http backend: env var holding the API key "
                         "(default SFT_API_KEY; unset means no auth header)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="http backend: sampling temperature (default 1.0 — this is "
                         "a diversity task, not a correctness one)")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="http backend: response cap (default 8000; big thinking "
                         "blocks need room or replies arrive truncated)")
    ap.add_argument("--agent", default=DEFAULT_AGENT,
                    help=f"opencode agent (default {DEFAULT_AGENT}); must be a "
                         "tool-free one — the default `build` agent treats the "
                         "exemplars as instructions and hangs in a tool loop")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output .jsonl path (default {DEFAULT_OUT}); the "
                         "hand-curated gold file is never written to")
    ap.add_argument("--exemplars", default=DEFAULT_EXEMPLARS,
                    help=f"gold .jsonl to imitate (default {DEFAULT_EXEMPLARS})")
    ap.add_argument("--exemplar-budget", type=int, default=9000,
                    help="max chars of exemplars embedded per prompt (default 9000)")
    ap.add_argument("--think-frac", type=float, default=0.75,
                    help="fraction of replies asked to include a thinking block (default 0.75)")
    ap.add_argument("--min-think-words", type=int, default=120,
                    help="reject thinking blocks shorter than this (default 120)")
    ap.add_argument("--system-frac", type=float, default=0.15,
                    help="fraction of calls asked for one system-prompt line (default 0.15)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-call timeout in seconds (default 600)")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="safety cap on model calls; 0 = count/batch * 3")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for topic/shape/exemplar sampling (default: random)")
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.exemplars):
        sys.exit("refusing to write generated data into the gold exemplar file; "
                 "pass a different --out")

    api_key = None
    if args.backend == "http":
        if not args.base_url:
            sys.exit("--backend http needs --base-url (e.g. https://api.deepseek.com/v1)")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"note: ${args.api_key_env} is unset — sending no Authorization "
                  "header (fine for a local server, not for a hosted API)",
                  file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    think_pct = max(0, min(100, round(args.think_frac * 100)))
    max_calls = args.max_calls or max(1, (args.count // args.batch + 1) * 3)

    pool = load_exemplars(args.exemplars, args.exemplar_budget)
    if not pool:
        sys.exit(f"no usable exemplars in {args.exemplars} — the generator "
                 "relies on them to match the target style")
    print(f"{len(pool)} exemplar conversations available from {args.exemplars}")

    # Dedup against every SFT file, so generated openers never echo the gold set.
    seen = load_seen(glob.glob(os.path.join(os.path.dirname(args.out), "*.jsonl")))
    added = 0
    calls = 0
    dupes = 0
    thinking = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while added < args.count and calls < max_calls:
            calls += 1
            prompt = build_prompt(args.batch, think_pct, args.min_think_words,
                                  pick_exemplars(pool, rng, args.exemplar_budget),
                                  rng, args.system_frac)
            if args.backend == "http":
                raw = run_model_http(args.base_url, api_key, args.model, prompt,
                                     args.timeout, args.temperature, args.max_tokens)
            else:
                raw = run_model(args.model, prompt, args.timeout, args.agent)
            convs, reasons = parse_conversations(raw, args.min_think_words)
            if not convs:
                detail = ", ".join(f"{n}x {why}" for why, n in reasons.most_common()) \
                    or ("no output" if not raw.strip() else "no JSON lines found")
                print(f"  call {calls}: 0 valid  ({detail})", file=sys.stderr)
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
                if any(thinking_words(t["text"]) for t in conv["turns"]):
                    thinking += 1
            f.flush()
            dropped = ", ".join(f"{n}x {why}" for why, n in reasons.most_common())
            print(f"  call {calls}: +{fresh} valid  (total {added}/{args.count}, "
                  f"{thinking} with thinking, {dupes} dupes skipped)"
                  + (f"  [dropped: {dropped}]" if dropped else ""))

    print(f"Added {added} conversations to {args.out} in {calls} model call(s); "
          f"{thinking} carry a thinking block")
    if added < args.count:
        print("Stopped short of target — raise --max-calls, or the model is "
              "returning too little valid JSONL (try a different --model, or "
              "lower --min-think-words).", file=sys.stderr)


if __name__ == "__main__":
    main()
