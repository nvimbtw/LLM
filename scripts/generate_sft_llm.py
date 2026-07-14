#!/usr/bin/env python3
"""
Generate synthetic SFT chat data with a cheap LLM, via the `opencode` CLI.

This is the model-backed companion to `generate_sft.py` (which uses fixed
templates). It asks a cheap/free model for conversations, validates every line
against the backend's SFT schema, and appends only the valid ones to
`backend/data/raw/sft/synthetic.jsonl`. Bad JSON or malformed conversations are
dropped, so a flaky small model can't corrupt the training set.

Model output must be JSONL — one conversation per line — matching the schema the
Rust side deserializes (see `SftConversation` / `SftTurn` in
backend/src/training.rs):

    {"system"?: string, "turns": [{"role": "user"|"assistant", "text": string}, ...]}

Rules enforced (a line failing any of these is discarded):
  - roles are exactly "user" or "assistant",
  - turns alternate, starting with "user",
  - the conversation ends on an "assistant" turn,
  - every "text" is a non-empty string.

A configurable fraction of assistant replies are asked to wrap their reasoning
in `<|thinking|>...<|/thinking|>` before the answer — this is what teaches the
model to *think inside a chat reply* during SFT (the chat template keeps the
block verbatim as the training target; see backend/src/llm/src/chat_template.rs).

No API key needed with the default free model. Requires the `opencode` CLI on
PATH (https://opencode.ai). Pick any model with `--model provider/name`.

Usage:
    python3 generate_sft_llm.py --count 200
    python3 generate_sft_llm.py --count 500 --batch 20 --think-frac 0.4
    python3 generate_sft_llm.py --model opencode/deepseek-v4-flash-free --count 100
"""

import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_OUT = "/home/x_x/Programming/LLM/backend/data/raw/sft/synthetic.jsonl"
DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"

# Strips ANSI escape codes (color, cursor moves) from opencode's TUI output so
# only plain text remains for JSON parsing.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

PROMPT_TEMPLATE = """\
You are generating training data for a small chat model. Output ONLY JSONL:
exactly {batch} lines, each a single JSON object on its own line, no prose, no
markdown fences, no commentary before or after.

Each object is one short chat conversation with this exact shape:
  {{"turns": [{{"role": "user", "text": "..."}}, {{"role": "assistant", "text": "..."}}]}}
Optionally include a "system" string key: {{"system": "...", "turns": [...]}}.

Hard rules (a line breaking any of these is useless):
- roles are only "user" or "assistant",
- turns alternate and start with "user",
- the conversation ENDS on an "assistant" turn,
- keep it short: 1 to 3 exchanges,
- vary the topics: simple math, definitions, short factual Q&A, everyday
  requests, formatting a short list. Keep answers correct and concise.

For about {think_pct}% of the lines, the FINAL assistant reply must begin with a
thinking block, then the answer, like:
  "<|thinking|>Let me work this out step by step. ...<|/thinking|>The answer is ..."
Put the literal markers <|thinking|> and <|/thinking|> in the text exactly as
shown. The rest of the lines have plain assistant replies with no thinking block.

Output the {batch} JSONL lines now:"""


def valid_conversation(obj):
    """True if obj matches the SFT schema and the enforced turn rules."""
    if not isinstance(obj, dict) or "turns" not in obj:
        return False
    if "system" in obj and not isinstance(obj["system"], str):
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
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
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
    ap.add_argument("--batch", type=int, default=20,
                    help="conversations requested per model call (default 20)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"opencode model id (default {DEFAULT_MODEL})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output .jsonl path (default {DEFAULT_OUT})")
    ap.add_argument("--think-frac", type=float, default=0.35,
                    help="fraction of replies asked to include a thinking block (default 0.35)")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-call timeout in seconds (default 180)")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="safety cap on model calls; 0 = count/batch * 3")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    think_pct = max(0, min(100, round(args.think_frac * 100)))
    max_calls = args.max_calls or max(1, (args.count // args.batch + 1) * 3)

    added = 0
    calls = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while added < args.count and calls < max_calls:
            calls += 1
            prompt = PROMPT_TEMPLATE.format(batch=args.batch, think_pct=think_pct)
            convs = parse_conversations(run_model(args.model, prompt, args.timeout))
            if not convs:
                print(f"  call {calls}: 0 valid lines (model gave nothing usable)",
                      file=sys.stderr)
                continue
            for conv in convs:
                if added >= args.count:
                    break
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                added += 1
            f.flush()
            print(f"  call {calls}: +{len(convs)} valid  (total {added}/{args.count})")

    print(f"Added {added} conversations to {args.out} in {calls} model call(s)")
    if added < args.count:
        print("Stopped short of target — raise --max-calls, or the model is "
              "returning too little valid JSONL (try a different --model).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
