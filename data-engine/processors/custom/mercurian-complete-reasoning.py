import re
from processors.helpers import Ctx, Param, processor


def reasoning_parts(text: str) -> tuple[str, str] | None:
    """Parse one complete source trace; reject missing, repeated or unclosed tags."""
    pattern = r"\s*<think>(.*?)</think>\s*(.+?)\s*"
    if "<|begin_of_thought|>" in text:
        pattern = r"\s*<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>\s*"
    match = re.fullmatch(pattern, text, re.DOTALL)
    if match is None:
        return None
    thought, answer = (part.strip() for part in match.groups())
    if not thought or not answer or re.search(r"</?think>|<\|", thought + answer):
        return None
    return thought, answer


@processor(name="mercurian-complete-reasoning", title="Mercurian complete reasoning traces", kinds=["sft"], mode="turns", description="Normalize OpenThoughts and think-tag traces without shortening reasoning or answers. Reject malformed, tool-bearing and oversized exchanges; follow with convert-chat and markdown-emphasis.", params=[Param(name="max_characters", label="Maximum source characters", type="int", default=6500, minimum=100, maximum=50000)])
def run(turns: list[dict], params: dict, ctx: Ctx) -> list[dict] | None:
    """Keep complete user/assistant reasoning exchanges within a character prefilter.

    This is not a token limit. Verify the converted output with the base tokenizer
    before selecting it for training; no trace is truncated or synthesized.
    """
    if len(turns) != 2 or [turn.get("role") for turn in turns] != ["user", "assistant"]:
        return None
    question, reply = (turn.get("content") for turn in turns)
    if not isinstance(question, str) or not isinstance(reply, str) or not question.strip():
        return None
    if len(question) + len(reply) > params["max_characters"]:
        return None
    if re.search(r"<tool_call>|<tool_response>|<\|tool-", question + reply):
        return None
    parts = reasoning_parts(reply)
    if parts is None:
        return None
    thought, answer = parts
    return [{"role": "user", "content": question.strip()}, {"role": "assistant", "content": "<think>" + thought + "</think>\n\n" + answer}]
