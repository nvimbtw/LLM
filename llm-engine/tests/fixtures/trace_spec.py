"""The question, and the five checkpoints it is observed at.

One definition, read by both the recorder and the assertions, so the bar and
the measurement can never drift apart.
"""

import torch

from llm.chat.markup import OpenBlocks, Rules
from llm.chat.template import Role, Turn, build_prompt
from llm.corpus import bpe
from llm.model import sample
from llm.orchestrate import chat as chat_pipeline

QUESTION = "the cat sat on the"
SYSTEM = "you are terse."
SEED = 7
MAX_TOKENS = 24
TOP_LOGITS = 5


def checkpoints(model, tokenizer) -> dict:
    question_ids = bpe.encode(tokenizer, QUESTION)
    prompt_ids = build_prompt(
        SYSTEM,
        [Turn(role=Role.USER, text=QUESTION)],
        tokenizer,
        model.ctx_window,
        min(chat_pipeline.RESERVE_FOR_REPLY, max(1, model.ctx_window // 2)),
        None,
    )

    logits, _ = model(
        torch.tensor([prompt_ids]), caches=model.empty_caches(), pos_offset=0
    )
    step = logits[0, -1]
    top = torch.topk(step, TOP_LOGITS)

    return {
        "question_ids": question_ids,
        "prompt_ids": prompt_ids,
        "logits": {
            "shape": list(logits.shape),
            "argmax": int(step.argmax()),
            "top": [
                [int(i), round(float(v), 4)]
                for i, v in zip(top.indices.tolist(), top.values.tolist())
            ],
        },
        "greedy": generate(model, tokenizer, prompt_ids, temperature=0.0, seed=None),
        "sampled": generate(model, tokenizer, prompt_ids, temperature=0.8, seed=SEED),
    }


def generate(model, tokenizer, prompt_ids, temperature, seed) -> dict:
    """The decode loop of `orchestrate.chat.stream`, reduced to observable state.

    Reproduced rather than driven through `stream` so the ids are visible per
    step; `test_stream_matches_trace` pins the pipeline itself against this.
    """
    rules = Rules.build(tokenizer)
    blocks = OpenBlocks(rules)
    generator = torch.Generator().manual_seed(seed) if seed else None

    logits, caches = model(
        torch.tensor([prompt_ids]), caches=model.empty_caches(), pos_offset=0
    )
    produced: list[int] = []
    position = len(prompt_ids)

    for _ in range(MAX_TOKENS):
        if position >= model.ctx_window - blocks.depth:
            break
        step = logits[0, -1].clone()
        blocks.mask(step)
        next_id = sample.pick(step, temperature, 40, 0.95, generator)
        if next_id in (rules.end_of_assistant, rules.end_of_text):
            break
        blocks.observe(next_id)
        produced.append(next_id)
        logits, caches = model(
            torch.tensor([[next_id]]), caches=caches, pos_offset=position
        )
        position += 1

    return {"ids": produced, "text": bpe.decode(tokenizer, produced)}
