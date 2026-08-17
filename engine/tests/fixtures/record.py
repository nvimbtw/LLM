"""Record the dataflow trace.

Deliberately a script and not a pytest flag: re-recording is how a behaviour
change gets blessed, and it should be an act, not a side effect of a failing
run. Commit the resulting diff as the evidence of what moved.

    engine/.venv/bin/python -m tests.fixtures.record
"""

import json
from pathlib import Path

from tests.fixtures import tiny_env, trace_spec

TRACE = Path(__file__).resolve().parent / "trace.json"


def record() -> Path:
    with tiny_env.data_root():
        from llm import storage
        from llm.corpus import bpe
        from llm.model import checkpoint

        tokenizer = bpe.load(storage.model_dir(tiny_env.TINY_MODEL))
        model, _ = checkpoint.load(storage.weights_path(tiny_env.TINY_MODEL), "cpu")
        payload = trace_spec.checkpoints(model, tokenizer)

    TRACE.write_text(json.dumps(payload, indent=2) + "\n")
    return TRACE


if __name__ == "__main__":
    path = record()
    data = json.loads(path.read_text())
    print(f"wrote {path}")
    print(f"  prompt   {len(data['prompt_ids'])} tokens")
    print(f"  greedy   {data['greedy']['text']!r}")
    print(f"  sampled  {data['sampled']['text']!r}")
