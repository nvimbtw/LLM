"""The benchmark pipeline.

    1. load the model + its own tokenizer
    2. score held-out compression against an n-gram floor
    3. generate from fixed prompts and measure surface properties
    4. combine into an auditable composite

Held-out text lives in `data/eval/corpus/` and must be text the model was not
trained on — kept out of `data/input/` on purpose, since everything under input/
is selectable as training data and a held-out set that can be accidentally
trained on is not a held-out set.
"""

import json
import time
from threading import Event

from llm import storage
from llm.config import EvalConfig
from llm.eval import generate
from llm.eval.runner import run_assistant, run_language, run_markup
from llm.train.loop import pick_device


def run(cfg: EvalConfig, on_progress=None, cancel: Event | None = None) -> dict:
    started = time.time()

    weights = storage.weights_path(cfg.model_name)
    if not weights.is_file():
        raise FileNotFoundError(f"No model {cfg.model_name!r} at {weights}")

    sampling = generate.SamplingConfig(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )

    device = pick_device()
    if cfg.suite == "markup":
        report = run_markup(cfg.model_name, device, cancel, on_progress)
    elif cfg.suite == "assistant":
        report = run_assistant(cfg.model_name, device, cancel, on_progress)
    elif cfg.suite == "language":
        report = run_language(cfg.model_name, sampling=sampling, device=device,
                              cancel=cancel, on_progress=on_progress)
    else:
        raise ValueError(f"unknown suite {cfg.suite!r} (language | markup | assistant)")

    result = report.as_dict()
    result["suite"] = cfg.suite
    result["total_secs"] = round(time.time() - started, 1)

    # Written beside the weights so a score travels with the model it describes.
    path = storage.model_dir(cfg.model_name) / f"benchmark-{cfg.suite}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
