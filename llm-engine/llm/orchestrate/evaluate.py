"""CLI benchmark entrypoint: enqueue through Rust's shared execution queue."""
from llm.backend import Hop


def run(cfg, on_progress=None, cancel=None):
    with Hop(timeout=3600) as backend:
        result, _ = backend.request("POST", "/internal/bench/queue", json=cfg.model_dump())
    return result
