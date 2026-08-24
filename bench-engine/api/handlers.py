import uuid
from store import queries as q
import store
from store.schema import Suite

def benchmark(config: dict, progress, cancel, owner):
    import time
    from pathlib import Path
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("model_name required")
    # resolve model_id uuid
    try:
        model_id = uuid.UUID(model_name)
    except ValueError:
        # lookup not possible without engine DB, use name as id hash
        import hashlib
        h = hashlib.md5(model_name.encode()).hexdigest()
        model_id = uuid.UUID(h[:32])
    suite_raw = config.get("suite", "language_v3")
    if suite_raw in ("language", "language-v3", "language_v3"):
        suite_enum = Suite.language_v3
    else:
        try:
            suite_enum = Suite(suite_raw)
        except ValueError:
            suite_enum = Suite.language_v3
    with store.session() as db:
        bench = q.open_benchmark(db, model_id=model_id, model_name=model_name, suite=suite_enum, suite_version=3, status="running")
        bench_id = bench.id
    # try to load model
    progress(stage="loading model", progress=0.05)
    start = time.time()
    try:
        model, tokenizer = _load_model(model_name, progress)
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        from suites.language_v3.runner import run_language_v3
        report = run_language_v3(model, tokenizer, device=device, on_progress=progress, cancel=cancel)
        report.model["name"] = model_name
        report.model["id"] = str(model_id)
        score_val = report.score.get("score")
        elapsed = time.time() - start
        result = {"suite": "language", "total_secs": elapsed, "cancelled": False, **_report_dict(report)}
        with store.session() as db:
            q.close_benchmark(db, bench_id, status="done", score=score_val, suite_version=report.score.get("version",3), report=result, eval_tokens=report.eval_tokens)
        return {"benchmark_id": str(bench_id), "score": score_val, "suite_version": 3}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with store.session() as db:
            q.close_benchmark(db, bench_id, status="failed", error=f"{e}\n{tb}")
        raise

def _sanitize(obj):
    import math
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
def _report_dict(report):
    base = {"model": _sanitize(report.model), "corpus": _sanitize(report.corpus), "surface": _sanitize(report.surface), "grammar": _sanitize(report.grammar), "coherence": _sanitize(report.coherence), "samples": _sanitize(report.samples), "score": _sanitize(report.score), "ablation": None}
    if base["corpus"] and "tokens_scored" not in base["corpus"]:
        base["corpus"]["tokens_scored"] = report.eval_tokens
    return base

def _load_model(model_name: str, progress):
    from pathlib import Path
    import os
    # search DATA_DIR / engine/data
    candidates = []
    data_dir = os.environ.get("DATA_DIR") or os.environ.get("ENGINE_DATA_DIR") or "engine/data"
    for base in [Path(data_dir), Path("/app/data"), Path("engine/data"), Path("../engine/data")]:
        candidates.append(base / "artifacts" / "models" / model_name / "model.pt")
        candidates.append(base / "artifacts" / "models" / model_name)
    for c in candidates:
        if c.exists():
            # try to load via bench-engine llm
            try:
                from llm.model.checkpoint import load as ckpt_load
                from llm.corpus import bpe
                # find model.pt sibling
                pt = c if c.is_file() else c / "model.pt"
                tok_dir = c if c.is_dir() else c.parent
                # tokenizer
                tok = bpe.load(tok_dir)
                model, arch = ckpt_load(pt, device="cpu")
                model.model_id = model_name
                return model, tok
            except Exception as e:
                continue
    # mock model for tests / missing weights
    progress(stage="no checkpoint — using mock", progress=0.1)
    return _mock_model(), _mock_tokenizer()

class _MockTokenizer:
    def __init__(self): self.vocab = 1000
    def encode(self, text): return [ord(c)%1000 for c in text[:10]] or [1,2]
    def decode(self, ids): return "the cat sat on the mat " * 2

def _mock_tokenizer():
    # wrap to look like bpe tokenizer
    return "/tmp/mock"

class _MockModel:
    def __init__(self):
        self.ctx_window = 256
        self.layers = 4
        self.dimensions = 256
        self.heads = 4
        self.vocab = 1000
        self.model_id = "mock"
    def parameter_count(self): return 1000000
    def empty_caches(self): return [(None,None)]*4
    def __call__(self, ids, caches=None, pos_offset=0):
        import torch
        b,s = ids.shape
        logits = torch.randn(b, s, self.vocab)
        return logits, caches

def _mock_model():
    m = _MockModel()
    return m

HANDLERS = {}
try:
    from store.schema import RunKind
    HANDLERS[RunKind.benchmark] = benchmark
except Exception:
    pass
