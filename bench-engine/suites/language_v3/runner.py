import math
import time
from dataclasses import dataclass, field
from pathlib import Path
import torch
from suites.language_v3 import prompts, generate, surface, grammar, score
from suites.language_v3 import corpus as corpus_stage
from suites.language_v3 import ngram

@dataclass
class ModelInfo:
    id: str
    name: str | None
    layers: int
    dimensions: int
    heads: int
    context: int
    vocab: int
    parameters: int
    @classmethod
    def of(cls, model, model_id: str) -> "ModelInfo":
        return cls(id=model_id, name=None, layers=getattr(model,"layers",0), dimensions=getattr(model,"dimensions",0), heads=getattr(model,"heads",0), context=getattr(model,"ctx_window",0), vocab=getattr(model,"vocab",0), parameters=getattr(model, "parameter_count", lambda:0)())

@dataclass
class LanguageV3Report:
    model: dict
    corpus: dict | None
    surface: dict
    grammar: dict
    coherence: dict
    samples: list[dict]
    score: dict
    eval_tokens: int = 0

def _find_eval_dir() -> Path:
    # bench-engine/eval_data/language, fallback to engine/data/eval/corpus
    candidates = [
        Path(__file__).parent.parent.parent / "eval_data" / "language" / "corpus",
        Path("/app/eval_data/language/corpus"),
        Path("/app/engine/data/eval/corpus"),
        Path(__file__).parent.parent.parent.parent / "engine" / "data" / "eval" / "corpus",
    ]
    for c in candidates:
        if c.exists():
            return c.parent
    return candidates[0]

def _lexicon_text(eval_dir: Path) -> str:
    corpus_dir = eval_dir / "corpus"
    if corpus_dir.exists():
        texts = []
        for p in sorted(corpus_dir.glob("*.txt")):
            try:
                texts.append(p.read_text(errors="ignore"))
            except Exception:
                pass
        if texts:
            return "\n".join(texts)
    return "the quick brown fox jumps over the lazy dog " * 100

def _sanitize(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
def _corpus_bits(model, tokenizer, eval_dir: Path, device: str) -> tuple[dict, int]:
    try:
        from suites.language_v3.corpus import load, evaluate
        split = load(eval_dir / "corpus")
        report = evaluate(model, tokenizer, split, device)
        d = report.as_dict() if hasattr(report, 'as_dict') else dict(report)
        return _sanitize(d), report.tokens_scored if hasattr(report,'tokens_scored') else 0
    except Exception as e:
        return {"error": str(e), "bits_per_byte": None, "baseline_bits_per_byte": None}, 0

def _gen_ppl(model, tokenizer, samples: list[str], device: str) -> float:
    try:
        from llm.corpus import bpe
        import math, torch
        joined = "\n".join(samples)
        ids = bpe.encode(tokenizer, joined)
        if len(ids) < 10:
            return math.nan
        from suites.language_v3.score import language_component
        # reuse token_logprobs logic
        from suites.language_v3 import corpus as cs
        window = model.ctx_window
        stride = max(1, window//2)
        logprobs = []
        scored_upto = 1
        start = 0
        while scored_upto < len(ids):
            chunk = ids[start:start+window]
            if len(chunk) < 2:
                break
            logits,_ = model(torch.tensor([chunk], device=device))
            lps = torch.log_softmax(logits[0].float(), dim=-1)
            for i in range(len(chunk)-1):
                idx = start + i + 1
                if idx < scored_upto:
                    continue
                logprobs.append(float(lps[i, chunk[i+1]]))
                scored_upto = idx+1
            if start + window >= len(ids):
                break
            start += stride
        if not logprobs:
            return math.nan
        bits_per_token = -sum(logprobs)/len(logprobs)/math.log(2)
        # convert to per-byte approx
        b = len(joined.encode())
        bits_per_byte = -sum(logprobs)/math.log(2)/ max(b,1)
        return bits_per_byte
    except Exception:
        return math.nan

def run_language_v3(model, tokenizer, device: str = "cpu", on_progress=None, cancel=None) -> LanguageV3Report:
    eval_dir = _find_eval_dir()
    corpus_report, eval_tokens = _corpus_bits(model, tokenizer, eval_dir, device)
    bpb = corpus_report.get("bits_per_byte", math.nan) if isinstance(corpus_report, dict) else math.nan
    baseline = corpus_report.get("baseline_bits_per_byte", math.nan) if isinstance(corpus_report, dict) else math.nan
    if on_progress:
        on_progress(stage="generating", progress=0.2)
    cfg = generate.SamplingConfig(max_tokens=128, temperature=0.8, top_k=40, top_p=0.95, seed=1234)
    all_prompts = prompts.ALL_PROMPTS
    samps = generate.samples(model, tokenizer, cfg, all_prompts, device=device, on_progress=on_progress, cancel=cancel)
    texts = [s.text for s in samps]
    # lexicon from held-out corpus text
    lex_text = _lexicon_text(eval_dir)
    lex = surface.Lexicon.build(lex_text)
    surf = surface.analyze(texts, lex)
    gram = grammar.evaluate(model, tokenizer, device=device)
    gen_ppl = _gen_ppl(model, tokenizer, texts, device)
    if on_progress:
        on_progress(stage="scoring", progress=0.9)
    bpb_val = bpb if isinstance(bpb, (int, float)) and math.isfinite(bpb) else math.nan
    base_val = baseline if isinstance(baseline, (int, float)) and math.isfinite(baseline) else math.nan
    comp = score.composite(bits_per_byte=bpb_val, grammar_accuracy=gram.accuracy, grammar_pairs=gram.pairs, real_word_rate=surf.real_word_rate, generated_words=surf.words, distinct_3=surf.distinct_3, gen_ppl=gen_ppl, gen_baseline=base_val)
    model_info = ModelInfo.of(model, getattr(model, 'model_id', 'unknown'))
    return LanguageV3Report(
        model=vars(model_info),
        corpus=_sanitize(corpus_report),
        surface=surf.as_dict(),
        grammar=_sanitize({"accuracy": gram.accuracy, "pairs": gram.pairs, "correct": gram.correct}),
        coherence=_sanitize({"gen_bits_per_byte": gen_ppl, "baseline_bits_per_byte": base_val}),
        samples=[{"prompt": s.prompt, "text": s.text, "tokens": s.tokens} for s in samps],
        score=comp.as_dict(),
        eval_tokens=eval_tokens + surf.words,
    )
