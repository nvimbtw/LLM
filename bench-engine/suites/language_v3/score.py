import math
from dataclasses import dataclass
SCORE_VERSION = 3
BPB_MID = 1.60
BPB_SCALE = 0.45
NONSENSE_HALVING = 0.10
VARIETY_FAIL = 0.30
VARIETY_PASS = 0.70
W_LANGUAGE = 0.35
W_GRAMMAR = 0.25
W_WORDS = 0.15
W_DIVERSITY = 0.15
W_COHERENCE = 0.10
def language_component(bits_per_byte: float) -> float:
    return 1.0 / (1.0 + math.exp((bits_per_byte - BPB_MID) / BPB_SCALE))
def words_component(real_word_rate: float) -> float:
    return 2.0 ** (-(1.0 - real_word_rate) / NONSENSE_HALVING)
def variety_gate(distinct_3: float) -> float:
    if not math.isfinite(distinct_3):
        return 1.0
    return _clamp((distinct_3 - VARIETY_FAIL) / (VARIETY_PASS - VARIETY_FAIL), 0.0, 1.0)
@dataclass
class ScoreBreakdown:
    version: int
    language: float
    grammar: float
    words: float
    diversity: float
    coherence: float
    variety_gate: float
    score: float
    score_stderr: float
    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k,v in vars(self).items()}
def diversity_component(distinct_3: float) -> float:
    if not math.isfinite(distinct_3):
        return math.nan
    return _clamp((distinct_3 - 0.30) / 0.40, 0.0, 1.0)
def coherence_component(gen_ppl: float, baseline: float) -> float:
    if not math.isfinite(gen_ppl) or not math.isfinite(baseline) or baseline <= 0:
        return math.nan
    # lower gen_ppl nearer baseline is better; very high gen_ppl = incoherent
    # map ratio baseline/gen_ppl clamped
    ratio = baseline / max(gen_ppl, 1e-9)
    return _clamp(ratio, 0.0, 1.0)
def composite(bits_per_byte: float, grammar_accuracy: float, grammar_pairs: int, real_word_rate: float, generated_words: int, distinct_3: float, gen_ppl: float = math.nan, gen_baseline: float = math.nan) -> ScoreBreakdown:
    language = language_component(bits_per_byte) if math.isfinite(bits_per_byte) else math.nan
    grammar = _clamp((grammar_accuracy - 0.5)/0.5, 0.0, 1.0) if math.isfinite(grammar_accuracy) else math.nan
    words = words_component(real_word_rate) if math.isfinite(real_word_rate) else math.nan
    diversity = diversity_component(distinct_3)
    coherence = coherence_component(gen_ppl, gen_baseline) if math.isfinite(gen_ppl) else math.nan
    gate = variety_gate(distinct_3)
    components = [(W_LANGUAGE, language), (W_GRAMMAR, grammar), (W_WORDS, words), (W_DIVERSITY, diversity), (W_COHERENCE, coherence)]
    present = sum(w for w,v in components if math.isfinite(v))
    if present <= 0:
        score = math.nan
    else:
        blend = sum(w*v for w,v in components if math.isfinite(v)) / present
        score = 100.0 * blend * gate
    return ScoreBreakdown(version=SCORE_VERSION, language=language, grammar=grammar, words=words, diversity=diversity, coherence=coherence, variety_gate=gate, score=score, score_stderr=_stderr(grammar_accuracy, grammar_pairs, words, real_word_rate, generated_words, present, gate, diversity, distinct_3))
def _stderr(accuracy, pairs, words_value, real_word_rate, word_count, present, gate, diversity, distinct_3) -> float:
    if present <= 0:
        return math.nan
    variance = 0.0
    if math.isfinite(accuracy) and pairs > 0:
        se = math.sqrt(max(accuracy*(1-accuracy),0.0)/pairs)
        variance += (W_GRAMMAR/present*2.0*se)**2
    if math.isfinite(real_word_rate) and word_count>0 and math.isfinite(words_value):
        se = math.sqrt(max(real_word_rate*(1-real_word_rate),0.0)/word_count)
        slope = words_value * math.log(2) / NONSENSE_HALVING
        variance += (W_WORDS/present*slope*se)**2
    if math.isfinite(distinct_3) and word_count>10:
        # rough binomial for diversity: treat distinct_3 as rate over ~word_count windows
        se = math.sqrt(max(distinct_3*(1-distinct_3),0.0)/ max(word_count,1))
        variance += (W_DIVERSITY/present* (1/0.40) * se)**2
    return 100.0 * gate * math.sqrt(variance)
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
