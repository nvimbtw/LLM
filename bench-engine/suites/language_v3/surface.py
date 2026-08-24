import math
import re
from dataclasses import dataclass
MARKER = re.compile(r"<\|/?[a-zA-Z0-9:_]+\|>")
@dataclass
class SurfaceMetrics:
    real_word_rate: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    repeat_3: float
    self_bleu_2: float
    words: int
    mean_sample_words: float
    per_sample_distinct_3: float
    per_sample_repeat_3: float
    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k,v in vars(self).items()}
class Lexicon:
    def __init__(self, words: set[str]):
        self.words = words
    @classmethod
    def build(cls, text: str) -> "Lexicon":
        return cls(set(words_in(text)))
    def contains(self, word: str) -> bool:
        return word in self.words
    def __len__(self) -> int:
        return len(self.words)
def words_in(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalpha():
            current.append(ch.lower())
        elif ch in ("'", "\u2019") and current:
            current.append("'")
        elif current:
            words.append("".join(current).rstrip("'"))
            current = []
    if current:
        words.append("".join(current).rstrip("'"))
    return [w for w in words if w]
def strip_markers(text: str) -> str:
    return MARKER.sub(" ", text)
def distinct_n(words: list[str], n: int) -> float:
    if n == 0 or len(words) < n:
        return math.nan
    grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(grams))/len(grams)
def repeat_rate(words: list[str], n: int) -> float:
    if n == 0 or len(words) < n:
        return math.nan
    grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    counts: dict[tuple, int] = {}
    for g in grams:
        counts[g]=counts.get(g,0)+1
    return sum(1 for g in grams if counts[g]>1)/len(grams)
def self_bleu(samples_words: list[list[str]], n: int = 2) -> float:
    if len(samples_words) < 2:
        return math.nan
    scores = []
    for i, hyp in enumerate(samples_words):
        if len(hyp) < n:
            continue
        hyp_grams = [tuple(hyp[j:j+n]) for j in range(len(hyp)-n+1)]
        if not hyp_grams:
            continue
        refs = set()
        for k, ref in enumerate(samples_words):
            if k == i:
                continue
            for j in range(len(ref)-n+1):
                refs.add(tuple(ref[j:j+n]))
        if not refs:
            continue
        overlap = sum(1 for g in hyp_grams if g in refs)
        scores.append(overlap/len(hyp_grams))
    if not scores:
        return math.nan
    return sum(scores)/len(scores)
def analyze(samples: list[str], lexicon: Lexicon) -> SurfaceMetrics:
    per_sample = [words_in(strip_markers(s)) for s in samples]
    pooled = [w for sample in per_sample for w in sample]
    if not pooled:
        return SurfaceMetrics(math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0, math.nan, math.nan, math.nan)
    real = sum(1 for w in pooled if lexicon.contains(w))
    per_d3 = [distinct_n(w,3) for w in per_sample if len(w)>=3]
    per_r3 = [repeat_rate(w,3) for w in per_sample if len(w)>=3]
    return SurfaceMetrics(
        real_word_rate=real/len(pooled),
        distinct_1=distinct_n(pooled,1),
        distinct_2=distinct_n(pooled,2),
        distinct_3=distinct_n(pooled,3),
        repeat_3=repeat_rate(pooled,3),
        self_bleu_2=self_bleu(per_sample,2),
        words=len(pooled),
        mean_sample_words=len(pooled)/len(per_sample),
        per_sample_distinct_3=sum(per_d3)/len(per_d3) if per_d3 else math.nan,
        per_sample_repeat_3=sum(per_r3)/len(per_r3) if per_r3 else math.nan,
    )
