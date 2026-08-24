import math
import glob
from pathlib import Path
from dataclasses import dataclass
import torch

@dataclass
class CorpusReport:
    bits_per_byte: float
    baseline_bits_per_byte: float
    tokens_scored: int
    def as_dict(self):
        return {"bits_per_byte": self.bits_per_byte, "baseline_bits_per_byte": self.baseline_bits_per_byte, "tokens_scored": self.tokens_scored}

def load(directory: Path):
    texts = []
    for p in sorted(Path(directory).glob("*.txt")):
        texts.append(p.read_text(errors="ignore"))
    joined = "\n".join(texts)
    mid = len(joined)//10
    train = joined[:-mid] if mid else joined
    heldout = joined[-mid:] if mid else joined
    return (train, heldout)

def evaluate(model, tokenizer, split, device="cpu"):
    train, heldout = split
    # stub bits per byte via simple forward
    try:
        from llm.corpus import bpe
        ids = bpe.encode(tokenizer, heldout[:4096])
        if len(ids) < 10:
            return CorpusReport(math.nan, math.nan, 0)
        import torch
        from suites.language_v3.ngram import ByteNgram
        # baseline
        baseline = ByteNgram(order=5)
        baseline.fit(train.encode()[:4096] if train else heldout.encode()[:4096])
        base_bpb = baseline.bits_per_byte(heldout.encode()[:1024])
        # model bpb via token_logprobs
        window = model.ctx_window
        stride = max(1, window//2)
        logprobs=[]
        scored_upto=1
        start=0
        while scored_upto < len(ids):
            chunk=ids[start:start+window]
            if len(chunk)<2: break
            logits,_=model(torch.tensor([chunk], device=device))
            lps=torch.log_softmax(logits[0].float(), dim=-1)
            for i in range(len(chunk)-1):
                idx=start+i+1
                if idx < scored_upto: continue
                logprobs.append(float(lps[i, chunk[i+1]]))
                scored_upto=idx+1
            if start+window >= len(ids): break
            start+=stride
        bits = -sum(logprobs)/math.log(2) if logprobs else math.nan
        bpb = bits / max(len(heldout.encode()[:4096]),1)
        return CorpusReport(bits_per_byte=bpb, baseline_bits_per_byte=base_bpb, tokens_scored=len(logprobs))
    except Exception as e:
        return CorpusReport(math.nan, math.nan, 0)
