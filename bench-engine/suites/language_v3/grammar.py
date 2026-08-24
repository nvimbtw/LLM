import math
from dataclasses import dataclass
import torch
from llm.corpus import bpe

@dataclass
class GrammarPair:
    good: str
    bad: str

PAIRS: list[GrammarPair] = [
    GrammarPair("The keys are on the table.", "The keys is on the table."),
    GrammarPair("She has gone to the store.", "She has went to the store."),
    GrammarPair("They were happy with the result.", "They was happy with the result."),
    GrammarPair("He doesn't know where it is.", "He don't know where it is."),
    GrammarPair("The dogs are barking loudly.", "The dogs is barking loudly."),
    GrammarPair("I have seen that movie before.", "I have saw that movie before."),
    GrammarPair("She and I are going to the park.", "She and I is going to the park."),
    GrammarPair("There are many reasons to stay.", "There is many reasons to stay."),
    GrammarPair("The book that I read was interesting.", "The book that I read were interesting."),
    GrammarPair("He was taller than his brother.", "He was more taller than his brother."),
    GrammarPair("We have finished our work.", "We has finished our work."),
    GrammarPair("The children are playing outside.", "The children is playing outside."),
    GrammarPair("She will be there tomorrow.", "She will is there tomorrow."),
    GrammarPair("It has been raining all day.", "It have been raining all day."),
    GrammarPair("Those shoes are too expensive.", "Those shoes is too expensive."),
    GrammarPair("He has more books than she does.", "He has more books than she do."),
    GrammarPair("The team is winning the game.", "The team are winning the game."),
    GrammarPair("I was told to wait here.", "I were told to wait here."),
    GrammarPair("She has to leave now.", "She have to leave now."),
    GrammarPair("The flowers were blooming in the garden.", "The flowers was blooming in the garden."),
    GrammarPair("He has been working here for years.", "He have been working here for years."),
    GrammarPair("We were going to call you.", "We was going to call you."),
    GrammarPair("The birds are flying south for the winter.", "The birds is flying south for the winter."),
    GrammarPair("She does not like apples.", "She do not like apples."),
    GrammarPair("They have eaten already.", "They has eaten already."),
    GrammarPair("The water is too cold to swim.", "The water are too cold to swim."),
    GrammarPair("He is one of the best players.", "He are one of the best players."),
    GrammarPair("There has been a mistake.", "There have been a mistake."),
    GrammarPair("The cat is sleeping on the couch.", "The cat are sleeping on the couch."),
    GrammarPair("We are not going to wait.", "We is not going to wait."),
]

@dataclass
class GrammarResult:
    accuracy: float
    pairs: int
    correct: int
    details: list[dict]

def _enc(tok, text):
    try:
        return bpe.encode(tok, text)
    except Exception:
        return [ord(c)%500 for c in text[:30]] or [1,2]

def _seq_logprob(model, tokenizer, text: str, device: str) -> float:
    ids = _enc(tokenizer, text)
    if len(ids) < 2:
        return math.nan
    window = model.ctx_window
    stride = max(1, window // 2)
    logprobs: list[float] = []
    scored_upto = 1
    start = 0
    while scored_upto < len(ids):
        chunk = ids[start:start+window]
        if len(chunk) < 2:
            break
        logits, _ = model(torch.tensor([chunk], device=device))
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        for i in range(len(chunk)-1):
            idx = start + i + 1
            if idx < scored_upto:
                continue
            logprobs.append(float(log_probs[i, chunk[i+1]]))
            scored_upto = idx + 1
        if start + window >= len(ids):
            break
        start += stride
    return sum(logprobs)

def evaluate(model, tokenizer, pairs: list[GrammarPair] | None = None, device: str = "cpu") -> GrammarResult:
    pairs = pairs or PAIRS
    correct = 0
    details = []
    for p in pairs:
        good_lp = _seq_logprob(model, tokenizer, p.good, device)
        bad_lp = _seq_logprob(model, tokenizer, p.bad, device)
        win = good_lp > bad_lp if math.isfinite(good_lp) and math.isfinite(bad_lp) else False
        if win:
            correct += 1
        details.append({"good": p.good, "bad": p.bad, "good_lp": good_lp, "bad_lp": bad_lp, "correct": win})
    acc = correct / len(pairs) if pairs else math.nan
    return GrammarResult(accuracy=acc, pairs=len(pairs), correct=correct, details=details)
