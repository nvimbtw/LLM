"""The language benchmark: the stages, run in order, into one report."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import torch

from llm import storage
from llm.corpus import bpe
from llm.eval import ablation
from llm.eval import chat as chat_stage
from llm.eval import corpus as corpus_stage
from llm.eval import generate, markup as markup_stage, score, surface
from llm.chat.markup import Rules
from llm.model import checkpoint


@dataclass
class LanguageReport:
    model_name: str
    parameters: int
    vocab: int
    corpus: dict | None = None
    surface: dict | None = None
    # Per-sublayer contribution, from silencing each in turn. Cheap enough to run
    # on every eval (two forwards per block over one window) and it answers a
    # question the composite score cannot: not how good the model is, but which
    # part of it is responsible.
    ablation: dict | None = None
    score: dict = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    cancelled: bool = False

    def as_dict(self) -> dict:
        return dict(vars(self))


def run_language(
    model_name: str,
    eval_dir: Path | None = None,
    sampling: generate.SamplingConfig | None = None,
    device: str = "cpu",
    cancel: Event | None = None,
    on_progress=None,
) -> LanguageReport:
    """Score a model: held-out compression, then generation surface properties.

    Grammar is reported as unavailable for now — the minimal-pair suite is not
    ported yet — and its weight is redistributed across the stages that did run,
    rather than the composite silently treating it as zero.
    """
    sampling = sampling or generate.SamplingConfig()
    eval_dir = eval_dir or (storage.eval_dir() / "corpus")

    model, _ = checkpoint.load(storage.weights_path(model_name), device)
    tokenizer = bpe.load(storage.model_dir(model_name))

    report = LanguageReport(
        model_name=model_name,
        parameters=model.parameter_count(),
        vocab=model.vocab,
    )

    def note(stage: str, detail: str = "") -> None:
        if on_progress:
            on_progress({"stage": stage, "detail": detail})

    note("corpus", "scoring held-out text")
    split = corpus_stage.load(eval_dir)
    corpus_report = corpus_stage.evaluate(model, tokenizer, split, device)
    report.corpus = corpus_report.as_dict() if corpus_report else None

    if cancel is not None and cancel.is_set():
        report.cancelled = True
        return report

    note("generation", f"{len(generate.DEFAULT_PROMPTS)} prompts")
    drawn = generate.samples(model, tokenizer, sampling, device=device)
    report.samples = [{"prompt": s.prompt, "text": s.text} for s in drawn]

    lexicon = surface.Lexicon.build(split.train + split.heldout)
    metrics = surface.analyze([s.text for s in drawn], lexicon)
    report.surface = metrics.as_dict()

    note("ablation", "per-sublayer contribution")
    heldout_ids = tokenizer.encode(split.heldout).ids[: model.ctx_window]
    if len(heldout_ids) > 1:
        window = torch.tensor([heldout_ids], dtype=torch.long, device=device)
        report.ablation = ablation.ablate(model, window, model_name).as_dict()

    note("score")
    breakdown = score.composite(
        bits_per_byte=corpus_report.bits_per_byte if corpus_report else math.nan,
        # Not ported yet; its weight redistributes rather than counting as zero.
        grammar_accuracy=math.nan,
        grammar_pairs=0,
        real_word_rate=metrics.real_word_rate,
        generated_words=metrics.words,
        distinct_3=metrics.distinct_3,
    )
    report.score = breakdown.as_dict()
    return report


@dataclass
class AssistantReport:
    """The assistant suite: bits per held-out reply byte, through a logistic."""

    model_name: str
    chat: dict | None = None
    score: dict = field(default_factory=dict)
    conversations: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


def run_assistant(model_name: str, device: str = "cpu", cancel: Event | None = None,
                  on_progress=None) -> AssistantReport:
    """Score held-out conversations through the same loss mask SFT trains on."""
    model, _ = checkpoint.load(storage.weights_path(model_name), device)
    tokenizer = bpe.load(storage.model_dir(model_name))

    conversations = chat_stage.load(storage.eval_dir() / "chat")
    scored: list[chat_stage.ReplyScore] = []
    for i, conversation in enumerate(conversations):
        if cancel is not None and cancel.is_set():
            break
        if on_progress and i % 10 == 0:
            on_progress({"stage": "assistant", "detail": f"{i}/{len(conversations)}"})
        reply = chat_stage.reply_logprob(model, tokenizer, conversation, device=device)
        if reply is not None:
            scored.append(reply)

    metrics = chat_stage.summarize(scored)
    return AssistantReport(
        model_name=model_name,
        chat=metrics.as_dict() if metrics else None,
        score=chat_stage.assistant_score(metrics),
        conversations=len(scored),
    )


@dataclass
class MarkupReport:
    """The markup suite: does the model use the marker grammar correctly?"""

    model_name: str
    positions: dict | None = None
    structure: dict | None = None
    score: dict = field(default_factory=dict)
    conversations: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


def run_markup(model_name: str, device: str = "cpu", cancel: Event | None = None,
               on_progress=None) -> MarkupReport:
    """Position agreement and generated-structure validity over held-out chat.

    The pair stage is not ported yet, so its 0.40 weight redistributes across the
    stages that did run rather than counting as zero.
    """
    model, _ = checkpoint.load(storage.weights_path(model_name), device)
    tokenizer = bpe.load(storage.model_dir(model_name))
    rules = Rules.build(tokenizer)

    conversations = chat_stage.load(storage.eval_dir() / "chat")
    if on_progress:
        on_progress({"stage": "markup", "detail": f"{len(conversations)} conversations"})

    positions = markup_stage.score_positions(model, tokenizer, conversations, rules, device=device)

    breakdown = markup_stage.composite_score(
        pair_accuracy=math.nan,   # marker-pair suite not ported yet
        pair_count=0,
        positions=positions,
        structure=None,
    )
    return MarkupReport(
        model_name=model_name,
        positions=positions.as_dict() if positions else None,
        structure=None,
        score=breakdown,
        conversations=len(conversations),
    )
