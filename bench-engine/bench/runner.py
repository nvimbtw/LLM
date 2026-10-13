"""English scoring policy. All model computation travels through Rust."""
import math
import os
import time
from pathlib import Path
from bench import corpus, score, surface
from bench.backend import Hop
from bench.prompts import DEFAULT_PROMPTS


def corpus_dir():
    return Path(os.environ.get('BENCH_CORPUS_DIR', str(Path(__file__).resolve().parents[1] / 'data' / 'corpus')))


def run(benchmark_id, config, owner, cancel):
    if cancel.is_set():
        return dict(model={"id":config["model_name"],"name":config["model_name"]},corpus=None,surface=None,ablation=None,score={},samples=[],suite="language",cancelled=True,total_secs=0.0)
    started = time.monotonic()
    split = corpus.load(corpus_dir())
    with Hop(owner,timeout=3600) as backend:
        raw, _ = backend.request('POST',f'/internal/inference/measure/{benchmark_id}',
                                 json={'config':config,'text':split.heldout,'prompts':DEFAULT_PROMPTS})
    report = dict(model=raw['model'],corpus=None,surface=None,ablation=raw['ablation'],
                  score={},samples=raw['samples'],suite='language',cancelled=raw['cancelled'] or cancel.is_set())
    if not report['cancelled']:
        measured = corpus.evaluate(raw['likelihood'],split)
        metrics = surface.analyze([sample['text'] for sample in raw['samples']],surface.Lexicon.build(split.train+split.heldout))
        report['corpus'] = measured.as_dict() if measured else None
        report['surface'] = metrics.as_dict()
        report['score'] = score.composite(bits_per_byte=measured.bits_per_byte if measured else math.nan,
            grammar_accuracy=math.nan,grammar_pairs=0,real_word_rate=metrics.real_word_rate,
            generated_words=metrics.words,distinct_3=metrics.distinct_3).as_dict()
    report['total_secs'] = round(time.monotonic()-started,1)
    return report
