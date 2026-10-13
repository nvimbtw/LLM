"""Versioned English scoring stays separate from model computation."""
import math
from bench import corpus, score


def test_likelihood_excludes_the_unscored_first_token_bytes():
    split = corpus.CorpusSplit(train='ordinary English prose '*100,heldout='some prose')
    raw = {'total':-10*math.log(2),'count':5,'unscored_bytes':2}
    measured = corpus.evaluate(raw,split)
    assert measured.bits_per_byte == 10/(len(split.heldout.encode())-2)
    assert measured.bits_per_token == 2
    assert measured.tokens_scored == 5


def test_unmeasured_corpus_is_absent():
    assert corpus.evaluate({'total':0,'count':0,'unscored_bytes':0},corpus.CorpusSplit('','')) is None


def test_score_keeps_its_version_and_range():
    report = score.composite(bits_per_byte=1.5,grammar_accuracy=math.nan,grammar_pairs=0,
                            real_word_rate=0.8,generated_words=100,distinct_3=0.9).as_dict()
    assert report['version'] == 2
    assert 0 <= report['score'] <= 100
