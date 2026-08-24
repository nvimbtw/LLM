import sys
sys.path.insert(0, ".")
import os
os.environ["BENCH_ENGINE_DATABASE_URL"] = "sqlite:///:memory:"
import store
store.configure("sqlite:///:memory:", create=True)
from suites.language_v3.score import composite
from suites.language_v3.surface import Lexicon, analyze
from suites.language_v3.prompts import ALL_PROMPTS
from suites.language_v3.grammar import PAIRS

def test_prompts():
    assert len(ALL_PROMPTS) == 32

def test_grammar_pairs():
    assert len(PAIRS) == 30

def test_surface():
    lex = Lexicon.build("the cat sat on the mat")
    m = analyze(["the cat sat", "the dog ran"], lex)
    assert m.words == 6
    assert 0 <= m.real_word_rate <= 1

def test_composition():
    import math
    c = composite(1.5, 0.8, 30, 0.9, 200, 0.75, math.nan, math.nan)
    assert 0 < c.score < 100
    assert c.version == 3

def test_mock_benchmark():
    os.environ["BENCH_ENGINE_DATABASE_URL"] = "sqlite:///:memory:"
    store.configure("sqlite:///:memory:", create=True)
    from api.handlers import benchmark
    import threading
    evt = threading.Event()
    def prog(**kw): pass
    res = benchmark({"model_name": "pytest-mock", "suite": "language-v3"}, prog, evt, None)
    assert "benchmark_id" in res
    assert res["suite_version"] == 3
    from store import queries as q
    from store.schema import Suite
    with store.session() as db:
        rows = q.benchmarks_in(db, Suite.language_v3)
        assert len(rows) == 1
        r = rows[0]
        assert r.report is not None
        assert "surface" in r.report
        assert "grammar" in r.report
        assert len(r.report["samples"]) == 32

def test_api():
    import main
    assert main.app.title == "Bench engine"
    assert len(main.ROUTERS) == 3
