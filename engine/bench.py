"""Throughput and parity harness for the training engine.

The port from the Rust/CUDA stack to PyTorch turned a shakedown pretrain from
about one hour into six. This measures where those hours go and proves that
making them fewer did not change what the model learns.

Three modes, because "faster" and "same" are different questions:

    micro    — steady-state ms/step with a per-phase breakdown (data, H2D,
               forward, backward, clip, optimizer). Warms up first, so
               torch.compile's one-time cost is not charged to the steps. This
               is the fast iteration loop.

    full     — end-to-end `loop.run`, timed at two different step counts. The
               difference between them divides out every fixed cost — compile,
               model construction, the first epoch_stream — and leaves the true
               marginal cost of a step, including the data pipeline that `micro`
               partly hides. This is the number that predicts a real run.

    parity   — the per-step loss curve, written to JSON. Two configurations that
               are supposed to be numerically identical must match bit for bit;
               a change that only trades precision must stay within tolerance.
               Speed work that silently changes the objective is not speed work.

Everything reads the real corpus and the real `DocSpan`s so the data pipeline
under test is the one that ships. Checkpoints go to a scratch DATA_DIR, never
the project's.

    .venv/bin/python bench.py micro
    .venv/bin/python bench.py full --steps 400
    .venv/bin/python bench.py parity --tag baseline
    .venv/bin/python bench.py parity --tag candidate --against baseline
"""

import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parent
# Before any storage lookup: the harness saves per-epoch checkpoints like any
# run, and those belong nowhere near data/artifacts.
os.environ.setdefault("DATA_DIR", str(ENGINE / "data"))
SCRATCH = Path(os.environ.get("BENCH_SCRATCH", "/tmp/llm-bench"))

import torch  # noqa: E402

from llm.config import TrainConfig  # noqa: E402
from llm.corpus.pack import DocSpan  # noqa: E402
from llm.model.transformer import Transformer  # noqa: E402
from llm.train import loop, plan as plan_mod  # noqa: E402
from llm.train.plan import DataPlan  # noqa: E402

# The architecture of the run this work exists to speed up, read off
# artifacts/models/<shakedown>/meta.json. `balanced` is the next size up, kept
# because optimizations that pay at 256d can stop paying at 512d — kernel launch
# overhead dominates the small one and arithmetic dominates the large one.
PRESETS = {
    "shakedown": dict(
        dimensions=256, heads=4, layers=6, ff_multiplier=4, ctx_window=512, batch_size=8
    ),
    "balanced": dict(
        dimensions=512, heads=8, layers=8, ff_multiplier=4, ctx_window=512, batch_size=8
    ),
}
VOCAB = 8192


# ── fixture ─────────────────────────────────────────────────────────────────


def fixture(tokens: int, seed: int = 0) -> tuple[np.ndarray, list[DocSpan]]:
    """A prefix of the real corpus, with the documents that fit inside it.

    Real spans matter: document lengths set how much padding `epoch_stream`'s
    separators add and how scattered its gather is, and a synthetic corpus of
    uniform documents would flatter the data pipeline exactly where it is
    weakest. Falls back to random tokens only when no corpus is packed, so the
    harness still runs on a fresh checkout.
    """
    from llm import storage

    corpus_path = storage.corpus_path()
    if not corpus_path.is_file():
        rng = np.random.default_rng(seed)
        corpus = rng.integers(0, VOCAB, size=tokens, dtype=np.uint16)
        docs = [
            DocSpan(dataset="synthetic", start=s, length=min(2000, tokens - s) - 1)
            for s in range(0, tokens, 2000)
        ]
        return corpus, [d for d in docs if d.length > 0]

    raw = np.memmap(corpus_path, dtype=np.uint16, mode="r")
    corpus = np.array(raw[:tokens])
    spans = json.loads(storage.segments_path().read_text())["documents"]
    docs = [
        DocSpan(dataset=d["dataset"], start=d["start"], length=d["length"], stage=0)
        for d in spans
        if d["start"] + d["length"] <= len(corpus)
    ]
    return corpus, docs


def build(preset: str, steps: int, **overrides) -> tuple[Transformer, np.ndarray, DataPlan, TrainConfig]:
    """Model, corpus, plan and config sized so the run is about `steps` long."""
    settings = dict(
        model_name="__bench__",
        epochs=1,
        max_vocab=VOCAB,
        warmup_steps=max(1, steps // 10),
        shuffle=True,
        seed=0,
        **PRESETS[preset],
    )
    settings.update(overrides)
    cfg = TrainConfig(**settings)

    # Slack for the separators epoch_stream inserts between documents, and for
    # the tail of the last document that falls outside the slice.
    want = steps * cfg.batch_size * cfg.ctx_window
    corpus, docs = fixture(int(want * 1.15) + cfg.ctx_window)
    plan = DataPlan(documents=docs, eot_id=0, shuffle=cfg.shuffle, seed=cfg.seed)

    vocab = max(VOCAB, int(corpus.max()) + 1)
    model = Transformer(
        vocab, cfg.dimensions, cfg.heads, cfg.layers, cfg.ff_multiplier, cfg.ctx_window
    )
    return model, corpus, plan, cfg


@contextmanager
def scratch_storage():
    """Point `storage` at a throwaway tree for the duration.

    `loop.run` saves a checkpoint per epoch like any run. The fixture is already
    in memory by the time this opens, so the corpus still comes from the real
    data directory and only the writes are diverted.
    """
    os.environ["DATA_DIR"] = str(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        os.environ["DATA_DIR"] = str(ENGINE / "data")


def actual_steps(corpus: np.ndarray, plan: DataPlan, cfg: TrainConfig) -> int:
    length = plan_mod.stream_length(len(corpus), plan)
    return plan_mod.batch_count(length, cfg.ctx_window, cfg.batch_size)


# ── micro ───────────────────────────────────────────────────────────────────


class Phase:
    """Wall-clock per phase, summed over the measured steps.

    CUDA events rather than `time.time()`: the kernels are queued asynchronously,
    so host timestamps around a launch measure the launch, not the work. Events
    are recorded into the same stream and read once at the end, which keeps the
    measurement off the critical path — a `synchronize()` per phase per step
    would itself become the thing being measured.
    """

    def __init__(self, device: str):
        self.on_cuda = device == "cuda"
        self.pairs: dict[str, list] = {}
        self.host: dict[str, float] = {}
        self._open: tuple[str, object] | None = None

    def gpu(self, name: str):
        return _Span(self, name)

    def add_host(self, name: str, seconds: float) -> None:
        self.host[name] = self.host.get(name, 0.0) + seconds

    def totals(self, steps: int) -> dict[str, float]:
        """Mean milliseconds per step, per phase."""
        if self.on_cuda:
            torch.cuda.synchronize()
        out = {n: v * 1000 / steps for n, v in self.host.items()}
        for name, pairs in self.pairs.items():
            out[name] = sum(a.elapsed_time(b) for a, b in pairs) / steps
        return out


class _Span:
    def __init__(self, phase: Phase, name: str):
        self.phase = phase
        self.name = name

    def __enter__(self):
        if self.phase.on_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.stop = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.phase.on_cuda:
            self.stop.record()
            self.phase.pairs.setdefault(self.name, []).append((self.start, self.stop))
        else:
            self.phase.add_host(self.name, time.perf_counter() - self.t0)


def micro(args) -> None:
    """Steady-state step cost, broken down by phase.

    Mirrors `loop.run`'s step rather than calling it: the loop has no step budget
    and no place to hang per-phase timers, and adding either to it for the
    benefit of a benchmark would put measurement scaffolding in the training
    path. It reuses the loop's own helpers, so precision, compilation and the
    optimizer are configured by the code under test, not by a copy of it.
    """
    device = loop.pick_device()
    total = args.warmup + args.steps
    model, corpus, plan, cfg = build(
        args.preset,
        total,
        precision=args.precision,
        compile_model=not args.no_compile,
        allow_tf32=not args.no_tf32,
    )
    model.to(device).train()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
        torch.cuda.reset_peak_memory_stats()

    amp = loop.autocast_dtype(cfg.precision, device)
    compiled_at = time.time()
    net = loop.compiled(loop.step_fn(model, model.vocab), cfg.compile_model and device == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay, fused=(device == "cuda"),
    )

    stream, stream_mask = plan_mod.epoch_stream(corpus, plan, 0, None)
    batches = plan_mod.iter_batches(stream, cfg.ctx_window, cfg.batch_size, stream_mask)

    phase = Phase(device)
    measuring = False
    step = 0
    started = 0.0

    while step < total:
        if step == args.warmup:
            # Everything before this point paid for compilation, autotuning and
            # the allocator's first-touch growth. Timers start clean.
            if device == "cuda":
                torch.cuda.synchronize()
            phase = Phase(device)
            measuring = True
            started = time.perf_counter()

        t0 = time.perf_counter()
        try:
            inputs, targets, weights = next(batches)
        except StopIteration:
            break
        if measuring:
            phase.add_host("data", time.perf_counter() - t0)

        with phase.gpu("h2d") if measuring else _nothing():
            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)

        with phase.gpu("forward") if measuring else _nothing():
            with torch.autocast(device_type="cuda", dtype=amp, enabled=amp is not None):
                loss = net(x, y, None)

        with phase.gpu("backward") if measuring else _nothing():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

        with phase.gpu("clip") if measuring else _nothing():
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        with phase.gpu("optimizer") if measuring else _nothing():
            optimizer.step()

        with phase.gpu("loss_item") if measuring else _nothing():
            loss.item()

        step += 1

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    done = step - args.warmup
    tokens = done * cfg.batch_size * cfg.ctx_window

    print(f"\n{args.preset}  {cfg.dimensions}d/{cfg.heads}h/{cfg.layers}L/"
          f"ctx{cfg.ctx_window}/bs{cfg.batch_size}  "
          f"{model.parameter_count():,} params  {device}")
    print(f"precision={cfg.precision} compile={cfg.compile_model} tf32={cfg.allow_tf32}"
          f"  (setup {time.time() - compiled_at - elapsed:.0f}s)")
    print(f"\n  {done} steps in {elapsed:.2f}s")
    print(f"  {elapsed / done * 1000:.2f} ms/step")
    print(f"  {tokens / elapsed:,.0f} tokens/sec")
    if device == "cuda":
        print(f"  {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB peak")

    print("\n  phase breakdown (ms/step, GPU phases overlap the host):")
    for name, ms in sorted(phase.totals(done).items(), key=lambda kv: -kv[1]):
        print(f"    {name:<12} {ms:7.3f}   {ms / (elapsed / done * 1000) * 100:5.1f}%")


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ── full ────────────────────────────────────────────────────────────────────


def full(args) -> None:
    """Marginal cost of a step through the real `loop.run`.

    Timed at two corpus sizes and differenced. A single timed run charges the
    steps for compilation, model construction and the first `epoch_stream`,
    which at a few hundred steps is most of the wall-clock; the difference
    between a short run and a long one cancels all of it, without the loop
    needing to know it is being measured.

    Cancelling it requires the fixed cost to be the *same* in both, which is why
    a throwaway run goes first. torch.compile is per-process: charge it to the
    short run and it never appears in the long one, the difference absorbs a
    negative constant, and the marginal step comes out roughly half what it is.
    """
    device = loop.pick_device()
    short = max(20, args.steps // 4)
    points = []

    for want in (short, short, args.steps):
        model, corpus, plan, cfg = build(
            args.preset,
            want,
            precision=args.precision,
            compile_model=not args.no_compile,
            allow_tf32=not args.no_tf32,
        )
        steps = actual_steps(corpus, plan, cfg)
        started = time.perf_counter()
        with scratch_storage():
            summary = loop.run(model, corpus, plan, cfg, device, on_progress=None)
        points.append((summary["steps"], time.perf_counter() - started))
        label = "warm-up (discarded)" if len(points) == 1 else f"planned {steps}"
        print(f"  {summary['steps']:>6} steps  {points[-1][1]:7.2f}s  ({label})", flush=True)

    (n1, t1), (n2, t2) = points[1:]
    if n2 <= n1:
        raise SystemExit("the long run was not longer; raise --steps")

    per_step = (t2 - t1) / (n2 - n1)
    fixed = t1 - per_step * n1
    tokens = cfg.batch_size * cfg.ctx_window

    print(f"\n{args.preset}  precision={cfg.precision} compile={cfg.compile_model}"
          f" tf32={cfg.allow_tf32}  {device}")
    print(f"  {per_step * 1000:.2f} ms/step marginal")
    print(f"  {tokens / per_step:,.0f} tokens/sec")
    print(f"  {fixed:.1f}s fixed overhead (compile, setup, first epoch stream)")
    # The headline the work is measured against: the shakedown corpus is 56M
    # tokens over 7 epochs, which is what "one hour became six" refers to.
    epoch_tokens = 56_000_000
    hours = epoch_tokens * 7 / (tokens / per_step) / 3600
    print(f"  → {hours:.2f}h projected for a 7-epoch 56M-token shakedown")


# ── parity ──────────────────────────────────────────────────────────────────


def parity(args) -> None:
    """The per-step loss curve, saved for comparison.

    Written through the real loop with a report every step, so the numbers are
    the ones the loop itself computes. `--against` compares two curves: an
    optimization that only reorganizes work must reproduce the baseline exactly,
    and one that trades precision must stay inside `--tol`.
    """
    device = loop.pick_device()
    torch.manual_seed(1234)
    model, corpus, plan, cfg = build(
        args.preset,
        args.steps,
        precision=args.precision,
        compile_model=not args.no_compile,
        allow_tf32=not args.no_tf32,
        # Determinism first: a fixed corpus and a fixed document order, so the
        # only thing that can move the curve is the arithmetic.
        shuffle=False,
    )

    losses: list[float] = []
    # A report every step. The loop derives its interval from the run's length,
    # so a run of a few hundred steps reports on every one of them.
    def capture(progress):
        if not progress.epoch_done:
            losses.append(progress.loss)

    with scratch_storage():
        loop.run(model, corpus, plan, cfg, device, on_progress=capture)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"parity-{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag,
        "preset": args.preset,
        "precision": cfg.precision,
        "compile": cfg.compile_model,
        "tf32": cfg.allow_tf32,
        "losses": losses,
    }, indent=1))
    print(f"{len(losses)} steps, final loss {losses[-1]:.6f} → {out}")

    if not args.against:
        return

    ref = json.loads((SCRATCH / f"parity-{args.against}.json").read_text())
    a, b = ref["losses"], losses
    n = min(len(a), len(b))
    if n == 0:
        raise SystemExit("no overlapping steps to compare")
    diffs = [abs(x - y) for x, y in zip(a[:n], b[:n])]
    worst = max(diffs)
    exact = sum(1 for d in diffs if d == 0.0)

    print(f"\nvs {args.against} ({ref['precision']}, compile={ref['compile']}):")
    print(f"  {n} steps compared, {exact} bit-identical")
    print(f"  max |Δloss| {worst:.3e}   mean {statistics.fmean(diffs):.3e}")
    print(f"  final {a[n-1]:.6f} → {b[n-1]:.6f}")
    verdict = "IDENTICAL" if worst == 0 else ("within tol" if worst <= args.tol else "DIVERGED")
    print(f"  {verdict} (tol {args.tol:.0e})")
    if worst > args.tol:
        raise SystemExit(1)


# ── check ───────────────────────────────────────────────────────────────────


def check(args) -> None:
    """Assert the invariants the speed work is not allowed to break.

    Curve comparison catches numeric drift but is blunt about where it came
    from: a run that ends at the same loss can still be batching the corpus
    differently. These pin the pieces that were rewritten for speed against a
    literal transcription of what they replaced, so a difference names itself.
    """
    failures = []

    def expect(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("batching")
    corpus, docs = fixture(200_000)
    plan = DataPlan(documents=docs, eot_id=0, shuffle=True, seed=7)
    stream, _ = plan_mod.epoch_stream(corpus, plan, 0, None)
    mask = (np.arange(len(stream)) % 3 == 0).astype(np.uint8)

    ctx, batch = 128, 6
    new = list(plan_mod.iter_batches(stream, ctx, batch, mask))
    old = list(_iter_batches_rowwise(stream, ctx, batch, mask))
    expect("same number of batches", len(new) == len(old), f"{len(new)}")
    expect(
        "inputs, targets and weights identical to the row-at-a-time version",
        all(
            np.array_equal(a, c) and np.array_equal(b, d) and np.array_equal(e, f)
            for (a, b, e), (c, d, f) in zip(new, old)
        ),
    )
    inputs, targets, _ = new[0]
    expect(
        "targets are inputs shifted one token left",
        np.array_equal(inputs[0, 1:], targets[0, :-1]),
    )

    print("loss")
    device = loop.pick_device()
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 97, device=device)
    y = torch.randint(0, 97, (2, 16), device=device)
    w = (torch.rand(2, 16, device=device) > 0.5).float()
    unmasked = loop._loss(logits, y, None, 97)
    expect(
        "unweighted loss is plain cross-entropy",
        torch.equal(
            unmasked, torch.nn.functional.cross_entropy(logits.reshape(-1, 97), y.reshape(-1))
        ),
    )
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 97), y.reshape(-1), reduction="none"
    )
    expected = (per_token * w.reshape(-1)).sum() / w.sum()
    expect(
        "masked loss averages over trainable tokens only",
        torch.equal(loop._loss(logits, y, w, 97), expected),
    )

    print("loss log")
    log = loop._LossLog(4, device)
    for v in [3.0, 1.0, 4.0, 1.0, 5.0]:
        log.record(torch.tensor(v, device=device))
    log.flush()
    expect("records every step in order", log.values == [3.0, 1.0, 4.0, 1.0, 5.0])
    expect("epoch view matches", log.epoch == log.values)
    log.start_epoch()
    expect("start_epoch clears the epoch view but not the run", log.epoch == [] and len(log.values) == 5)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    print("all checks passed")


def _iter_batches_rowwise(stream, ctx_window, batch_size, mask=None):
    """`iter_batches` as it was written before it was vectorized.

    Kept here, in the harness rather than in the engine, purely as the reference
    the fast version is diffed against. If the two ever disagree the fast one is
    wrong — the semantics are this.
    """
    windows = (len(stream) - 1) // ctx_window
    usable = windows - (windows % batch_size)

    for start in range(0, usable, batch_size):
        inputs = np.empty((batch_size, ctx_window), dtype=np.int64)
        targets = np.empty((batch_size, ctx_window), dtype=np.int64)
        weights = np.empty((batch_size, ctx_window), dtype=np.float32) if mask is not None else None

        for row in range(batch_size):
            offset = (start + row) * ctx_window
            inputs[row] = stream[offset : offset + ctx_window]
            targets[row] = stream[offset + 1 : offset + ctx_window + 1]
            if weights is not None:
                weights[row] = mask[offset : offset + ctx_window]

        yield inputs, targets, weights


# ── cli ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--preset", default="shakedown", choices=sorted(PRESETS))
        p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
        p.add_argument("--no-compile", action="store_true")
        p.add_argument("--no-tf32", action="store_true")
        return p

    m = common(sub.add_parser("micro", help="steady-state ms/step with a phase breakdown"))
    m.add_argument("--steps", type=int, default=60)
    m.add_argument("--warmup", type=int, default=15)
    m.set_defaults(fn=micro)

    f = common(sub.add_parser("full", help="marginal ms/step through the real loop"))
    f.add_argument("--steps", type=int, default=300)
    f.set_defaults(fn=full)

    p = common(sub.add_parser("parity", help="per-step loss curve, and compare two"))
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--tag", default="run")
    p.add_argument("--against", default=None)
    p.add_argument("--tol", type=float, default=1e-3)
    p.set_defaults(fn=parity)

    sub.add_parser("check", help="assert the speed work preserved behaviour").set_defaults(
        fn=check
    )

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
