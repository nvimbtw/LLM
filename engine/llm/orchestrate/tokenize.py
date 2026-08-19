"""The tokenizer pipeline: datasets in the data engine -> one named vocabulary.

Phase 1 of training, extracted and given a name. Nothing here touches the GPU;
it is a run so that it queues, reports and cancels like everything else that
takes minutes.
"""

import time
from threading import Event

from llm.config import TokenizerConfig
from llm.corpus import remote, vocab
from llm.train.progress import ProgressCallback, Tokenize, log_progress


def run(
    cfg: TokenizerConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
    owner: str | None = None,
) -> dict:
    """Learn a vocabulary, or report the one already built for this request.

    Cancellation is coarse: learning merges is a single call into the tokenizers
    library, so a run cancelled mid-build stops when that call returns rather
    than at the next boundary.
    """
    started = time.time()
    with remote.Client(owner) as client:
        return _build(cfg, client, on_progress, cancel, started)


def _build(cfg, client, on_progress, cancel, started) -> dict:
    selection = vocab.resolve(client, cfg.datasets)

    if not cfg.rebuild and vocab.matches(cfg.tokenizer_id, selection, cfg.max_vocab):
        built = vocab.info(cfg.tokenizer_id)
        print(f"[1/1] Reusing the vocabulary already built for {cfg.tokenizer_id}")
        stats = vocab.VocabStats(
            datasets=selection,
            vocab_size=built["vocab_size"],
            merges=built["merges"],
            fingerprint=built["fingerprint"],
            sampled_chunks=built.get("sampled_chunks", 0),
            sampled_bytes=built.get("sampled_bytes", 0),
        )
        if on_progress is not None:
            on_progress(
                Tokenize(
                    vocab=stats.vocab_size,
                    merges=stats.merges,
                    progress=1.0,
                    done=True,
                )
            )
    else:
        named = ", ".join(str(selected) for selected in selection)
        print(f"[1/1] Learning a vocabulary from {named} (max_vocab {cfg.max_vocab})")
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled before the build started")
        stats = vocab.build(client, cfg.tokenizer_id, selection, cfg.max_vocab, on_progress)

    print(
        f"      vocab {stats.vocab_size:,}, {stats.merges:,} merges, learned from "
        f"{stats.sampled_bytes / 1e6:.1f} MB drawn across {stats.sampled_chunks:,} "
        f"point(s) in {len(stats.datasets)} dataset(s)"
    )
    return {
        "tokenizer_id": cfg.tokenizer_id,
        "vocab_size": stats.vocab_size,
        "merges": stats.merges,
        "max_vocab": cfg.max_vocab,
        # What the merges were fitted to, not just what was selected — the two
        # differ now, and a vocabulary that looks wrong is otherwise unanswerable.
        "sampled_chunks": stats.sampled_chunks,
        "sampled_bytes": stats.sampled_bytes,
        "datasets": [selected.record() for selected in stats.datasets],
        "fingerprint": stats.fingerprint,
        "total_secs": round(time.time() - started, 1),
    }
