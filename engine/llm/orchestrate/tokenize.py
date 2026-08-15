"""The tokenizer pipeline: corpora on disk -> one named, reusable vocabulary.

Phase 1 of training, extracted and given a name. Nothing here touches the GPU;
it is a run so that it queues, reports and cancels like everything else that
takes minutes.
"""

import time
from threading import Event

from llm import storage
from llm.config import TokenizerConfig
from llm.corpus import vocab
from llm.train.progress import ProgressCallback, Tokenize, log_progress


def run(
    cfg: TokenizerConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
) -> dict:
    """Learn a vocabulary, or report the one already built for this request.

    Cancellation is coarse: learning merges is a single call into the tokenizers
    library, so a run cancelled mid-build stops when that call returns rather
    than at the next boundary.
    """
    started = time.time()
    selection = vocab.resolve(cfg.datasets)
    if not selection:
        raise FileNotFoundError(f"No corpora under {storage.input_dir()} to learn from")

    if not cfg.rebuild and vocab.matches(cfg.tokenizer_id, selection, cfg.max_vocab):
        built = vocab.info(cfg.tokenizer_id)
        print(f"[1/1] Reusing the vocabulary already built for {cfg.tokenizer_id}")
        stats = vocab.VocabStats(
            files=built["files"],
            vocab_size=built["vocab_size"],
            merges=built["merges"],
            datasets=[(name, cap) for name, cap in built["datasets"]],
            fingerprint=built["fingerprint"],
        )
        if on_progress is not None:
            on_progress(
                Tokenize(
                    files=stats.files,
                    vocab=stats.vocab_size,
                    merges=stats.merges,
                    progress=1.0,
                    done=True,
                )
            )
    else:
        named = ", ".join(f"{name}[:{cap}]" if cap else name for name, cap in selection)
        print(f"[1/1] Learning a vocabulary from {named} (max_vocab {cfg.max_vocab})")
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled before the build started")
        stats = vocab.build(cfg.tokenizer_id, selection, cfg.max_vocab, on_progress)

    print(
        f"      vocab {stats.vocab_size:,}, {stats.merges:,} merges, learned from "
        f"{stats.sampled_bytes / 1e6:.1f} MB sampled across {stats.sampled_files:,} of "
        f"{stats.files:,} file(s)"
    )
    return {
        "tokenizer_id": cfg.tokenizer_id,
        "vocab_size": stats.vocab_size,
        "merges": stats.merges,
        "max_vocab": cfg.max_vocab,
        "files": stats.files,
        # What the merges were fitted to, not just what was selected — the two
        # differ now, and a vocabulary that looks wrong is otherwise unanswerable.
        "sampled_files": stats.sampled_files,
        "sampled_bytes": stats.sampled_bytes,
        "datasets": [[name, cap] for name, cap in stats.datasets],
        "fingerprint": stats.fingerprint,
        "total_secs": round(time.time() - started, 1),
    }
