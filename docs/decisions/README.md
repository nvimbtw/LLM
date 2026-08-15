# Decision records

Findings that cost real time to learn, kept out of the source files they
describe. A record exists when the *why* would otherwise be re-derived — or
re-broken — by someone doing the obvious thing.

| # | Record | Constrains |
|---|---|---|
| 0000 | [Archive: the Rust CUDA engine](0000-archive-rust-cuda-engine.md) | nothing — history |
| 0001 | [One queue for one device](0001-one-queue-for-one-device.md) | scheduling |
| 0002 | [Three benchmarks, not one](0002-three-benchmarks-not-one.md) | `llm/eval/` |
| 0003 | [The marker grammar is enforced while decoding](0003-marker-grammar-enforced-while-decoding.md) | `llm/chat/markup.py` |
| 0004 | [Corpus ingestion streams](0004-corpus-ingestion-streams.md) | `llm/corpus/` |
| 0005 | [Vocabulary layout](0005-vocabulary-layout.md) | `llm/corpus/special.py` |
| 0006 | [Pretrain and SFT are one pipeline](0006-pretrain-and-sft-are-one-pipeline.md) | `llm/orchestrate/` |
| 0007 | [The thinking budget](0007-the-thinking-budget.md) | `llm/chat/template.py` |
| 0008 | [Two databases, one opaque owner](0008-two-databases-one-opaque-owner.md) | the seam between backend and engine |
| 0009 | [A chat turn is not a queued run](0009-a-chat-turn-is-not-a-queued-run.md) | scheduling |
| 0010 | [A corpus is a function of its vocabulary](0010-a-corpus-is-a-function-of-its-vocabulary.md) | `llm/corpus/`, `llm/orchestrate/train.py` |
| 0011 | [A build may not scale with its corpus](0011-a-build-may-not-scale-with-its-corpus.md) | `llm/corpus/`, `llm/train/diagnostics.py` |

0000 is the exception: it describes code that no longer exists. It is kept
because the failures it records — a backward pass that never ran, a benchmark
denominated in the wrong unit — are ones a reimplementation walks straight back
into.

Several records predate the redesign and describe the system as it was when the
finding was made. Where a record and the code disagree, the code is current and
the record is evidence. 0001 is the largest of those: its scheduling half is
superseded by 0008 and 0009, and everything it says about the post-train cycle
describes code that has since been deleted — kept because the measurements in it
are the reason it was.
