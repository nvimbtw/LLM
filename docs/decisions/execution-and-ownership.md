# Execution records follow ownership; admission follows the device

Date: 2026-10-09

Chats and files move to Rust because that process owns users, authorization,
WebSockets and tool execution. Tokenizer-dependent preparation stays in Python;
replicating templates/counts in Rust would introduce a second model vocabulary.
Completed replies are committed before success frames, with request IDs detecting
retries. Cross-database model FKs become logical IDs and attribution snapshots.

English scoring moves to bench-engine with its own history database. Giving it
weights would duplicate loading/device ownership; it instead requests model
measurements through Rust. Other suites survive as history without active
execution choices. Storage transfer compares rows before source archival;
silent deletion or best-effort partial copying would make rollback ambiguous.

A common queue remains in the LLM engine because ordering, dependencies, cancel
and event subscriptions need one authority. Training and tokenizer domain records
hold their respective configuration/output contracts. Benchmark dispatches hold
requests and remote IDs, while reports live with the scoring policy. Duplicating
queue status in every domain service was rejected because two copies could
contradict each other after a crash. Report status describes a measurement;
queue status describes its scheduled execution.

One gate covers model loading, training, benchmark measurements and interactive
inference in the single LLM process. Dispatching a benchmark while holding the
gate was rejected: the returned inference request would deadlock. Interactive
calls refuse busy rather than evict a model under training. Queue pause is
process-local control, never a durable stopped state inherited after reboot.

Recovery generations remain immutable models, so references never silently
change weights. Retention removes superseded automatic saves while preserving
published, explicitly kept and referenced generations. Optimizer state no longer
repeats the weights file. Atomic publication protects the previous point through
write failure; inability to register a published point stops further checkpoint
production so orphan storage cannot grow without bound.
