# Sprint ownership and recovery

Chat history prefetch, pending-request sharing and session cache invalidation
remain in `lib/chat-history.ts`. Chat sockets send a request UUID; completed
frames represent committed backend messages. See [conversation protocol](../backend/conversations.md).
The job board supports training, RL, tokenizer and benchmark work, with typed
`output` links; interactive chat has no job kind. New benchmarks offer English
language only. Historical suite tabs retain their original report interpretation.
Failed/cancelled training jobs offer **Recover latest**, using their stored
configuration; checkpoints offer **Keep** to exempt a generation from automatic
retention. Queue pause lasts only until engine restart. Testing and latency
comparison are separate from these implementation changes.

Legacy `language_v3` benchmark reports remain in a separate history tab with their
original score version. They cannot be newly queued or retried; the active
benchmark composer offers `language` only.

Benchmark model summaries allow unavailable Transformer layer, width and head
counts for custom definitions. Score cards omit those shape labels and retain
the actual parameter count. Compression remains unavailable when no reference
corpus is installed.
