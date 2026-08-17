# A chat turn is not a queued run

The old system put chat in the same line as training: one device, one queue,
every use of it a row. A chat turn now bypasses that and is answered on the
socket that asked for it.

**This is a deliberate reversal of [0001](0001-one-queue-for-one-device.md), and
the tradeoff is real.** A turn asked while a run is training now contends with
it for the device instead of waiting for it. Both are slower for it.

The alternative is worse. Training runs are hours; a turn is seconds. A queue
that is correct about the device makes chat unusable for as long as anything is
training, which is most of the time the machine is interesting. Nobody waits
three hours to ask a question, so in practice the queue would be paused to chat —
which is the same contention, arrived at by hand.

`RunKind.chat` still exists in the schema. It has no handler, and a run with no
handler fails rather than hanging (`test_a_kind_with_no_handler_fails_rather_than_hanging`).
It is there for the day a turn is worth queueing again — a batch of evaluation
prompts, say — not as a path anything currently takes.

## What the queue is still the authority on

Training and benchmarks. Both are long, both write weights or scores, and both
are things you start and walk away from. For those the row is the point: it
outlives the tab, survives a restart, and is what a reattaching client reads.

## What a turn writes, and when

The reply row is written when the **last** token lands, not per token. A dropped
connection therefore loses the reply rather than storing half of one — which is
the honest outcome for something that took four seconds and can be asked again.

That is the opposite of the training contract, where the row exists before the
work starts precisely so closing the tab cannot lose it. The difference is cost:
losing four seconds of generation is an inconvenience, losing three hours of
training is not.
