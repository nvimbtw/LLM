---
id: ISSUE-0008
title: Unwraps on file-parsed model data (R3, R11)
scope: backend/llm
priority: P1
status: done
rule: R3, R11
location: backend/src/llm/src/training/transformer.rs:935,944,1533,1543 backend/src/tokenize/src/bpe.rs:206-208
blocked-by: []
blocks: []
---

## Description
`read_2d` and `read_1d` closures inside `load` (lines 935, 944) and `from_bytes`
(lines 1533, 1543) use `.try_into().unwrap()` on byte slices from a model file. R3
requires a justification why failure is impossible (or use `?`/`expect` instead);
R11 explicitly forbids `unwrap` on file data. While `chunks_exact(4)` makes this
infallible, add a comment. Same pattern at `tokenize/src/bpe.rs:206-208`
(load_merges).

## Comments
### Verifier · migrated · bounce
transformer.rs now has justification comments on the unwraps (lines 954, 966), but
`tokenize/src/bpe.rs:209-211` still has 3 bare `.try_into().unwrap()` calls in
`load_merges` with no comment. The issue explicitly covers that location.

### Builder · 2026-07-10
Fix already committed in `5d964da` but the status was never advanced past the
bounce. `bpe.rs:205-207` now carries a justification comment above the loop
explaining that `chunks_exact(12)` structurally guarantees each 4-byte sub-slice
so `try_into` can't fail, and a corrupt trailing chunk is dropped rather than
panicking. This matches the accepted pattern in transformer.rs `read_2d`/`read_1d`
(lines 1056/1068). `cargo build -p tokenize` clean. Both named locations now
covered — advancing to verification.

### Verifier · 2026-07-10 · pass
Checked: (1) bpe.rs `load_merges` (lines 205–211) has a justification comment
above the loop citing `chunks_exact(12)` guaranteeing 4-byte sub-slices — pass.
(2) transformer.rs `read_2d`/`read_1d` unwraps (lines 1056, 1068) still carry
their `// chunks_exact(4) yields exactly-4-byte slices, so try_into can't fail here.`
comments — pass. (3) `cargo build -p tokenize` from `backend/` is clean (no
warnings, no errors) — pass. All three conditions hold; advancing to done.
