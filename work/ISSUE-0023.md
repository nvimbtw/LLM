---
id: ISSUE-0023
type: issue
title: Dropped Results and unwraps without justification across backend (R3, R11)
scope: backend
priority: P1
status: done
rule: R3, R11
location: backend/src/chat.rs:153-159,176,183,389,392 backend/src/jobs.rs:88,95,98,124,138,143,150,208,212,263,264,274,282,285,286,312,329,366,370,388,394,398 backend/src/routes/training.rs:105-210
blocked-by: []
blocks: []
---

## Description
Pervasive `let _ = session.text(...)` / `let _ = session.close(...)` across chat.rs
and training.rs drop `Result` without comment. ~20 `.unwrap()` calls on
`Mutex::lock()` in jobs.rs have no poison-justification comment (despite
`catch_unwind` protecting the worker). `let _ = tok_tx.send(...)` and `let _ =
done_tx.send(...)` at chat.rs:389,392 drop send failures without note. Every dropped
`Result` or `unwrap` needs either a comment explaining why failure is
impossible/ignorable, or proper error propagation.

## Comments
### Builder · 2026-07-10
Note: the `chat.rs` sites are actually in `src/routes/chat.rs` (the issue's
`src/chat.rs` path was stale); all cited patterns were found there. Fixed by
scope:

**`jobs.rs` — two documented helpers instead of ~20 scattered justifications
(also satisfies R6):**
- Added `lock<T>(&Mutex<T>)` and routed all 12 `.lock().unwrap()` through it. It
  recovers the guard on poison (`unwrap_or_else(|p| p.into_inner())`) rather than
  `unwrap`-panicking: the guarded sections are single field reads/writes that
  can't panic, so poison isn't expected, but a long-running server shouldn't let
  one unlucky panic cascade `.unwrap()` panics through every socket handler. The
  rationale lives once, on the helper.
- Added `emit(&broadcast::Sender<String>, String)` and routed the 8 dropped
  `tx.send(...)` live-frame broadcasts through it. `send` errors only when no
  socket is attached (no receivers) — normal, since the job runs regardless and
  re-attaching sockets catch up from the `JobState` snapshot. One documented drop.
- `model_tx.send` (now line 364) was already justified by its existing comment
  (async db consumer; only fails at shutdown) — left as-is.

**`routes/chat.rs` — justifying comments** on the 5 `session.text/pong/close`
best-effort sends (a dead peer just ends the read loop) and on the
`tok_tx`/`done_tx` sends in the inference thread (receiver may be gone if the
turn was abandoned; generation winds down on its own).

**`routes/training.rs` — justifying comments** on every `session.*` best-effort
send and on the `.ok()` attach-frame parse (bad JSON ⇒ treated as "unknown job");
the two `job.state.lock().unwrap()` sites now call `jobs::lock`.

No behaviour change beyond the lock helper's poison-recovery (strictly fewer
panic paths, per R11). Verified: `cargo build` **0 warnings**, `cargo clippy`
**0 warnings** across the whole `backend` crate. A final grep confirms every
remaining `let _ =` / `.ok()` in the three files is either helper-internal or
carries an adjacent justification.

### Verifier · 2026-07-10 · pass

**What was checked:**

**(1) No bare `.lock().unwrap()`**  
Grepped `\.lock\(\)\.unwrap\(\)` across `backend/src/` → **0 matches**. All 12
locking sites in `jobs.rs` call the documented `lock()` helper (poison recovery).
Both lock sites in `routes/training.rs` call `jobs::lock()`. `routes/chat.rs` has
no mutex locks.

**(2) `emit` helper documented; all broadcast `tx.send` drops go through it**  
Verified `emit()` at `jobs.rs:415-417` with inline doc (lines 411-415). Grepped
all 8 `emit(&job.tx, …)` calls and confirmed none bypass it. The sole
`let _ = job.model_tx.send(record)` at line 364 has its own adjacent justification
(lines 350-355) — an async db consumer; only fails at shutdown.

**(3) Every `let _ =` / `.ok()` has an adjacent comment or lives in a documented helper**  

| File | `let _ =` | `.ok()` | Justification |
|------|-----------|---------|---------------|
| `jobs.rs` | 2 | 0 | 1 inside documented `emit` helper; 1 (`model_tx.send`) justified by 6-line comment |
| `routes/chat.rs` | 7 | 0 | All 7: "Best-effort" comments on dead-peer/abandoned-turn drops |
| `routes/training.rs` | 10 | 1 | All 10 `let _ =`: "Best-effort" comments; the 1 `.ok()` on attach-frame parse has inline justification (bad JSON ⇒ unknown job) |

**(4) `cargo build` and `cargo clippy` in `backend/` → 0 warnings each**  
- `cargo build` → `Finished dev profile … 0 warnings`  
- `cargo clippy` → `Finished dev profile … 0 warnings`

All four conditions satisfied → **pass**.
