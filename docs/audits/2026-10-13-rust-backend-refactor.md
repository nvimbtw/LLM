# Rust backend refactor and independent review — 2026-10-13

The delegated agent refactored Rust coordination code using the current
AGENTS.md and C1–C12 conventions. The parent reviewed the implementation,
requested corrections, exercised the actual backend binary and verified the
PostgreSQL and wire contracts. Starting commit: `75225967a3`, on
`refactor/slate-architecture-and-code-quality`.

## Assessment

Keep this refactor. Request policy, persistence, browser transport and spawned
work now have understandable boundaries. Future changes to a tool result,
episode request or socket lifetime have a named place to go and meaningful
contract tests. The most valuable improvement is making ownership and effect
ordering visible. Most original Rust functions were already modest in length.

There is a cost: the changed concerns occupy 26 Rust files instead of nine and
2,956 physical lines instead of 2,402, including documentation and tests.
These additional modules separate real concerns; they are not a plugin framework
or generic repository layer. No performance, latency or memory improvement was
measured or claimed.

The work changes Rust under `backend/` and its documentation. Frontend source,
Python, dependencies, SQL query text, SQLx metadata and migrations are unchanged.
The new internal Rust turn signature is updated at its sole project caller;
HTTP routes, statuses, service JWTs and JSON/WebSocket schemas remain compatible.

## Structural evidence

Before is the exact starting Git commit, including the original test source.
After covers every changed/new Rust file, including the independent regression
and harness repair. Function measurements use parsed Rust syntax and physical
body spans; comment counts inspect those spans. Extraction reduces entry-file
size while increasing the number of focused modules.

| Changed Rust concerns | Before | After |
| --- | ---: | ---: |
| Files | 9 | 26 |
| Physical lines, including tests and docs | 2,402 | 2,956 |
| Named functions/methods | 109 | 155 |
| Functions with declaration documentation | 30 | 155 |
| Function-body comment lines | 84 | 0 |
| Bodies longer than 60 lines | 2 | 0 |
| Largest function body | 101 | 43 |
| Backend README lines | 659 | 217 |

| Entry file | Before | After |
| --- | ---: | ---: |
| `backend/src/main.rs` | 112 | 34 |
| `backend/crates/api/src/turn.rs` | 447 | 111 |
| `backend/crates/api/src/gym.rs` | 452 | 101 |
| `backend/crates/api/src/routes/stream.rs` | 379 | 108 |
| `backend/crates/api/src/conversation.rs` | 240 | 167 |
| `backend/crates/service/src/proxy.rs` | 438 | 296 |

The README retains its original heading anchors and links five focused topics.
The whole Rust workspace still contains older documentation and formatting debt;
these measurements describe the changed concerns.

## What is easier to change

`ChatTurn` binds the actual dependencies to an authenticated owner and chat ID.
Its private fields derive the service caller and tool-session key, removing the
possibility of callers supplying inconsistent versions of that identity.
`turn/request.rs` owns follow-up and steering policy, `turn/tools.rs` owns pending
tool tasks, and `turn/relay.rs` owns persistence before completion delivery.

`conversation/input.rs` owns request identity, attachments, canonical inputs,
history floors and token-range checks. `conversation.rs` exposes the order of
storage and service effects. Authorization still precedes preparation; accepted
inputs and file binding remain atomic; assistant persistence precedes success.
The SQL and transaction implementation were not rewritten.

WebSocket upgrade routes delegate to chat, job and gym sessions. A reader guard
owns each concurrent browser reader. Gym code separates episode interpretation,
generation, ordered commands and observation rendering, while keeping its driver
separate from stored chat because those workflows have different contracts.

The service proxy shares signing and status handling across JSON, multipart,
binary and streamed requests. Its decoder owns partial bytes. The startup module
owns supervisors through server drain; Actix workers receive distinct typed
proxies rather than subprocess ownership.

## Intentional behavior corrections

The old service adapter decoded UTF-8 separately for each arbitrary HTTP body
chunk. An independent public-Proxy test sent `ž漢🦀` with a split inside `ž`:
the original backend produced `��漢🦀`. The same real chunked-HTTP test passes
after the fix. The decoder retains raw bytes until a complete event exists.
Deterministic tests additionally cover every split and one-byte reads.

Reader cleanup now runs when a session future is dropped or unwinds, as well as
at normal completion. Tests drop blocked reader/tool owners and check resource
release; normal task completion is covered too. This does not undo external
work or claim immediate upstream model cancellation.

The pure inference-config helper now rejects primitive payloads before mutation.
The production preparation path already requires an object containing a model
name, so that defensive correction does not change public request behavior.

## Before and after examples

Snippets are source excerpts; the paths identify their containing operations.

### Chat identity and dependencies

Before, `turn::run` accepted ten arguments: socket, engine, registry, caller,
pool, owner, chat UUID, tool-session string, request and interjection receiver.
Caller/owner and chat UUID/session string could disagree.

After, its signature is:

```rust
pub async fn run(
    session: &mut actix_ws::Session,
    turn: ChatTurn<'_>,
    ask: Value,
    interjections: mpsc::UnboundedReceiver<String>,
)
```

`ChatTurn::new(engine, registry, pool, owner, chat_id)` derives the remaining
identities internally. This is an enforced invariant rather than a bag of the
same unrelated arguments.

### A reader has an owner during cancellation

Before, `routes/stream.rs` spawned a reader and manually called `reader.abort()`
only after the awaited turn returned. An early drop skipped that statement.

After, `routes/stream/reader.rs` gives the handle an owner:

```rust
impl Drop for SocketReader {
    /// Cancel the reader task on normal completion, session cancellation or unwinding.
    fn drop(&mut self) {
        self.0.abort();
    }
}
```

The session still explicitly drops the guard before closing the socket normally.
A dropped session now reaches the same cleanup through ownership.

### Policy is separate from persistence

Before, `conversation::prepare` parsed request IDs, selected a history floor,
validated attachments, converted messages and stripped request fields among its
service/database calls. Its body was 68 lines.

After, the central policy/effect sequence includes:

```rust
input::apply_history_floor(&mut history, &ask);
let files = input::attachment_ids(&ask)?;
let inputs = input::new_inputs(additions, request)?;
let turns = input::inference_turns(&history, &inputs);
queries::append_many(pool, owner, id, &inputs, &files)
    .await
    .map_err(database)?;
```

The operation's body is 26 lines. Boundary tests cover invalid UUIDs, five-file
limits, unexpected roles, invalid/overflowing token counts and opaque extension
fields. The storage operation still owns the transaction.

### Network chunks no longer define character boundaries

Before, `service/proxy.rs` decoded each chunk immediately:

```rust
buffer.push_str(&String::from_utf8_lossy(chunk.as_ref()));
```

After, `service/proxy/sse.rs` first buffers bytes:

```rust
self.pending_bytes.extend_from_slice(chunk);
```

Only after locating a complete LF-delimited event does it decode that byte span:

```rust
if let Some(frame) = parse(&String::from_utf8_lossy(&self.pending_bytes[consumed..end])) {
    frames.push(frame);
}
```

The adapter retains the established LF-only subset, data-line concatenation,
unknown event names, malformed-JSON-as-null and unfinished-tail behavior. CRLF
framing and a local partial-frame size limit remain future work.

### Comments describe the function contract

Before, the chat-session body contained this explanation:

```rust
// From here on the socket is read concurrently with the turn: a person may
// type and send again before this reply is done, and the only way to learn
// that is to keep reading. `stream` moves into this task for the rest of
// the turn's lifetime — `turn::run` never touches it directly, only the
// channel it feeds.
```

After, the function declaration has one contract sentence:

```rust
/// Authenticate, accept one question, then own a concurrent steering reader until the turn ends.
```

`SocketReader`, `ChatTurn` and the named stages express the body structure.
Needed lifecycle information remains at the declaration.

Before, `conversation::prepare` had no function documentation. After:

```rust
/// Authorize and prepare one round, then atomically persist its accepted input messages and bind uploads.
/// Service preparation precedes attachment validation; generation receives only authorized stateless history.
```

This records effect ordering that cannot be inferred from a type signature,
without narrating assignments inside the function.

## Independent verification

The existing PostgreSQL cluster has a template collation mismatch. Verification
used a fresh, unprivileged native PostgreSQL 18.6 cluster with C locale in a
temporary directory. No Docker, sudo or shared-database repair was needed.

Four original tool-round tests initially failed before assertions because their
SQLx Tokio runtime lacked Actix's local task set. The parent repaired only that
harness and proved all four against the original production source before using
them for comparison. Their conversation assertions are preserved.

| Check | Result |
| --- | --- |
| `SQLX_OFFLINE=true cargo build --workspace` | Passed, no compiler warnings. |
| Strict workspace/all-targets Clippy | Passed, no suppressions added; six previous API issues resolved. |
| Full workspace test suite against disposable PostgreSQL | 130 library tests and 56 exercised integration checks passed. Cargo reports 191 successes because five real-engine checks return early without the LLM virtualenv. |
| Original-source comparison with repaired test harness | 114 library tests and 55 exercised integration checks passed; the same five early returns. |
| Real HTTP UTF-8 regression | Failed on original source with the expected corrupted text; passed after refactor. |
| Final primitive-input boundary test | Passed after the final defensive helper correction. |
| Actual backend binary with four local service stubs | Health, signup/login/profile, all four distinct signed proxies, actor propagation and CORS passed. |
| Changed Rust formatter checks | Passed using each crate's edition. |
| Whole-workspace formatter | Still fails on exactly 17 untouched baseline files, down from 19; no new formatting debt. |
| Frontend production build for documentation | Passed after the module-topic split. No frontend source changes. |
| Diff whitespace, Markdown targets and source scope | Passed. |

The comparison initially shared a Cargo target directory between checkouts.
An isolated service test exposed a stale original library artifact; rebuilding
the service package made it pass without changing source. Final build, Clippy
and full-suite checks were rerun from the current checkout after that cleanup.
Future comparisons should use separate target directories.

The real-binary smoke test uses the disposable database and local HTTP stubs;
it is not evidence of working GPU inference or a real Python service stack.
The five early returns are three engine-auth seam checks and two real-engine
supervisor lifecycle checks. Their test functions are counted as successful by
the existing harness, but their intended assertions did not run.

## Review corrections and practical limits

The parent requested derived/private chat identities rather than independently
supplied duplicate fields, corrected per-crate formatter use, repaired the
existing chat-tool harness, added the public HTTP regression and verified actual
startup wiring. The agent also corrected a documentation claim of lossless
interjections: current content-based acknowledgements precede persistence.

Remaining work from the [original audit](2026-10-13-codebase.md) is still relevant:

- F06: joined-text interjection acknowledgements lack IDs and can lose or repeat
  steering. Persist IDs before acknowledgement and make replay idempotent.
- C7/F10: bound interjection queues, partial frames, response bodies and admission
  waits. This refactor exposes their owners but introduces no new budget policy.
- F19: the last allowed chat round can execute tools whose follow-up is discarded.
  Define admission/terminal-call policy before changing execution behavior.
- Gym truncation/error handling and best-effort status writes can produce an
  incorrect grade or status. Handle incomplete outcomes explicitly with tests.
- F11/F22: token revocation and password hashing on HTTP workers remain unchanged.
  Those require deliberate auth/session and execution-policy changes.
- The conversation lease remains process-local; multiple backend processes need
  a shared lease. Supervisor restart/health-wait shutdown policy is unchanged.
- Older routes, auth helpers, SQL layout and tool providers still contain style
  and documentation debt. This is an orchestration refactor, not complete Rust
  conformity with every rule.

No live model, real external search/shell or Docker stack was exercised. Smaller
entry files and more tests support maintainability; they do not establish a
performance gain or erase the remaining correctness findings.
