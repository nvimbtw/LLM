# Performance audit — front end, Rust backend and the seam between them

**Date:** 2026-10-29
**Branch:** `perf/audit-and-optimize` (from `dev`, uncommitted)
**Scope:** the browser application, the Rust API, and everything that carries a
message between them. Model inference itself is out of scope; the per-token
plumbing around it is not.

This is an audit with implementation. It records what was measured, what was
changed, what the change bought, and what is still worth doing. It does not
replace any module contract.

---

## 1. Method

1. Read `AGENTS.md`, `docs/conventions.md`, `docs/README.md`,
   `docs/frontend/README.md` and `docs/backend/README.md` from the working tree,
   then the whole chat path: socket adapter, turn reducer, session, hook,
   thread, message renderer, parser, composer, sidebar catalog, and on the Rust
   side the turn relay, the service SSE decoder and the conversation queries.
2. Wrote a harness that replays a streamed reply in 40-character increments
   through `parseMessage`, at sizes from 3.2k to 129k characters, summing the
   time of every call — the shape a real reply has, where the parser is asked
   for the accumulated message once per token.
3. Implemented the changes that the measurements justified, then re-ran the
   same harness and the identity checks below.
4. Verified with `typecheck`, `lint`, `bun test`, `next build`, and on the Rust
   side `cargo build`, `cargo clippy -D warnings` and `cargo test --workspace`.

---

## 2. The path a token takes

```
Python: next(generator) ──► asyncio.to_thread ──► "event: token"
   │                                                 │
   │                                    Rust service proxy (sse::frames)
   │                                                 │
   │                              api::turn::relay → one WS text frame
   │                                                 │
   │                              browser WebSocket → JSON.parse
   │                                                 │
   │                              ChatSession.receive → reduceChatTurn
   │                                                 │
   │                              changeMessages → setMessages → React render
   │                                                 │
   └─────────────────────────────► parseMessage(entire reply) + full subtree render
```

Three of those hops are per token and were doing work proportional to the
*whole reply*, not to the token:

- `parseMessage` re-scanned the accumulated message from index 0 on every call,
  so a reply of length *n* delivered in *k* pieces cost O(n²).
- The renderer rebuilt every node of the tree on every commit, including the
  `word-ink` spans that had finished animating long ago.
- The composer, its Tiptap editor, the frosted surface over the transcript, the
  question rail and the sidebar context all re-rendered per token as well.

At 60 tokens per second the first of those was already the largest cost in the
application; together they meant that a long answer got slower to draw the
longer it got, which is exactly the wrong direction.

---

## 3. Measured cost before the change

Synthetic reply of nested model markup, replayed in 40-character increments.
Total time is the sum over every `parseMessage` call in the run, i.e. the
parsing cost of writing that reply once:

| Reply length | Calls | Total parse time | Per call |
| --- | --- | --- | --- |
| 3.2k chars | 80 | 30.5 ms | 0.38 ms |
| 12.9k chars | 323 | 188 ms | 0.58 ms |
| 32.2k chars | 805 | 1.09 s | 1.35 ms |
| 64.4k chars | 1 610 | 4.31 s | 2.68 ms |
| 128.8k chars | 3 220 | 17.19 s | 5.34 ms |

The per-call cost grows with the reply, which is the quadratic term: the same
work is repeated on every token. The last row is a single reply whose formatting
alone would occupy roughly 17 seconds of main-thread time spread over its
arrival.

---

## 4. What changed

### 4.1 One state update per frame, not per token

`frontend/lib/frame-coalescer.ts` (new), `frontend/hooks/use-chat.ts`

`ChatSession` still notifies once per token — it is the authoritative
accumulator and its behaviour is covered by `lib/chat-session.test.ts` — but
`useChat` no longer turns each notification into a React update. Stream
notifications are buffered, and the newest one is applied on the next animation
frame. Nothing is lost: a streamed message is accumulated, not appended, so every
value between two frames is superseded by the last.

Anything that must not be overtaken flushes first: a new turn (`start`), a
history read (`history`), and unmount. A `start` that arrived behind an
unflushed token can therefore never appear before the text it follows.

`FrameCoalescer` is deliberately separate from React and unit-tested
(`lib/frame-coalescer.test.ts`): a burst delivers only its last value, a frame
schedules the next one, `flush` delivers immediately, `dispose` drops.

### 4.2 The parser resumes instead of restarting

`frontend/lib/message-parser.ts`, `frontend/lib/math-parser.ts`

`parseMessage` keeps a checkpoint of the last message it parsed: the tree at the
last complete tag, the open-element path, and the offset where the still-arriving
text begins. A call whose input extends the previous one continues from that
offset; anything else (another chat, a user bubble, a replaced reply) takes the
ordinary full parse.

Two details make it safe rather than clever:

- The trailing text is **not** stored in the checkpoint. It is still-arriving
  prose that may contain half a tag, so it is attached to the copy that is handed
  out and re-scanned on the next call. A partial `<|pa` is text now and an element
  the moment it closes.
- Maths is lifted per text run at the moment that run stops growing, instead of
  by a second pass over the finished tree. `splitText` is reused from
  `math-parser.ts`, and the literal set (`code block`, `code inline`,
  `tool-call`) is now exported as `MATH_LITERAL` so both callers agree.

Equality with the old full parse is asserted for every 3-character prefix of a
mixed reply (headings, bold, inline code, thinking, links, lists, a code block
containing `$`, a divider, a quote, inline and display maths) in
`lib/message-parser.test.ts`.

| Reply length | Before | After | Ratio |
| --- | --- | --- | --- |
| 3.2k chars | 30.5 ms | 4.3 ms | 7× |
| 12.9k chars | 188 ms | 28 ms | 6.6× |
| 32.2k chars | 1.09 s | 77 ms | 14× |
| 64.4k chars | 4.31 s | 265 ms | 16× |
| 128.8k chars | 17.19 s | 1.13 s | 15× |

What remains is the prefix check (`input.startsWith(checkpoint.source)`, one
linear comparison per call) plus the copy of the open path. Both are linear in
the reply and cheap per byte, which is why the ratio is roughly constant rather
than improving with size. At a realistic reply length the cost is now far below
a frame.

### 4.3 The tree keeps the identity of everything a token did not touch

`frontend/lib/message-parser.ts`, `frontend/components/chat/rich-message.tsx`

The parser now builds the tree by copying only the path from the changed text
node to the root; every sibling keeps its object identity. `RichContent` renders
each top-level block through a memoised `RichBlock` that compares the node by
identity plus the four render flags, so a block nothing was appended to is
skipped outright.

Measured on a 400-block, 37k-character reply replayed in 40-character steps:
**1.4 top-level blocks redrawn per frame, 99.3% of blocks skipped.** Before, all
400 were rebuilt on every token.

### 4.4 Streaming text settles into plain text

`frontend/components/chat/inked.tsx`

Inside a block that *is* growing, `Inked` split the whole run into per-word
spans on every render — for a long paragraph that is again O(n²), now in element
allocation instead of parsing. The component now keeps only the last 240
characters live (as animated spans) and renders the settled prefix as one
memoised text node.

That is visually identical: `word-ink` is an arrival animation
(`opacity` 0.3 → 1, `letter-spacing` 0.055em → normal, 340 ms) and a settled word
carries no styling the characters do not. The window is long enough that a word
finishes moving before it settles.

### 4.5 The composer is off the per-token path

`frontend/components/chat/chat-composer.tsx`, `frontend/hooks/use-throttled-number.ts`
(new), `frontend/components/chat/chat-thread.tsx`

`ChatComposer` is the largest subtree under the thread: a Tiptap editor, a model
picker, an upload strip and a `backdrop-blur-2xl backdrop-saturate-150` surface
over scrolling content. It is now memoised, and the two props that used to be
inline arrows (`onRenameTitle`, `onPruneContext`) are `useCallback`s so the memo
can actually hold.

The one prop that genuinely changes while a reply streams — the context token
figure — is passed through `useThrottledNumber(…, 250)`: the count is followed
at four updates a second instead of sixty. The composer now redraws when it has
something new to say rather than whenever a token arrives.

### 4.6 The question rail redraws only its current tick

`frontend/components/chat/chat-position-rail.tsx`

Each tick became a memoised `QuestionMark` with a stable jump callback, so a
frame that changes neither a question nor the current position rebuilds nothing.
Previously every tooltip, label normalisation and button was recreated per
token.

### 4.7 Registry reads are shared

`frontend/hooks/use-models.ts`, `frontend/hooks/use-model-names.ts`

The chat page mounted `useModels` twice (composer and picker) and the train board
three times; the gym page mounted `useModelNames` twice, and that request carries
every checkpoint. Each mount was its own request. Both hooks now share a
short-lived module cache with in-flight de-duplication — 30 s for `/models`,
60 s for `/models?checkpoints=true` — invalidated by the caller's refresh key and
by `deleteModel`.

### 4.8 A queue poll that says nothing costs nothing

`frontend/hooks/use-jobs.ts`

`GET /runs` is polled every three seconds for the life of the jobs page. Most
answers are byte-identical to the board already on screen, and `setBoard` still
re-rendered every row, chart and control. The payload is now compared with the
last adopted one and dropped when unchanged.

### 4.9 The chats context value is stable

`frontend/components/chat/chats-provider.tsx`

The provider handed a freshly built object to every consumer on every provider
render. It is now a `useMemo` over the snapshot and the callbacks, so a
streaming thread no longer implies a new context for the sidebar.

### 4.10 The local plan is read once per change

`frontend/hooks/use-local-plan.ts`

`useSyncExternalStore` calls its snapshot on every render of every subscriber,
and the snapshot was a synchronous `localStorage.getItem` — once per token per
subscriber. The raw value is now cached per chat and the cache is cleared by the
same events that invalidate the store.

### 4.11 The database pool is sized, and configurable

`backend/crates/db/src/lib.rs`

The pool was hard-coded to 5 connections. A pool connection is held for
milliseconds per query rather than for the life of a stream, so the number is
about how many queries may overlap: one turn's history read, another turn's
append, the sidebar catalog read and a queue poll are already four. The default
is now 16, overridable with `DB_MAX_CONNECTIONS` (clamped to 1–256, malformed
values fall back to the default).

---

## 5. Verification

Front end (`frontend/`):

| Check | Result |
| --- | --- |
| `bun run typecheck` | clean |
| `bun run lint` | clean |
| `bun test` | 113 pass, 0 fail (26 files) |
| `bun run build` | production build succeeds |

Rust (`backend/`, `SQLX_OFFLINE=true`, `DATABASE_URL` from `backend/.env`):

| Check | Result |
| --- | --- |
| `cargo build` | succeeds |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --workspace` | all suites pass |

Note on the environment: the database integration suites initially failed with
`template database "template1" has a collation version mismatch` — a local
Postgres upgraded under an older cluster, unrelated to this work. After
`ALTER DATABASE template1 REFRESH COLLATION VERSION` (and the same for the
`postgres` and `llm` databases) every suite passed.

New tests:

- `lib/frame-coalescer.test.ts` — burst collapse, per-frame rescheduling,
  immediate flush, dispose.
- `lib/message-parser.test.ts` — incremental output equals a from-scratch parse
  at every prefix, stable node identity across growth, partial tags, maths
  placement and code literals, non-prefix replacement, streaming predicates.

---

## 6. Still worth doing, in order

1. **One thread hop per token, in the Python streaming loop**
   (`llm-engine/api/routes/inference.py`). Every token creates a task around
   `asyncio.to_thread(advance, generator)` and shields it. That is a thread-pool
   round trip, a task and a cancellation scope per token, on the event loop that
   also serves every other request. Run the generator on one dedicated worker
   thread and hand pieces to the loop through a small queue (or `anyio.to_thread`
   with a persistent worker), treating disconnect as "stop pulling". This is the
   largest remaining per-token cost in the service layer.
2. **Batch token frames on the Rust side**
   (`api/src/turn/relay.rs`). Every SSE event becomes one WebSocket text frame
   and one `JSON.parse` in the browser. Coalescing consecutive `token` frames
   every 8–16 ms would cut frame count by an order of magnitude at 60 tok/s —
   but only if the written-token count stays truthful, which today is derived
   from one event per token on the client. Batch together with a count field (or
   count on the engine side) rather than silently undercounting. Not done here
   because it is a wire-contract change.
3. **KaTeX is in every chat page's initial JavaScript.** `components/chat/math.tsx`
   imports it statically, and `RichMessage` is on the chat route. It is a large
   dependency for a feature most replies do not use. A lazy import would cut
   first-load bytes but swaps in after paint and reflows the paragraph — which
   the component's own comment deliberately avoids. Worth measuring against a
   real first-load budget before changing.
4. **No response compression.** Actix-web's `Compress` has no predicate and would
   sit in front of the SSE hops, so it was not added. JSON routes (`/chats`,
   `/models`, conversation history) are the ones that would benefit; enable it
   for those alone, or terminate compression at the reverse proxy, and leave the
   streaming paths untouched.
5. **`API_URL` is hard-coded to `http://localhost:8080`** (`frontend/lib/api.ts`).
   In a real deployment that is a cross-origin call: an extra preflight per
   non-simple request and no shared connection to the page's own origin. Same
   origin behind a proxy would remove both.
6. **`conversation::prepare` reloads the whole conversation for every tool
   round.** A turn that calls three tools reads its own history four times. The
   round already holds the messages it just appended; passing them forward would
   remove the repeat.
7. **The composer's frosted surface** (`bg-card/75 backdrop-blur-2xl
   backdrop-saturate-150`) is the most expensive paint in the application,
   because it blurs live scrolling content underneath. It is a deliberate design
   choice; if the composer ever feels heavy on low-end hardware, this is the one
   property to measure, not the layout.
8. **A long single paragraph still costs O(paragraph) per frame** in `Inked` for
   its 240-character live window and in the parser for its tail. Both are bounded
   and small; they only matter if replies routinely arrive as one enormous block.

### Corrections to earlier assumptions

- **No new index migration is needed for `messages`.** An earlier plan called for
  `(chat_id, position)` and `(chat_id, created_at)`. The table already carries
  `UNIQUE(chat_id, position)`, whose index serves ordered scans and
  `max(position)` for appends, and `20261017000000_cover_chat_token_totals.sql`
  replaced the plain `messages(chat_id)` index with `messages(chat_id) INCLUDE
  (tokens)` for the sidebar aggregate. Adding either would duplicate an existing
  index and slow writes for nothing.
- **Actix `Compress` has no predicate** in the pinned version, which is why
  compression is a recommendation and not a change.

---

## 7. Files changed

| File | Change |
| --- | --- |
| `frontend/lib/message-parser.ts` | resumable parse, identity-preserving path copies |
| `frontend/lib/math-parser.ts` | exports `splitText`, `MATH_LITERAL` |
| `frontend/lib/frame-coalescer.ts` | new: per-frame coalescing, flush and dispose |
| `frontend/lib/frame-coalescer.test.ts` | new |
| `frontend/lib/message-parser.test.ts` | new |
| `frontend/hooks/use-chat.ts` | stream state applied once per frame |
| `frontend/hooks/use-throttled-number.ts` | new: trailing-rate number |
| `frontend/hooks/use-models.ts` | shared cache, in-flight de-duplication |
| `frontend/hooks/use-model-names.ts` | shared cache, in-flight de-duplication |
| `frontend/hooks/use-jobs.ts` | skip identical poll payloads |
| `frontend/hooks/use-local-plan.ts` | cached snapshot |
| `frontend/components/chat/rich-message.tsx` | memoised top-level blocks |
| `frontend/components/chat/inked.tsx` | settled prefix, live tail |
| `frontend/components/chat/chat-composer.tsx` | memoised |
| `frontend/components/chat/chat-thread.tsx` | stable composer props, throttled figure |
| `frontend/components/chat/chat-position-rail.tsx` | memoised ticks |
| `frontend/components/chat/chats-provider.tsx` | memoised context value |
| `backend/crates/db/src/lib.rs` | pool size from `DB_MAX_CONNECTIONS` |

---

## 8. Limits of this audit

- No browser CPU profile or React DevTools trace was captured. The render-side
  claims rest on node identity, which is measured directly (§4.3), and on the
  documented contract of `React.memo`; the parse-side claims are measured end to
  end.
- The parse harness uses a uniform synthetic reply. It reproduces the shape of
  streaming (many small appends) but not the distribution of real answers.
- The first-paint cost of the chat route was not measured against a budget, so
  the KaTeX item (§6.3) is a hypothesis with a known trade-off, not a finding.
