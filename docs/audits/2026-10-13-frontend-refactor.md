# Frontend refactor — implementation and independent review

2026-10-13 · Implemented by GPT-6.1 Sol; reviewed by the parent agent.

**Verdict: keep this refactor.** The chat, composer, job telemetry and settings
paths have substantially clearer responsibility boundaries. Pure transformations
can be checked without rendering React, and sockets, uploads and timers have
explicit owners. The strongest improvements are reviewable behavior contracts and
race fixes, together with smaller integration files.

This is a focused refactor of the major stateful frontend paths. Other large
components and earlier audit findings remain. There is no defensible single
percentage for “code quality,” and no runtime speedup was measured.

All comparisons use the task-start working tree, including the user's existing
uncommitted work, rather than git HEAD. No commit, push, deployment, Docker run,
backend change or dependency addition was performed.

## Rules adopted

- [AGENTS.md](../../AGENTS.md) now requires reading the entire actual coding-rules
  file and affected module map before edits. An index, summary or memory is insufficient.
- [Coding conventions](../conventions.md) now contains C1–C12 across Rust, Python
  and TypeScript, language supplements, and the historical R1–R21 compatibility map.
- Named functions get brief purpose/input/output documentation at their declaration.
  Refactored implementations express their flow through names and structure;
  useful invariants move to function documentation and step narration is removed.
- The approved proposal is retained as a clearly marked historical review record.
- The frontend module map shrank from 1,755 to 162 lines. Its contracts and detailed
  topics remain accessible through sibling documents.

## Measured changes

| Integration file | Before, lines | After, lines | Change |
| --- | ---: | ---: | --- |
| `ChatComposer file` | 1,212 | 444 | 63.4% smaller |
| `useChat file` | 858 | 254 | 70.4% smaller |
| `useJob file` | 310 | 120 | 61.3% smaller |

| Main function | Before, body lines | After, body lines |
| --- | ---: | ---: |
| `ChatComposer` body | 977 | 360 |
| `useChat` body | 652 | 184 |
| `useJob` body | 249 | 101 |

Physical body lines include callbacks and JSX. Comment removal and extraction both
contribute to the reduction; these numbers do not independently measure complexity.
The implementation still exists in focused modules.

Across the entire handwritten TypeScript inventory, including tests, file count
changed from **305 to 323**, and physical lines from
**47,064 to 47,849**. Named function bodies over 60 lines changed
from **129 to 130**. The global count has not improved.
There are more explicit types, contracts and tests, and several new coordination
hooks still exceed the ordinary-function guideline. This is why the verdict rests
on ownership and testability, rather than a blanket claim that the frontend is now small.

## What became easier to change

| Concern | Previous coupling | Current boundary and practical benefit |
| --- | --- | --- |
| Stored chat history | Folding and tool parsing lived inside the streaming hook | `lib/chat-messages.ts` owns history-to-turn transformation; it can be changed and tested independently of a socket or React |
| Live chat turns | Token counting, tool phases and message-tail updates were nested in one send callback | `lib/chat-turn.ts` reduces typed events without effects; the same multi-round/token contracts have direct tests |
| Socket and steering lifetime | Mutable refs, completion callbacks and queue draining were interleaved with rendering state | `ChatSession` owns one active turn and FIFO queue; identity checks reject delayed callbacks after cancel/navigation |
| Composer | Editor keys, command menus, voice positions, uploads and rendering lived in one component | Separate editor, command, dictation and attachment hooks; `ChatComposer` integrates their public contracts |
| Attachments | URL revocation and DELETE ran in replayable React state updaters | `ComposerAttachments` owns effects and upload identity; late remove/clear cleanup is deterministic |
| Job telemetry | Twelve independent setters and a large socket switch combined transformation with connection handling | `job-telemetry.ts` owns immutable reductions/retention; `useJob` owns its selected resource and transport |
| Account preferences | Authentication boolean and updater-created timers governed persistence | Actual token identity owns loading and debounced writes; failed or stale reads cannot authorize settings overwrite |

The new owners are concrete classes/functions with explicit effects. They do not
introduce a generic state framework or speculative plugin system. More files mean
some additional navigation, but their names identify real responsibilities.

## Before and after: attachment removal

Before — actual task-start `chat-composer.tsx` declaration. Revoking a URL and
sending DELETE happen inside a React state updater:

```ts
const removeAttachment = React.useCallback((key: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.key === key)
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl)
      if (target?.attachmentId) {
        void discardAttachment(target.attachmentId).catch(() => {})
      }
      return prev.filter((a) => a.key !== key)
    })
  }, [])
```

After — the draft owner executes removal once, then publishes its state:

```ts
/** Removes a staged file and discards its current or eventual upload. */
remove = (key: string): void => {
  this.releaseItems(
    this.items.filter((item) => item.key === key),
    true
  )
  this.publish(this.items.filter((item) => item.key !== key))
}
```

The completion path now handles a file removed before its upload finished:

```ts
private complete(key: string, stored: Attachment): void {
    if (this.abandoned.delete(key)) {
      void this.effects.discard(stored.id).catch(() => {})
      return
    }
    this.publish(
      this.items.map((item) =>
        item.key === key
          ? { ...item, status: "ready", attachmentId: stored.id }
          : item
      )
    )
  }
```

Independent browser reproduction: remove a pending tile, then complete its mocked
upload. **Before: zero DELETEs for the eventual row. After: exactly one DELETE.**
Removing an already-ready file also produces one DELETE. All requests were mocked.

## Before and after: settings writes

Before — the state updater schedules persistence, while account loading is keyed
only by `isAuthenticated`:

```ts
const setSetting = React.useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: value }
        if (loaded.current) schedule(next)
        return next
      })
    },
    [schedule]
  )
```

After — mutation requires a successfully loaded owner whose token remains current,
and persistence runs outside the updater:

```ts
const setSetting = React.useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      const owner = ownerRef.current
      if (!owner?.active || !owner.loaded || getToken() !== owner.token) return
      const settings = { ...owner.settings, [key]: value }
      owner.settings = settings
      setAccount({ token: owner.token, settings })
      owner.writer.schedule(settings)
    },
    []
  )
```

The loading effect owns that token's writer, rejects stale results, and enables
writes only after a successful validated GET. Known preference fields are parsed
at the adapter boundary; writes are pinned to their owning token.

Independent browser reproduction:

| Sequence | Before | After |
| --- | --- | --- |
| Replace A with B; B's GET fails; change a preference | One PUT under B carrying A's settings | Zero PUTs; unknown B settings remain untouched |
| Replace with C; C's GET succeeds; change a preference | C's PUT still contains A's prompt | C's PUT contains C's own prompt |

My review sent the initial boolean-keyed provider back for correction. The final
browser result verifies the corrected implementation, rather than relying on the
smaller provider's appearance.

## Before and after: job frames

Before — an excerpt from the socket callback in `useJob`:

```ts
case "train": {
  const s = frame as unknown as TrainStats
  setTrain(s)
  if (Number.isFinite(s.loss)) {
    setLoss((prev) => {
      const base = prev.length >= MAX_LOSS_POINTS
        ? prev.filter((_, i) => i % 2 === 0)
        : prev
      return [...base, { epoch: s.epoch, chunk: s.chunk, loss: s.loss }]
    })
  }
}
```

After — the complete frame callback delegates interpretation/reduction and keeps
transport ownership and authoritative terminal refresh visible:

```ts
function onFrame(frame: Record<string, unknown>): void {
      if (!active) return
      const event = jobTelemetryEvent(frame)
      if (event) setTelemetry((previous) => reduceJobTelemetry(previous, event))
      if (
        frame.type === "done" ||
        frame.type === "error" ||
        frame.type === "cancelled"
      )
        void refresh()
    }
```

Retention, snapshot reconstruction and per-kind event handling live in small pure
functions with explicit contracts. The legacy wire assertions remain visible in
`jobTelemetryEvent`; this extraction does not pretend they are complete validation.

## Before and after: live chat updates

Before, `sendMessage` defined repeated append/fold helpers, each copying the message
array and modifying its last assistant row, alongside mutable token counters and
socket completion effects. The function body occupied 303 physical lines.

After, arrival ownership and transformation meet at one explicit seam:

```ts
turn.state = reduceChatTurn(turn.state, event)
this.effects.changeMessages({ type: "stream", message: turn.state.message })
```

`reduceChatTurn` returns a new state for tokens, tool calls/results, interjections
and round durations. `ChatSession` handles transport and terminal reconciliation.
This supports deterministic tests for stale history completion, cancellation,
FIFO steering and disposal without bringing up React or a backend.

## Intentional behavior corrections

1. Navigating away closes the chat socket, settles its send, and suppresses stale
   frame/history callbacks. Navigation discards local steering; Stop drains it.
2. Removing/clearing a pending upload discards its eventual server row once.
   Rapid file picks share one synchronous admission count.
3. Enter cannot send ordinary messages while uploads are pending. Standalone local
   commands still run before that gate, so `/clear` can abandon a pending upload.
4. Selecting another job resets its telemetry/loading state and rejects detached
   responses. Cancelled frames refresh authoritative status alongside done/error.
5. Settings writes and timers run outside React updaters. Actual session identity,
   successful-read authorization and token-pinned writes prevent the demonstrated
   cross-account settings contamination.
6. Feedback payload calculation no longer relies on mutating a closure from a
   React updater. More general feedback-write ordering remains follow-up work.

## Independent verification

| Check | Result |
| --- | --- |
| Task-start baseline | Typecheck, lint, 25 tests and production build passed |
| Final TypeScript | Passed |
| Final ESLint | Passed |
| Final test suite | 46 passed, 0 failed, across 12 files; 21 added tests |
| Production build | Passed, including renderer artifact check and static docs generation |
| History differential probe | 1,000 seeded scenarios agree with the original production transformation |
| Telemetry differential probe | 12,000 frame-by-frame comparisons agree with the original callback |
| Chart compaction probe | 1,200-frame sequence retains original results |
| Immutability | Differential probes verify input rows/states remain unchanged |
| Browser checks | Nine cases pass on the built frontend with mocked API responses; no page exceptions |
| Formatting and diff hygiene | Changed source formatted; `git diff --check` passes |

The nine browser cases cover hydration, Enter submission, draft clearing, local
plan/settings commands, ready attachment removal, late-upload cleanup and failed/
successful account replacement. No real backend records or messages were created.

The tests exercise state and failure contracts. They provide stronger evidence
than helper-layout assertions, while leaving clear limits: no real-model inference,
backend/PostgreSQL integration, actual microphone recognition, exhaustive browser
matrix, performance benchmark or complete visual regression suite was run. No Docker
was used. Baseline production verification used webpack because Turbopack rejects
the temporary snapshot's dependency symlink; the final real checkout uses Turbopack.

## Remaining weaknesses and my assessment

I would keep these changes. Core ownership is more explicit, the pure/event seams
are useful, and meaningful regressions are now testable. The refactor's value is
concentrated in the frontend's most stateful paths. It does not yet make every
frontend module comply with the new standards.

- **Large integration functions remain.** ChatComposer still has substantial JSX;
  useChat still coordinates history, feedback and pruning. Dedicated editor/command
  hooks also exceed the ordinary-function target. Future extraction should follow
  real variation or ownership, avoiding wrappers created solely for line quotas.
- **Other account providers retain earlier debt.** ChatsProvider and
  ComposerChoiceProvider still use boolean-keyed account loading. The settings
  correction does not establish complete account isolation across the application.
- **A stale 401 still expires the current browser session.** I reproduced this
  separately in the unchanged central `apiFetch`: start request A, replace its
  token with B, then return A's 401; B is cleared and the browser redirects to
  `/login`. Follow-up: expire only the session that authorized the failed request,
  with a replacement-token regression. This does not undo the demonstrated
  settings-value/write isolation, but remains a cross-cutting auth race.
- **Telemetry wire typing is incomplete.** Sparse RL diagnostics do not match the
  mandatory pretraining arrays in DiagPoint. Introduce a truthful wire union or
  coordinated schema; preserve absent measurements instead of filling them with zero.
- **Queued-job detail transitions still need reconciliation.** The hook preserves
  one-shot attachment behavior; the earlier audit's queued→running observation
  problem needs polling/re-attachment. Resetting telemetry is a separate correction.
- **Feedback/pruning still have older async races.** The streaming owner guards its
  own events; competing feedback writes and prune/navigation results need their own
  identity/order contracts.
- **Settings load failure has limited UX.** It safely blocks writes, but there is
  no new retry/error presentation. A successful session reload enables mutations.
- **Attachment cleanup remains best effort.** Unmount preserves the prior unsent
  row policy; pending upload disposal deserves its own explicit discard/retention
  contract. DELETE failures have no new user-facing recovery path.
- **Resource bounds need further work.** Steering remains unbounded; snapshot log
  retention and excess failed file tiles still warrant explicit limits. Synchronous
  WebSocket-constructor failures need deliberate settlement.
- **Comment/doc debt remains elsewhere.** Untouched functions, including bodies in
  type-import-only changed files, retain existing narration and missing JSDoc.

The next useful slices are account-provider/401 isolation, job status observation,
then the large command/training forms and telemetry wire contracts. This report
records those limits so the new rules guide the next change without hiding debt.

## Location of the implementation

The [frontend module map](../frontend/README.md) links all new responsibility
boundaries and their contracts. Source changes are confined to `frontend/`;
module documentation is under `docs/frontend/`. Rules/adoption records are in
AGENTS.md, docs/conventions.md and docs/audits. Changes remain uncommitted.
