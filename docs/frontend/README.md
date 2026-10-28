# Frontend — Next.js UI

The frontend owns interaction, presentation and browser state. Requests go through
Rust's HTTP/WebSocket routes; the browser never calls a Python service directly.
Read the complete [repository conventions](../conventions.md) and this map before
editing, then follow the topic for the contract being changed.

## Stack and entry points

Next.js 16 App Router, React 19, strict TypeScript and Bun. Styling uses Tailwind
CSS v4 and the accessible primitives in `components/ui/`. Tiptap owns rich input;
KaTeX typesets maths; Recharts renders telemetry.

| Location | Responsibility |
| --- | --- |
| [`app/`](../../frontend/app) | Server-first routes, layouts and fixture pages |
| [`components/`](../../frontend/components) | Domain presentation composed from shared UI primitives |
| [`hooks/`](../../frontend/hooks) | React state, subscriptions and browser-capability coordination |
| [`lib/`](../../frontend/lib) | Domain transforms, HTTP/socket adapters and explicit resource owners |
| [`app/globals.css`](../../frontend/app/globals.css) | Theme roles, sizing, shared surfaces and motion |
| [`gpu/`](../../frontend/gpu) | Optional live-throbber renderer |
| [`public/`](../../frontend/public) | Static assets and generated renderer bindings |

The signed-in frame is `app/(app)`: sidebar, scroll container and content. Desktop
has no additional header; mobile has a compact navigation opener. `/login` and
`/signup` live outside it. `/docs` is public.
`app/(app)/admin` contains the queue, workbench directory and history. Executions
and composers live under `/train`, `/bench`, `/gyms` and `/tokenizers`; old
`/jobs` URLs redirect with their query intact. [Routing](routing.md) maps canonical
pages, compatibility redirects and the distinct resource/execution identities.

## Development contracts

[Development](development.md) covers component boundaries, hooks, authentication
context, data fetching, errors, routing and primitives. Named functions, hooks and
components have brief JSDoc explaining purpose, inputs and outputs. Implementation
bodies contain no comments. Keep calculations and event reduction separate from
network/storage effects, and give sockets, timers and object URLs explicit owners.

[Notebook editing](editing.md) defines shared page-native fields, Save/Cancel,
tab-local draft recovery, conversation renaming, mobile navigation and confirmed
resource deletion. [Design](design.md) defines the active composition, copy, list, field, colour,
texture and boundary rules; read it before changing presentation.

Pages remain server components until browser interaction requires a client
boundary. Authenticated browser data loads through domain hooks and centralized
`lib/` adapters: browser-local bearer tokens are unavailable to server components.
Use strict input/result types and preserve interpreted wire contracts deliberately.

The [docs site](docs-site.md) maps Markdown files to statically generated routes,
rewrites relative links, and fails the build on unresolved links. Adding a sibling
topic here adds a route automatically; deployed docs require a frontend rebuild.

## Chat and composer

| Topic | Contracts |
| --- | --- |
| [Streaming and loading](chat-streaming.md) | Token deltas, multi-round completion, interjections and bottom-following |
| [History prefetch](chat-loading.md) | Cache sharing, freshness, session isolation and picker readiness |
| [Message presentation](chat-messages.md) | Stored traces, attachments, re-asking, maths and streamed words |
| [Composer](composer.md) | Uploads, dictation, context meter, pruning, choice persistence and commands |
| [Plans](composer-plan.md) | Local plans, revision-aware edits, ordering and fixture behavior |
| [Settings and profile](settings.md) | Dialog sections, account preferences and persistence |
| [Notifications](notifications.md) | Account-local history, toast lifetimes, dismissal and keyboard access |
| [Private projects](projects.md) | Project navigation, notes, conversation membership and composer/session ownership |

A token frame is a decoded delta. Concatenate it verbatim; insert no whitespace.
Only final `done` ends a turn. Tool results and confirmed interjections fold into
that turn's trace. Unconfirmed steering drains FIFO into ordinary turns after
completion, error or Stop. Interjections remain text-only.

The refactored chat path has explicit responsibility boundaries:

- [`lib/chat-messages.ts`](../../frontend/lib/chat-messages.ts) folds chronological
  stored rows into visible turns, including tool results and interjections.
- [`lib/chat-turn.ts`](../../frontend/lib/chat-turn.ts) purely reduces live events,
  counts arriving tokens once, and reconciles live-only tallies with saved history.
- [`lib/chat-session.ts`](../../frontend/lib/chat-session.ts) owns one chat's socket,
  steering queue and terminal reconciliation. Turn identity guards delayed callbacks.
- [`hooks/use-chat.ts`](../../frontend/hooks/use-chat.ts) coordinates cached history,
  context floors, feedback and the session's React subscription.

Navigating away disposes the session, closes its socket and ignores delayed socket
or history results. Stop settles the active send before draining queued steering;
navigation discards that local queue instead of sending into a chat being left.
Send-time preferences are read when each ordinary turn starts, while a queued
message retains the model and thinking budget chosen when it was queued.

`ChatComposer` renders and coordinates distinct editor concerns:

- [`hooks/use-composer-editor.ts`](../../frontend/hooks/use-composer-editor.ts)
  owns Tiptap setup, Enter/list/code behavior, literal drafts and page typing.
- [`hooks/use-composer-commands.ts`](../../frontend/hooks/use-composer-commands.ts)
  owns command discovery, menu selection and local execution.
- [`hooks/use-composer-dictation.ts`](../../frontend/hooks/use-composer-dictation.ts)
  owns the mapped voice range and transcript replacement.
- [`hooks/use-composer-attachments.ts`](../../frontend/hooks/use-composer-attachments.ts)
  subscribes to the upload draft in
  [`lib/composer-attachments.ts`](../../frontend/lib/composer-attachments.ts).
- [`lib/composer-editor.ts`](../../frontend/lib/composer-editor.ts) contains code
  indentation and the non-inclusive interim transcript mark.

Uploads start when picked. Sending releases local previews while retaining claimed
server rows; remove and `/clear` discard unsent rows. Removing a file while it
uploads also discards its eventual row once upload succeeds. Failed uploads remain
visible. Both Enter and the Send button wait for in-flight uploads.

## Jobs, gyms and models

[Modular architecture foundations](architectures.md) maps shared runtime decoders,
identity-scoped reads, neutral source editors and declared parameter controls.
[Administrator architecture drafts](architecture-authoring.md) maps connected
catalog/copy/editing, validation/acceptance and frozen-source inspection pages.
Their real browser/service evidence and final gate status are tracked in the
[feature acceptance map](../llm-engine/modular-architecture-acceptance.md).

[Data processing](data-processing.md) covers processor directory/creation pages,
source editing, saved/active versions and the dataset save/repreview loop.

[Jobs and administration](jobs.md) documents the queue, training composers,
follow-ups, diagnostics, checkpoints and corpus-build figures.
[Gyms](gyms.md) distinguishes browser-driven evaluation from queued RL practice,
including episode transcripts and their socket protocol.
[Models](models.md) covers architecture fallback, lineage and attention viewing.
[Tokenizers](tokenizers.md) covers vocabulary detail/editing and page-native builds.
[Ownership and recovery](recovery.md) records the current sprint boundaries and
compatibility rules for older benchmark reports.

[`hooks/use-job.ts`](../../frontend/hooks/use-job.ts) owns the selected job's REST
row and socket lifetime. [`lib/job-telemetry.ts`](../../frontend/lib/job-telemetry.ts)
contains immutable snapshot/event reduction and chart retention policies. The row
is authoritative; live telemetry enriches it. Terminal `done`, `error` and
`cancelled` frames refresh it. Switching jobs resets telemetry and ignores stale
responses from the prior resource.

Engine telemetry's legacy payload assertions are isolated in `jobTelemetryEvent`.
They are a compatibility seam, not a complete runtime schema: RL diagnostics are
sparse and must not be rejected or filled with zero by a pretraining-only validator.
Optional measurements remain absent.

## Appearance and account pages

[Design direction](design.md) summarizes Slate's editorial product and visual
language. [Appearance](appearance.md) defines its current theme roles, sizing,
surfaces, artwork and motion details.

[Authentication](authentication.md) covers login/signup composition,
decorative pointer effects and field accessibility.
[Live throbber](live-throbber.md) covers the optional renderer, motion, token
contract, fallback, showcase and build.

Use existing theme roles and accessible controls. Decorative assets ignore pointer
events and assistive technology. Reduced motion preserves state information while
removing nonessential movement. `/markup`, `/throbber` and `/colors` render the real
components or tokens so previews continue to describe the application.

Notifications use one browser-local, account-scoped owner for retained history
and glass toasts; the sidebar bell and Ctrl+Alt+N open a settings-style modal.
Opening does not dismiss retained items. See [Notifications](notifications.md).

Settings writes use a single debounced owner in
[`lib/settings-writer.ts`](../../frontend/lib/settings-writer.ts). Preference
updates calculate the next value before scheduling persistence; React state
updaters perform no writes or timer effects. Model/budget picks keep their separate
immediate account resource and readiness contract. Settings values and pending writes
belong to the actual session token through `useSessionToken`; unmatched identities
show defaults and failed reads cannot authorize overwriting unknown account values.
See [Settings and profile](settings.md) for replacement-session and failure behavior.

## Verification

Run from `frontend/`, using the existing installed dependencies:

```sh
bun run typecheck
bun run lint
bun test
bun run build
```

Focused tests exercise byte-exact token concatenation, multi-round history/live
agreement, queued steering, cancelled/disposed callback races, late upload cleanup,
chart retention and debounced writes. Browser interactions still need a browser
check; unit checks alone do not validate focus, drag/drop, dictation APIs or visual
layout. Cross-module backend integration tests are separate.

For theme-token changes also run `node scripts/audit-contrast.mjs` and
`node scripts/audit-theme.mjs`. For renderer changes use `bun run build:throbber`
and `bun run check:throbber`; the normal build checks the renderer artifact.
