# CLAUDE.md

> **Role:** See [`AGENTS.md`](../AGENTS.md) at the project root for the agent
> role system. Unless told otherwise, you are the **Fixer**.

Guidance for building and fixing the frontend half of the project.

## Tech stack

- **Framework**: Next.js 16 (App Router), React 19
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS v4 via `@tailwindcss/postcss`
- **UI primitives**: shadcn/ui (`components/ui/`)
- **Rich text**: Tiptap (prosemirror-based)
- **Charts**: Recharts
- **Package manager**: Bun (see `bun.lock`)

## Directory structure

```
frontend/
├── app/             App Router pages and layouts (server components by default)
│   ├── layout.tsx   Root layout: ThemeProvider → Sidebar layout
│   ├── page.tsx     Home page (redirects to /chat)
│   ├── chat/[id]/
│   ├── conversations/
│   ├── login/
│   ├── signup/
│   ├── models/
│   ├── training/
│   ├── usage/
│   └── profile/
├── components/
│   ├── ui/          shadcn/ui primitives (button, input, card, sidebar, etc.)
│   ├── auth/
│   ├── chat/
│   ├── conversations/
│   ├── models/
│   ├── profile/
│   ├── sidebar/
│   ├── training/
│   └── usage/
├── hooks/           Custom React hooks (use-chat, use-models, use-auth, etc.)
├── lib/             Pure utilities, API helpers, socket clients
│   ├── api.ts           HTTP helpers, auth token management
│   ├── chat-socket.ts   one socket per chat, streaming a reply
│   ├── jobs.ts          the queue: types and REST
│   ├── jobs-socket.ts   one socket per run, relaying its frames
│   ├── benchmark.ts     scores and the reports behind them
│   ├── message-parser.ts
│   ├── user-parser.ts
│   └── utils.ts          cn() and other small helpers
├── public/
└── next.config.ts
```

## Conventions

### Components

- **Server-first**: App Router pages are async server components by default. Only add `'use client'` when hooks, browser APIs, or event handlers are required.
- **One component per file**, named after the file. Exported as a named function, not as `default`.
- **Props interface**: Define a named `interface` or `type` in the same file, exported, prefixed with the component name.
  ```tsx
  export interface ChatThreadProps { chatId: string }
  export function ChatThread({ chatId }: ChatThreadProps) { ... }
  ```
- **Business components** live in `components/<domain>/` and compose UI primitives from `components/ui/`.
- **Don't re-invent UI primitives**: Use the shadcn/ui components from `components/ui/` for buttons, inputs, dialogs, cards, etc. If you need a variant the existing component doesn't support, add a prop to it — don't hand-roll a replacement.

### Hooks

- Domain hooks live in `hooks/` and are named `use-<domain>.ts`. Each file exports one primary hook.
- Hooks call `lib/api.ts` helpers for HTTP, and `lib/` socket functions for WebSocket. They do not inline `fetch()` or `new WebSocket()`.
- **State is local to hooks.** Components consume state and callbacks from hooks; they don't own `useState` for domain data.

### Data fetching

- **Server pages** fetch initial data in the async component body. Use `fetch` with `authHeaders()` from `lib/api.ts`.
- **Client components** fetch via hooks (e.g. `useChat`, `useModels`) that use `useEffect` + `fetch` internally, or through the socket layer for streaming.
- Mutations (create chat, send message, etc.) go through the socket or through POST/PUT/DELETE via `fetch` with `authHeaders`.

### Routing

- Page folders match the route: `app/chat/[id]/page.tsx` → `/chat/:id`.
- Use Next.js `<Link>` for navigation, not raw `<a>` tags (avoids full page reloads).

### Styling

- Tailwind utility classes in JSX. No separate `.module.css` files.
- Use `cn()` from `@/lib/utils` for conditional class merging.
- Follow the existing design: Geist font (sans), the shadcn/ui color tokens, the sidebar layout.

### Where the data lives

The backend owns people, sessions and presets. Everything else — models,
conversations, runs, scores, usage — belongs to the Python engine, and the
backend forwards to it. So a route here is `/chats`, `/models`, `/runs`,
`/benchmarks`, `/usage`, and the shape that comes back is the engine's.

Two names are adapted at the wire rather than renamed through every component,
each in one place: the engine's `run_id` is this side's `id_job` (`lib/jobs.ts`),
and its `benchmark_id` is `id_run` (`lib/benchmark.ts`). Nothing else is
reshaped — a field the engine adds arrives without a change here.

### Stream protocol (chat)

One socket per chat, opened at `/ws/chats/{id}`:

```
client → { token }                            (auth frame)
server → { type: "ready" }
client → { model_name, text, system?, budget?,
           temperature?, top_k?, top_p? }
server → { type: "token", token }             (repeated)
server → { type: "done", message_id, content, tokens, usage }
server → { type: "error", error }
```

The chat exists before the socket opens — its id is in the URL — so there is no
"which conversation is this" frame and no id to wait for coming back. The reply
is written when the last token lands, so closing the tab loses it rather than
storing half of one.

`budget` is the composer's thinking-budget dropdown
(`components/chat/budget-selector.tsx`) — `"low" | "med" | "high"`, matching the
labels the SFT corpus is tagged with, since the string sent becomes the marker
the model was trained under. **"Auto" is the absence of a budget, not a fourth
value**: it is omitted from the frame entirely, so the engine renders no marker
and the model is prompted exactly as it always was. Unlike `systemPrompt` it is
*not* a setting — how hard you want the model to think is a property of the
question, so it lives as composer state and rides through `PendingMessage` when
a new chat navigates.

`temperature` / `top_k` / `top_p` are the decoding knobs, and unlike `budget`
they **are** settings (`lib/settings.ts`, rendered as sliders in the ⌘K modal) —
how random you want decoding to be is a standing preference, not a property of
one question. Each is omitted individually when unset or non-finite, and the
engine then falls back to its own defaults.

### The jobs queue

Everything that uses the GPU — training and benchmarks — is a run in the
engine's queue. See `lib/jobs.ts` (types + REST) and `lib/jobs-socket.ts` (one
socket per run, relaying the engine's own frames).

A `tokenize` run is the exception that proves the line: it never touches the
device. It is queued anyway because it is minutes of CPU that would fight a
running train for it, and because the queue already has the frames, the cancel
and the job page it wants. Composed at `/tokenizers`, beside the list of what has
been built — a vocabulary is picked far more often than it is built.

- **The board polls.** It moves when a run is queued, starts or ends — rare
  enough that a 3s refetch of `GET /runs` says the same thing a nudge would,
  without a second live connection to keep alive across sleeps and restarts.
- `hooks/use-jobs.ts` owns the board; `hooks/use-job.ts` owns one job — its
  durable row plus whatever it is broadcasting. **Panels render from the row
  first and treat live state as an enrichment**, because a queued or finished
  job has no frames at all.
- Composers live at `/jobs/new/*` and are *only* forms: submitting queues a run
  and hands off to `/jobs/{id}`, where the telemetry lives. `?from=<job id>`
  seeds a composer from a previous run's stored config.
- **A chat turn is not a run.** It is answered directly, on the socket that
  asked for it — an interactive turn behind a three-hour training run is a chat
  nobody can use.
- **Queued is a state with a body, not an empty version of the finished one.**
  Every kind renders `JobQueuedPanel` while it waits: the config recap, its
  place in the line, and what the run will do once it starts.
- `ConfigRecap` is shared and switches on `job.kind`. A new kind adds its rows
  there rather than growing a recap of its own.
- A **chain** is a flat list of steps, each running against the model the one
  before it produced. A follow-up waits rather than queueing: its model does not
  exist yet, so a queue position would be a lie about when it runs.
- `/benchmark` is history and analysis only — it no longer starts runs.
- The throbber's per-kind modes (`training` / `benchmarking` / `generating`) are
  how a running job says what it is. Give **one** instance a changing `mode` so
  it morphs between them as the queue advances rather than cutting. `NowBand` is
  that instance.

### Throbber or dot

`components/ui/throbber.tsx` is a **figure**, not an icon: a **rail** — a resting
track with a brighter segment moving on it. Every state is that same rail, and what
tells them apart is where the segment is and how much of the rail it covers, with
colour as the second signal. It is one fixed footprint everywhere, `THROBBER_W` ×
`THROBBER_H` (84×22), and callers cannot resize it: `className` is for margin,
placement and colour only. That is deliberate. It used to be sized per call site, and
the result was a different figure in every row, none of which looked like the same
object.

Three rules govern changes to it:

- **Geometry is built once at mount.** A frame writes two coordinates, two opacities
  and (in one state) a dash — never path data. An earlier design rebuilt five
  60-segment paths per frame, and the browser re-parsing those was the whole reason it
  ran differently on different machines.
- **The phase is integrated, never read off the clock.** Each frame adds `rate × dt`,
  so a mode change alters the speed the figure is moving at and it carries straight
  through. A state change can never jump it.
- **A segment arrives from outside the rail and leaves outside it.** It travels past
  both ends and the rail clips it, so a full-size pill slides in behind the track's
  rounded end rather than growing out of it. Anything placed past the ends is hidden,
  which is why the states may return `from`/`to` outside 0–1.
- **A state is a function of its phase returning four numbers** (`Frame`), so any two
  mix by weight — give one instance a changing `mode` and it morphs instead of
  cutting. Mixing the *output* rather than a vector of shared parameters is what keeps
  each state to three lines of arithmetic with a name. Add a state by adding a
  function to `STATES` and a rate to `RATES`; don't add a field to `Frame` unless the
  renderer genuinely cannot draw the state without it.

So use it only where a figure belongs — somewhere the waiting *is* the subject, with
room for it:

- the chat thread (loading, and the thinking/generating status row),
- the training hero corner (`TrainingHero`),
- the queue's now-band,
- the benchmark empty state.

Everywhere else — beside a line of text, inside a button, in a sidebar row — use
`components/ui/running-dot.tsx`. Those places need the *fact* of activity, not
something to look at. Don't reach for a third indicator: a lucide `Loader2` on a
button is the thing `RunningDot` replaced.

### Scripts

```sh
bun run dev         # dev server
bun run build       # production build
bun run lint        # ESLint
bun run format      # Prettier
bun run typecheck   # tsc --noEmit
```
