# Frontend — Next.js UI

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
│   ├── layout.tsx   Root layout: ThemeProvider → providers
│   ├── (app)/       The signed-in frame: sidebar, header, scroll container
│   │   ├── admin/       the admin panel — the queue, and the way to the rest
│   │   ├── chat/[id]/ · conversations/ · models/ · usage/ · profile/
│   │   ├── jobs/        [id] and new/{train,benchmark}; the bare path redirects
│   │   ├── benchmark/ · tokenizers/
│   │   └── markup/ · throbber/   fixture pages, no data behind them
│   ├── (auth)/      login, signup — outside the signed-in chrome
│   └── docs/        the markdown in docs/, rendered
├── components/
│   ├── ui/          shadcn/ui primitives (button, input, card, sidebar, etc.)
│   ├── admin/
│   ├── auth/
│   ├── chat/
│   ├── conversations/
│   ├── jobs/
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

## The docs site (`/docs`)

The markdown in [`docs/`](../) is rendered by this module at
[`app/docs/[[...slug]]/page.tsx`](../../frontend/app/docs) — public, outside the
`(app)` group, so reading the docs needs no account and does not depend on the
signed-in chrome.

- [`lib/docs.ts`](../../frontend/lib/docs.ts) walks `../docs` and maps files to
  routes: `docs/README.md` → `/docs`, `docs/backend/README.md` → `/docs/backend`,
  `docs/backend/schema.md` → `/docs/backend/schema`. A directory with no
  `README.md` gets a generated index of its children instead of a 404.
- **Build time only.** `generateStaticParams` prerenders every page, so the
  filesystem outside this module is never touched by a running server. The
  consequence: a docs-only commit needs a frontend rebuild before it shows on
  the deployed site, and any container build must include `../docs` in its
  context.
- **Relative links are rewritten** to routes, so the same markdown reads
  correctly on the site and in an editor. A link that resolves to nothing
  **fails the build** — deliberately; that is what keeps the docs from drifting
  away from the tree they describe.
- `AGENTS.md` at the repo root is rendered too, at `/docs/agents`, via
  `EXTERNAL_PAGES` in `lib/docs.ts`. Add to that map to publish another file
  from outside `docs/`.

- **Prose styling is `.doc-prose` in [`app/globals.css`](../../frontend/app/globals.css)**,
  hand-set against the app's own tokens rather than `@tailwindcss/typography` —
  the plugin ships its own type scale and greys, and a page set in them reads
  like a different product. Headings take the display serif, body copy is capped
  at 68 characters, and tables and code are allowed to break that measure.
- **HTML comments are stripped** before rendering. `<!-- TODO: … -->` is
  invisible in every other markdown viewer and stays invisible here.

Adding a doc requires no frontend change: write the markdown, and the page and
its sidebar entry appear on the next build.

## Colour

The palette is Kanagawa (Dragon in dark, a warm-paper light mode) and the token
values live in [`app/globals.css`](../../frontend/app/globals.css) with the
reasoning for each non-obvious one written on the token itself. Read those
comments before changing a value — several encode a measurement, not a taste.

Every pair is checked by

```sh
cd frontend && node scripts/audit-contrast.mjs
```

which prints the contrast of every token against every surface it lands on, in
both modes, against three floors: **4.5** for text (the app sets almost nothing
at "large" sizes, and its eyebrows are small and letterspaced), **3.0** for
graphical objects that carry meaning (chart marks, form-control boundaries), and
**1.8** for hairlines, which carry none and only have to survive a bright screen.
Exactly one row fails by design — `--tier-low`, for the reason written on it.

If you change a token, run it, and keep the hue: the fixes it prompts should be
lightness moves, which is what keeps the scheme recognisably Kanagawa.

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

### Failure states

Two components in [`components/ui/error-state.tsx`](../../frontend/components/ui/error-state.tsx),
split by who can act on the failure. Never hand-roll a third.

- **`FormError`** — a submission was rejected. Sits against the control that
  produced it, `role="alert"`, and is the one place red earns its keep.
- **`LoadError`** — a section could not fetch what it needed. Not set in red
  (the reader did not cause it), takes the footprint of the content it replaces
  so a failed section does not collapse the page, and takes an `onRetry` when
  the caller can refetch.

### Selects

Use `components/ui/select.tsx` — never a native `<select>`. Two things about
the Base UI primitive underneath, both learned the hard way:

- `<SelectValue />` on its own renders the **raw value**. Give it a render
  function that maps the value to its label, or the trigger will read `0` while
  the list it opens reads `Layer 0`.
- An option meaning "nothing chosen" needs a sentinel value of its own; the
  empty string is how a Select says *no selection*. Map it back at the edge
  (see `LEARN_WITH_RUN` in `components/training/params-rail.tsx`).

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

### Settings

`lib/settings.ts` is the single source of truth. One entry in `SETTINGS` (plus a
field on the `Settings` type) gives you a control in the ⌘K modal and, for a
toggle, a slash command — nothing else needs touching.

- An entry names a **`group`**, and `GROUPS` declares the bands in the order they
  appear. That is the only placement decision adding a setting requires.
- **A blurb belongs to a group, not a row.** `GROUPS` carries the optional
  sentence — which is how the three decoding knobs say the one thing worth
  saying about them, that they compose in order. Three rows each explaining
  themselves in two lines is what made the modal read as documentation with
  controls embedded in it.
- Theme is deliberately **not** a `SETTINGS` entry: next-themes owns and
  persists it (the no-flash script reads it before React exists), so declaring
  it here would give the app two answers to what the theme is. It gets a
  hand-written row in the `appearance` band that reads and writes the real one.

### Slash commands & fuzzy finder

Typing a leading `/` in the composer brings up the floating fuzzy finder popover
(`components/chat/slash-command-menu.tsx`, driven by `lib/commands.ts`).

- **Stage 1 (Command Discovery)**: Fuzzy-matches command names, aliases, and
  descriptions (e.g. `/model`, `/thinking`, `/show-thinking`, `/temperature`,
  `/top-k`, `/top-p`, `/system-prompt`, `/notify`, `/theme`, `/settings`,
  `/help`, `/clear`).
- **Stage 2 (Argument Selection & Range Guides)**:
  - `/model <name>`: Fuzzy-filters and selects available models.
  - `/thinking <budget>`: Sets the thinking budget (`auto | low | med | high`).
  - `/show-thinking` / `/notify`: Toggles reasoning visibility and notifications.
  - `/temperature`, `/top-k`, `/top-p`: Displays allowed ranges, current values,
    and clickable preset buttons.
  - `/system-prompt <text>`: Previews current prompt and offers templates or `clear`.
  - `/theme <light | dark | system>`: Switches visual theme.
- **Keyboard navigation**: Seamless `↑` / `↓` navigation, `Tab` autocomplete,
  `Enter` execution / selection, and `Esc` dismissal. Standalone commands execute
  locally with toast feedback and are never dispatched over the chat socket.

### The admin panel

Everything behind the admin flag is reached through **one row** in the sidebar
footer, beside Settings — not a labelled nav group. The row is a mode of the app
you have access to, which is what the footer holds; the content above it is the
subject you are reading.

- `/admin` is the panel: `JobsView` as the hero, `AdminDirectory` below it.
  **Add an admin destination to the directory, not to the sidebar** — a
  directory has room for a sentence saying what a page is, and a nav row does
  not. The two bands are what puts work on the GPU (`WORKBENCH`) and what only
  renders a fixture (`INTERFACE`).
- **`NavJobs` stays in the sidebar content** and is not navigation: it is the
  ambient "the GPU is busy" signal, a window onto the panel rather than a way
  in. `NavAdmin`'s `QueueTell` is the same fact when the sidebar is collapsed to
  icons and that group is hidden.
- Admin-ness is a client fact (`useAuth().isAdmin`, read from the stored access
  level), so these are hidden rather than guarded. **The endpoints are the
  boundary** — the engine's queue routes are admin-only, and `JobsProvider` is
  gated on the same flag so a normal session does not spend its life collecting
  403s.

### The jobs queue

Everything that uses the GPU — training and benchmarks — is a run in the
engine's queue. See `lib/jobs.ts` (types + REST) and `lib/jobs-socket.ts` (one
socket per run, relaying the engine's own frames).

A `tokenize` run is the exception that proves the line: it never touches the
device. It is queued anyway because it is minutes of CPU that would fight a
running train for it, and because the queue already has the frames, the cancel
and the job page it wants. Composed at `/tokenizers`, beside the list of what has
been built — a vocabulary is picked far more often than it is built.

- **The board lives at `/admin`**, as the hero of the panel — `JobsView` is
  mounted there, and `/jobs` is a redirect kept only so bookmarked URLs work.
  `/jobs/{id}` and `/jobs/new/*` are unchanged.
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

`components/ui/throbber.tsx` is a **figure**, not an icon: a hexagonal field of
nineteen dots with a **tree grown across it**. Ink leaves the centre, runs along
lattice edges and branches; each dot it reaches lights and swells while the rest
of the field goes on breathing. It is one fixed footprint everywhere,
`THROBBER_W` × `THROBBER_H` (about 27×25), and callers cannot resize it:
`className` is for margin, placement and colour only. That is deliberate — it
used to be sized per call site, and the result was a different figure in every
row, none of which looked like the same object.

**The tree is grown, not drawn.** A fresh one is grown by random walk at the
start of every pass, so the mark is never the same twice. What is fixed is the
*character* of the growth, and that is what carries the state — see `GROWTH`:
`limbs` (directions leaving the root), `branch` (how far an inner dot splits)
and `depth` (how far ink travels). One long wandering route is a token being
sampled; a broad shallow bush is several paths held open; the full spanning net
is every parameter moving at once.

Rules governing changes to it:

- **Geometry is built once at mount.** Every lattice edge is rendered up front
  and a tree is merely a choice of which are visible this pass, so a frame
  writes opacities and one dash length and never path data. An earlier design
  rebuilt paths per frame, and the browser re-parsing those was the whole reason
  it ran differently on different machines.
- **The phase is integrated, never read off the clock.** Each frame adds
  `rate × dt`, and a mode change blends the *rate*, so the figure changes pace
  rather than jumping.
- **Growing allocates, so it happens on a pass boundary** — about once a second
  — and never inside a frame. The PRNG is seeded per instance and per pass, so a
  tree is re-growable rather than merely random.
- **A state is three growth numbers plus an advance rule**, not a bespoke
  animation. Add a state by adding entries to `GROWTH`, `RATES` and `TINT`, and
  a branch to the advance `switch` in `draw` only if the new state genuinely
  cannot be told by how its tree grows.
- Unlike the rail it replaced, this figure **re-starts on purpose**: a state
  change grows a new tree immediately rather than waiting out the pass in
  progress, because repeated growth is its whole language.

The `/throbber` page shows every state side by side. Earlier candidates — a
rule, a stanza of set text, a comb of stems, arcs over a baseline, a folded
line, a causal-mask triangle, orbiting rings, a square lattice of cells — are
recorded with their reasons in the git history of `components/throbber/`; the
short version is that a figure with no silhouette reads as a picture of an axis,
and a figure with one parameter cannot tell eight states apart.

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
