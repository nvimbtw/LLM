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
│   ├── (app)/       The signed-in frame: sidebar and scroll container. No
│   │                header bar — see the layout for why.
│   │   ├── admin/       the admin panel — the queue, and the way to the rest
│   │   ├── chat/[id]/ · conversations/ · models/ · usage/ · profile/
│   │   ├── jobs/        [id] and new/{train,benchmark}; the bare path redirects
│   │   ├── benchmark/ · tokenizers/
│   │   ├── gyms/        the catalogue, [gym] to start a run, [gym]/runs/[id] live
│   │   └── markup/ · throbber/   fixture pages, no data behind them
│   ├── (auth)/      login, signup — outside the signed-in chrome
│   └── docs/        the markdown in docs/, rendered
├── components/
│   ├── ui/          shadcn/ui primitives (button, input, card, sidebar, etc.)
│   ├── admin/
│   ├── auth/
│   ├── chat/
│   ├── conversations/
│   ├── data/        the Hub, the datasets stored, the cleaning workbench
│   ├── gyms/        the run form, the scoreboard, one episode's transcript
│   ├── jobs/
│   ├── models/
│   ├── profile/
│   ├── sidebar/
│   ├── training/
│   └── usage/
├── hooks/           Custom React hooks (use-chat, use-models, use-auth, etc.)
│                 — and the browser-capability ones: use-mobile,
│                 use-reduced-motion, use-speech-recognition, use-audio-level
├── lib/             Pure utilities, API helpers, socket clients
│   ├── api.ts           HTTP helpers, auth token management
│   ├── chat-socket.ts   one socket per chat, streaming a reply
│   ├── jobs.ts          the queue: types and REST
│   ├── jobs-socket.ts   one socket per run, relaying its frames
│   ├── data.ts          the data engine: datasets, processors, previews
│   │                    — and the training rail's dataset list, since a run
│   │                    selects from here by id
│   ├── benchmark.ts     scores and the reports behind them
│   ├── gyms.ts          gyms, runs, episodes: types and REST
│   ├── gym-socket.ts    one socket per run, driving every episode in it
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

The palette is Kanagawa (Dragon in dark, a warm-paper light mode) — *our*
Kanagawa: derived from rebelot's and then moved where a screen full of small
text needed it moved. Upstream is lineage, not a source of truth; do not
"restore" a value to it.

It lives in [`app/globals.css`](../../frontend/app/globals.css) in three layers,
which exist so that what a colour **is** stays separate from what **uses** it:

| layer | tokens | holds |
| --- | --- | --- |
| 1. palette | `--kanagawa-dragon-*`, `--kanagawa-lotus-*` | the hexes, named in Kanagawa's vocabulary. Never referenced by a component. |
| 2. theme | `--background-light` / `--background-dark`, one pair per role | which colour each role takes in each mode, both modes side by side |
| 3. active mode | `--background`, … | `:root`/`.theme-light` resolve the light half, `.dark` the dark one. The only layer components see. |

So trying something is a one-line edit in layer 2 — point a role at a different
palette entry and both the app and the `/colors` fixture follow. A hex belongs
in layer 1 and nowhere else; a component reads layer 3 and nothing else. The
reasoning for a *value* is written on it in the palette, the reasoning for a
*role's choice* on it in the theme.

`.theme-light` is `:root`'s twin, declared in the same rule so the two cannot
drift: it lets a subtree render the light theme while the app is in the dark
one, which is what the `/colors` fixture uses to show both at once.

**Selected text** is a role like any other — `--selection` / `--selection-foreground`,
resolved in `::selection` in the base layer. Both modes take Kanagawa's own
`Visual` colour (waveBlue1 under Dragon, lotusViolet3 under Lotus), which is the
one role where upstream's answer is exactly the right one. The foreground is
pinned rather than inherited, so the ratio is a fact rather than a function of
whatever happens to be selected — which is also why selecting coloured text
drops its colour, the same as in an editor.

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

WCAG is the floor, not the whole test, and

```sh
cd frontend && node scripts/audit-theme.mjs
```

is the perceptual audit that sits on top of it: the OKLCH ladder of every token,
APCA Lc (vendored apca-w3) with the minimum font size each body pair supports,
and a chromatic-aberration check on every text pair. That check is why the body
ink is a red-violet *whisper* (#4f4f58) rather than the old full violet: a cool
text hue on a warm ground makes the lens hunt between two focal planes at every
letter edge, and the rule catches it as `energy ≥ 3.0` (hue gap × text chroma).
`--primary` links score high on that measure too and are exempt only because
they are short, underlined and spaced — the flag applies to body-carrying pairs.

Both read the stylesheet through
[`scripts/theme-tokens.mjs`](../../frontend/scripts/theme-tokens.mjs), which
resolves the three layers the way the browser does and hands them one flat
`{ role: "#rrggbb" }` map per mode — so an audit measures what ships, not what
a regex could find in one block.

If you change a token, run both, and keep the hue: the fixes they prompt should
be lightness moves, which is what keeps the scheme recognisably Kanagawa.

A palette change is judged against the old values on the `/colors` fixture
page — both themes, the token ladder, and the frozen pre-change snapshot with
its measured deltas.

## Sizing

The same two-layer split the colour tokens got, applied to everything with a
size. In `globals.css`:

1. **The scales** (`@theme`) — what a size *is*. The type steps `2xs`…`5xl`, each
   carrying its own line-height and letter-spacing; the radius steps derived
   from `--radius`; the 4px spacing unit.
2. **The roles** (`:root`, under `SIZING ROLES`) — what *uses* a size.
   `--type-eyebrow-tracking`, `--space-sheet`, `--radius-sheet`,
   `--elevation-overlay`. Components reference roles, never raw steps.

There is no third layer: a size is a size in both themes.

### The two composite utilities

Two devices appear so often, and drifted so far, that they are classes rather
than sets of tokens to remember:

- **`eyebrow`** — the small uppercase label over a section, a stat, a field.
  Sets size, weight, letter-spacing, word-spacing and the transform. Colour is
  *not* included: most take `text-muted-foreground`, the sidebar's take a
  sidebar token. Use `tag` for its mono cousin inside chat traces, which is
  sized in `em` because it sits in text that is itself scaled.
- **`figure`** — the number in a stat: display face, `font-variant-numeric:
  tabular-nums`, one weight. Deliberately sets **no font-size**; the call site
  picks the step (`figure text-2xl`) so the step's own tracking applies. Never
  add `tracking-tight` to a figure — the scale already tracks by size, and the
  constant overrides it.

Before this pass the eyebrow had 17 distinct spellings across 69 uses, with 7
different letter-spacings for one element; the figure had 10 sizes, half of them
fighting the scale with `tracking-tight`.

### Radius roles

`--radius-sheet`, `--radius-control`, `--radius-control-sm`, `--radius-marker`,
and — pinned off the scale — `--radius-composer` and `--radius-bubble`, which
follows it. Chat is the one place the app is allowed to be round, and those
values are absolute so that squaring the app's corners cannot reach into the
composer.

No component spells `rounded-lg`/`rounded-md` any more. The only bare radius
utilities left are `rounded-full` (pills, avatars), `rounded-none`, and three
micro values on chart bars and inline code where the role step would round them
into blobs.

## Elevation

Four levels on one ladder — `shadow-raised`, `shadow-sheet`,
`shadow-sheet-hover`, `shadow-overlay` — and one rule for deciding between the
middle two:

- **A surface you can act on is a sheet.** `bg-card`, an edge at `border/45`,
  and `shadow-sheet` — `ui/panel.tsx`, `ui/card.tsx` and the model tiles. A tile
  that is also a link adds `group-hover:-translate-y-px
  group-hover:shadow-sheet-hover` (and `motion-reduce:` opts out of the move).
- **A surface that summarises is furniture.** No box, no fill: rules only. The
  **ruled ledger** is the idiom — `grid … gap-x-8 gap-y-5 border-y border-border
  py-4`, cells carrying nothing but `pr-6`. It is what the headline figures use
  on `/models`, `/usage`, a model page, and inside the benchmark and job panels.

The app used to give both the same 1px box at the same radius, which is why a
stat band and a model tile carried equal weight and a page of them read flat.
The distinction is the fix; the shadow is only how one half of it is drawn.

Either side of the sheet:

- **`shadow-raised`** — a control that sits a hair proud of its surface: the
  checkbox, the switch thumb, the active tab, the composer.
- **`shadow-overlay`** — anything that floats free of the page: a menu, a select
  popup, a dialog, a toast, a chart tooltip. It separates from a sheet that is
  *already* casting a shadow, so the difference is travel (48px) rather than
  darkness.

**Two surfaces are frosted**, and only two: the composer and the toast
(`bg-*/75 backdrop-blur-2xl backdrop-saturate-150`). Both land on top of content
that is still live underneath — a thread being scrolled, a queue that is moving —
and a solid card cuts a hole in it. The blur is what stops what passes behind
from being *readable*, and the saturate puts back the colour that blurring alone
averages away. Everything else that floats is opaque: a menu or a dialog replaces
your attention rather than sharing it.

Every elevation lives in the theme layer with the other roles. The two modes are
deliberately **not** the same shape: light lifts with a shadow mixed from the ink
(grey on cream reads as dirt), while dark — where a shadow has nothing to fall on
— lifts with its surface step plus a highlight along the top edge, the way a pane
catches the light above it. This is why `shadow-sm`/`md`/`lg`/`xl` are banned:
Tailwind's defaults are black at a fixed opacity, invisible on the dragon ground,
which is what made those controls read flat in dark mode.

A page that opens with a ruled ledger drops the rule under its `PageHeader`
(`className="mb-6 border-b-0 pb-0"`): the ledger's own top rule is the section
break, and two parallel rules 32px apart read as an accident.

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
  (see `LEARN_WITH_RUN` in `components/training/run-config.ts`).

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
- **Reach for a role, not a raw value.** `rounded-(--radius-sheet)`, not
  `rounded-lg`; `shadow-overlay`, not `shadow-md`; `eyebrow`, not `text-2xs
  font-medium tracking-[0.14em] uppercase`. An arbitrary value (`text-[11px]`,
  `rounded-[4px]`) means either the role is missing or you want the wrong thing —
  see [Sizing](#sizing).

### Where the data lives

The backend owns people, sessions and presets. Everything else — models,
conversations, runs, scores, usage — belongs to the Python engine, and the
backend forwards to it. So a route here is `/chats`, `/models`, `/runs`,
`/benchmarks`, `/usage`, `/tokenize`, and the shape that comes back is the
engine's.

Two names are adapted at the wire rather than renamed through every component,
each in one place: the engine's `run_id` is this side's `id_job` (`lib/jobs.ts`),
and its `benchmark_id` is `id_run` (`lib/benchmark.ts`). Nothing else is
reshaped — a field the engine adds arrives without a change here.

### Stream protocol (chat)

One socket per chat, opened at `/ws/chats/{id}`:

```
client → { token }                            (auth frame)
server → { type: "ready" }
client → { model_name, text, attachment_ids?, system?, budget?,
           temperature?, top_k?, top_p?, context_multiplier? }
server → { type: "token", token }             (repeated)
server → { type: "tool_call", tool, argument } (the reply asked for something)
server → { type: "done", message_id, content, tokens, usage, final }
server → { type: "tool_result", tool, argument, body }
client → { type: "interject", text }          (any time after the opening ask)
server → { type: "interjected", text }        (spliced into the next round)
server → { type: "error", error }
```

The chat exists before the socket opens — its id is in the URL — so there is no
"which conversation is this" frame and no id to wait for coming back. The reply
is written when the last token lands, so closing the tab loses it rather than
storing half of one.

**A turn that calls a tool runs more than one round**, so `token` and `done` can
each arrive twice — the reply that called, then the reply that answers with what
came back. `final` on a `done` is the one that ends the turn; the client closes
the socket on that and not on the first, which would cut the turn off before the
answer it went to fetch. Absent is treated as final.

The two tool frames bracket the wait: `tool_call` the moment the marker closes
(so a search can be shown as started while the rest of the reply is still
streaming), `tool_result` once the backend has run it. `body` is in the marker
vocabulary — a `<|list|>` of findings, or a `<|warn|>` if the tool failed.

A fast model streams straight through at whatever rate it generates —
`onToken` appends to `messages` on every frame, no artificial pacing. What has
to keep up with that is the thread's own scroll, next.

**The composer stays live while a turn streams, and `interject` is what typing
into it then does.** `hooks/use-chat.ts::sendMessage` checks whether a turn is
already in flight; if so, the message goes into a small local `queued` list
(rendered by `components/chat/queued-message.tsx`, deliberately not a bubble —
muted text and a hairline rule rather than a fill, so it reads as *not sent
yet*) and out over the open socket as `interject`, rather than opening a second
one. `interject` has no `attachment_ids`: see the "Interjecting mid-turn"
section of `docs/backend/README.md` for why the backend keeps that contract
text-only.

The backend can only splice an interjection in between a round's tool result
and the next round's request — see the same section — so it is not
instant. The `interjected` confirmation is the frontend's cue that it landed:
`onInterjected` moves the message out of `queued` and into the turn's own
trace, reusing `groupTurns`'s existing fold rather than a renderer of its own —
a `user` row arriving while the previous assistant turn is still mid-flight
(`toolAnswers` set, `content` still empty) can only be an interjection, the
same test the `tool` row branch already uses to know a round hasn't finished.
**A message still `queued` when the turn ends is sent as an ordinary new turn**
rather than lost — the turn closing with no further rounds means the backend
never got the chance to fold it in, not that it should be discarded. All three
endings drain it: `done`, an error, and Stop, which settles the socket by hand
and so has to put the turn's state back itself. The drain reads `isSending`
and `queued` through refs written at those endings rather than through state,
because it runs before React has re-rendered — a drain that asked state
whether a turn was running would be told "yes" by the turn that just ended and
quietly re-queue the message it was draining.

### Stream protocol (a gym run)

One socket per run, opened at `/ws/gyms/runs/{id}` (`lib/gym-socket.ts`), and
deliberately the chat vocabulary with four frames added:

```
server → { type: "episode_start", episode_id, index, task_type, prompt }
server → { type: "round_done",   episode_id, text, usage }
server → { type: "episode_done", episode_id, score, correct, answer, ground_truth }
server → { type: "run_done",     score, correct, graded, episode_count }
```

Two differences from chat, and both are the reason it is a socket of its own.
The engine's `done` arrives as **`round_done`**, because an episode takes a
round per command and a client counting `done` would see an episode end every
time the model ran `ls`. And every frame carries its `episode_id`, so a run of
twenty tasks is one connection rather than twenty.

The socket carries only the sampling — the model and the tasks were fixed when
the run was created — and **closing it abandons the run**: episodes already
graded are stored, the rest are never run.

`context_multiplier` is in that sampling for a reason particular to gyms: a task
plus its system block is ~70 tokens before the shell has answered anything, and
half the window is reserved for the reply, so a model trained under 256 cannot
hold a task unless the window is stretched. `RunForm` says so before a run is
started rather than after it scores zero.

### Reading a gym episode

`components/gyms/episode-transcript.tsx` renders an episode with
`components/chat/trace.tsx` and `components/chat/rich-message.tsx`, composed the
way `chat-message.tsx` composes them: the task is a user bubble, the rounds that
called the shell are the trace, the round that stopped calling is the reply.

That reuse is the point rather than a saving. What a gym shows has to be what
the model actually produced — the same markup, the same reasoning panel, the
same command output in the same code block — because the model is doing in the
gym exactly what it does in a conversation. A renderer of its own here would be
a second opinion about that, and the first time the two disagreed the gym would
be the one lying.

`hooks/use-gym-run.ts` is what makes a live run and a reloaded one the same
shape: it builds a `Transcript` from frames while they arrive and from stored
steps on a reload, so the page below never learns which it is looking at.

### Following the bottom

`chat-thread.tsx` keeps the thread pinned to the bottom while a reply streams
in, without fighting a reader who scrolls up mid-turn to check something
earlier. Two refs carry the state a re-render shouldn't own:

- **`stickRef`** — whether to follow at all. Set by the user's own wheel/touch
  gestures (`trackStickIntent`), not by where the scroll happens to sit: it
  reads live `scrollTop` only on a genuine gesture, so a programmatic scroll
  this component just made never reads as the user asking to unstick.
  Re-engages the moment a gesture lands within `STICK_TO_BOTTOM_THRESHOLD_PX`
  of the bottom — scrolling back down by hand resumes following, no separate
  "jump to latest" affordance needed.
- **`lastHeightRef`** — the last measured `scrollHeight`, so the follow effect
  only fires on genuine growth (a new line) and not on every token mutating
  text within the line that's already on screen.

**The scroll itself is always instant (`behavior: "auto"`), never
`"smooth"`.** It used to glide during streaming, and a fast model breaks
exactly that: a line wrapping is a new `scrollTo` call, a fast enough model
wraps a line every render, and each call retargets a `smooth` animation that
hadn't finished chasing the *previous* target. The visible position ends up
permanently trailing the true bottom rather than catching up — which is what
"the text disappears behind the composer" actually was: not a rendering
slowdown, a scroll animation that can't keep pace with how often it's being
retargeted. Snapping on every growth event keeps the view exactly pinned
regardless of arrival rate; the growth events are themselves small (one
wrapped line at a time), which is what keeps this reading as a glide rather
than a jump even though nothing is animating.

### Attachments

A file is uploaded **when it is picked**, not when the turn is sent
(`lib/attachments.ts`). That is the whole design decision on this side, and it
buys one thing: the composer on the home screen has no conversation to attach
to, so any scheme that holds files until send makes the first message of a new
chat a different code path from every other one. An upload belongs to its owner
until a turn claims it, and `attachment_ids` on the socket's opening frame is
what claims it.

The consequences are all in `chat-composer.tsx`:

- **Emptying the strip comes in two kinds.** After a send the rows belong to the
  turn, so only the local object URLs are released. Abandoning a draft
  (`/clear`) has to delete the rows too, or every discarded draft leaves stored
  files nothing will ever claim.
- **Nothing sends while an upload is in flight.** A turn stored without the file
  the person is watching arrive cannot be given one afterwards — there is no
  route that edits a stored message.
- **A failed upload stays visible** rather than being dropped from the strip, and
  is not sent. A file past the per-turn cap becomes one of these without being
  uploaded at all, so a drop of twelve says which two are not going.

Rendering one costs a fetch: the bytes are behind the same bearer token
everything else carries and an `<img src>` cannot send a header, so
`useAttachmentUrl` fetches the blob and hands back an object URL it revokes on
unmount. Thumbnails sit in a fixed box, so a thread of mixed aspect ratios does
not reflow as each one lands.

**The strip has one height, not one shape.** A picture and a document are
different objects and get different tiles, but both are `h-28`, because a mixed
row whose tops do not line up reads as two things that happen to be adjacent.
The file tile lifts the extension out of the name and into the meta line
(`TXT · 5.0 MB`): a card that wide truncates the end of a long filename, and the
end is exactly the part that says what the file is.

### Dictation

The microphone next to Send is the browser's own `SpeechRecognition` — free,
keyless, nothing in the bundle, and absent on Firefox, where the button does not
render at all. What it costs in privacy and why it beat a local Whisper is in
[`decisions/voice-input-uses-the-browser.md`](../decisions/voice-input-uses-the-browser.md).

Two hooks, because they are two browser APIs with two lifetimes:

- **`use-speech-recognition.ts`** owns the recogniser. It hands back `final` and
  `interim` as separate strings — the second is a guess the recogniser keeps
  rewriting — plus `onDevice`, true when Chrome's local pack is already
  installed. It restarts its own session on `onend`, because a session ends by
  itself after a stretch of silence even with `continuous` set; a run of
  sessions too short to have heard anything is how that restart chain ends
  rather than spinning against a browser that has said no.
- **`use-audio-level.ts`** opens a *second* capture of the same microphone,
  because `SpeechRecognition` exposes no audio to draw. It calls back once per
  frame with three bands and never sets state; `VoiceWaveform` writes the
  transforms to the DOM itself.

In `chat-composer.tsx` the spoken text lives in a **range** — where this
dictation is writing — that is replaced whole on every event rather than appended
to, since the interim tail is revised and not added to. The range is ProseMirror
positions, so the editor's `onTransaction` maps it through anything typed
alongside it, skipping the composer's own inserts by a meta flag. The interim
tail carries the `interimTranscript` mark (faded, `inclusive: false`), which is
stripped when dictation ends and which `docToMarkdown` would ignore anyway.

Sending or `/clear` stops dictation. The footer band holds the three bars in its
middle column while it runs — it is a `1fr auto 1fr` grid rather than a two-ended
row precisely so the bars are centred on the band itself, and so the title and
the context meter never move to make room for them. The bars are also the only
thing that animates: the mic button just fills blue and swaps to a stop glyph,
the same swap the send button makes mid-turn. `Ctrl`/`Cmd`+`Shift`+`Space`
toggles it; the mic button takes both a click (dictate until clicked again) and
a hold (dictate until released), told apart by how long the press lasted.

### The context meter

The right end of the footer band: a bar and `used/window`, tinted at 70% and
again at 90% (`meterTone`). It counts two things that are known to different
standards, and keeping them apart is the whole of the design.

- **The conversation behind the draft is an estimate.** The last persisted total
  from the sidebar, plus whatever a turn in flight has tallied client-side as it
  streamed — see `chat-thread.tsx`. Neither half was checked against a
  tokenizer.
- **The draft in the composer is a real count.** `hooks/use-draft-tokens.ts`
  posts it to `/tokenize/count` 200ms after typing settles, and the engine
  encodes it under that model's own vocabulary. A guess from character count
  would be wrong by an amount that changes with the model — these vocabularies
  are trained per model — against windows that start at 256 tokens, where a
  handful either way is a percentage.

So the bar is drawn as two segments, the draft's at 45% opacity, and the tooltip
says which number is which. The alternative — one figure labelled "estimated" —
threw away the half that is exact.

Three things the hook does that are not optional:

- **It counts the markdown, not the text on screen.** `docToMarkdown` is what
  `submit` sends, and a list's `- ` is in the string the model is handed.
- **It holds the last count while a request is in flight**, and on failure keeps
  it rather than falling to zero. A meter that emptied itself mid-sentence would
  be a worse lie than one a keystroke stale.
- **It returns to zero without a round trip** when the composer empties, adjusted
  during render rather than in an Effect — otherwise the count of a turn already
  sent survives a frame into the next one.

A recognised standalone command (`/clear`, `/model …`) reads as zero: it runs
locally and never reaches the model, so it costs the window nothing.

Counting per keystroke is only affordable because the engine caches the parsed
vocabulary — see [`../engine/README.md`](../engine/README.md). Attachments
contribute nothing here, and that is a property of the system rather than an
omission: nothing reads them
([`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md)).

### Opening one

`AttachmentViewer` is a panel beside the thread, not a dialog over it — the
point of opening a file mid-conversation is to read it *against* what was said
about it, and a modal would take the conversation away to show its attachment.
`ChatThread` is a flex row: the chat column is centred in whatever width is
left, so it drifts left as the panel arrives and back when it closes, with no
second layout to maintain. Below `lg` there is no room for both and the panel
takes the width; a half-conversation beside a half-document is neither.

Which state lives where matters: the open file is `ChatThread`'s, not a
message's, because opening a second one has to *replace* the first — a turn that
owned its own viewer could not do that.

Four previews, and each is its own component because each needs a different
fetch and a hook cannot be called conditionally:

| Kind | What it is | Shown as |
| --- | --- | --- |
| `image` | `image/*` | The picture, contained |
| `text` | `text/*`, JSON/XML/YAML, or a known extension | The decoded contents |
| `embed` | `application/pdf` | An `<iframe>` — the browser's own viewer |
| `none` | anything else | A note, and the download in the header |

Extensions are trusted over the media type for the text case, because the
browser guesses the type from the extension anyway and guesses badly: a `.bak`
arrives as `application/x-trash`, which is a verdict rather than a type.

**A text preview is capped** at `PREVIEW_CHARS` (200k). People attach novels —
one of the files this was built against is a five-megabyte `.txt` — and laying
that out as one text node stalls the tab. The rest is stored and downloadable,
and the panel says so rather than letting the reader discover the file stops
mid-sentence.

Nothing here is sent to the model, and nothing is counted against the context
window — see
[`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md).

### Acting on a message you sent

A user bubble carries three actions (`UserActions` in
`components/chat/chat-message.tsx`): copy, "edit and ask again", "ask again".

**Both re-ask; neither rewrites.** A new turn is appended with the same (or
amended) question, and the original stays where it is. That is a property of the
data, not a UI choice — there is no route that removes a stored message
(`routes/chats.rs` has create, rename, and delete-the-whole-chat, nothing
finer), and the thread *is* the prompt, so editing in place would either lie
about what was sent or silently drop everything after it. Giving these branching
semantics means adding a truncate to the backend and the engine first.

Edit goes through the composer rather than sending directly, via its `draft`
prop — so the question can actually be changed, and so sending still happens on
the one path that owns sending. The prop is keyed by object identity, not by the
string: editing the same message twice has to load it twice.

### The trace

A turn's working — every round before the one that answered, plus what each call
came back with — renders as one panel above the answer
(`components/chat/trace.tsx`), not as the rows it is stored as.
`hooks/use-chat.ts::groupTurns` does the folding: `assistant → tool → assistant`
is the right shape for the *model*, whose next prompt has to contain all three,
and the wrong shape for a reader who asked one question.

**However many rounds it took.** A turn is not capped at one call and one answer
— the backend allows `MAX_TOOL_ROUNDS` of them, and a model that reads a failed
command and tries something else is the normal case, not the exception. Both the
folding and the live stream accumulate: `groupTurns` appends each round's markup
to `working` rather than assigning it once, and the streaming path folds the
answer-so-far back into the working when a later round calls again. Getting
either wrong shows up immediately — as a second reasoning block and a bare call
hanging under the trace, or, in the stored view, as a middle round vanishing.

The panel's rail is its timeline. Reasoning is prose against it, a tool call is a
tick off it (`components/chat/trace-step.tsx`), and the whole thing is
achromatic except a step in flight — that tick warms to `--throbber-spark`, the
token the status row's throbber is using at the same moment. A turn that called
nothing has no trace and renders exactly as it always did, reasoning inline.

`/markup` carries two fixture traces, including the failure path.

While a call is in flight the thread's status row names it —
`Search: tallest mountain in europe`, `Bash: ls -l /etc` — off the same reading
the live tick uses (`Trace::callInFlight`, the first call with no answer yet).
How a tool is said in English is `lib/tools.ts` and nowhere else: a noun for
the status row, a present and a past verb for the trace, and a fallback to the
tool's own wire name so a new tool reads acceptably before anyone has written
words for it. That table is the frontend's only knowledge of which tools exist
— the vocabulary stopped carrying one marker per tool, so nothing else here can
tell `bash` from `web-search`.

`budget` is the composer's thinking-budget choice, one section of
`components/chat/composer-picker.tsx` (the type and its options live in
`lib/think-budget.ts`, shared with `/thinking`'s argument list) — `"low" |
"med" | "high"`, matching the labels the SFT corpus is tagged with, since the
string sent becomes the marker the model was trained under. **"Auto" is the absence of a budget, not a fourth
value**: it is omitted from the frame entirely, so the engine renders no marker
and the model is prompted exactly as it always was. It rides through
`PendingMessage` when a new chat navigates, same as `model` — see
"The composer's own choice" below for both.

`temperature` / `top_k` / `top_p` are the decoding knobs, and unlike `budget`
they **are** settings (`lib/settings.ts`, rendered as sliders in the ⌘K modal) —
how random you want decoding to be is a standing preference, not a property of
one question. Each is omitted individually when unset or non-finite, and the
engine then falls back to its own defaults.

`context_multiplier` is a setting the same way: how much headroom past the
trained context window you want is a standing preference, not a property of one
question. 1 (the default) is the model exactly as trained; the engine builds an
NTK-scaled RoPE table fresh per turn for anything above that, computed from the
model's own `ctx_window` rather than a token count this side would have to keep
in sync with it — see `docs/engine/README.md`.

### `/prune`

Cuts a chat's usable range back to what it would be at `context_multiplier: 1`,
while leaving the multiplier itself wherever it's set — real headroom past the
trained window again, instead of already sitting near a raised cap. Nothing is
deleted: every message stays exactly where it was, visible in the thread and
counted in `/usage`. What changes is what a *future* turn is built from.

The mechanism is deliberately stateless on the engine's side. `GET
/chats/{id}/context-floor?model_name=...` is a pure read — it answers "where
would the cut land under *this* model's own `ctx_window`," using the identical
walk `build_prompt` already runs, and writes nothing down. `lib/context-floor.ts`
is what remembers the answer: a message id **and the model it was cut for**,
kept in `localStorage`, per chat, at the same durability tier as every other
chat setting in this file — lost on a cleared browser or a different device,
same as your temperature would reset.

**A floor is only ever resent to the model it was computed for.** `ctx_window`
is a property of the model, not the chat, so a cut sized to one model doesn't
mean anything for another — `sendMessage` in `use-chat.ts` compares the stored
`modelId` against the model the turn is actually going to and sends
`history_floor` only on a match; switching models makes a stale floor inert
rather than wrong, without discarding it, so switching back makes it live
again with no extra round trip. The engine's own fallback (a stale or
unrecognised floor is treated as no floor rather than erroring, `_floored` in
`api/routes/chats.py`) is what makes "inert" safe to fall back to: a client's
memory of a prune is a convenience, not something a turn should fail over.

If a prune ever needs to survive a lost `localStorage` or a second device, this
is the point to reconsider: persist the floor against the chat server-side
instead of trusting the client to keep resending it.

The composer footer subtracts the floor's own reported token count from
`chatSummary.tokens` rather than asking the engine for a second, smaller total
— `chatSummary.tokens` is also what `/usage` bills from, and has to keep
counting every token this chat ever cost regardless of what a later prune
excludes from the model's view of it.

**The meter's badges are how a prune is actually seen.** Past the trained
window under a raised multiplier, one reads `+190% ×4` — how far over the
trained size you are, and what's letting you get away with it. Once `/prune`
succeeds, a second badge appears beside it: `62% unseen`, how much of this chat
the model can no longer see. `use-chat.ts` reports this as 0 — same as never
having pruned — the moment the stored floor stops matching the selected model,
so switching models doesn't leave a stale "unseen" figure claiming a cut that
isn't actually being applied to what's being asked now. They are independent
facts shown independently,
not two views of one state — pruning brings usage back under the trained
window the instant it runs, but the conversation can grow past that window
again afterwards, at which point both are true at once (some of the chat is
unseen, *and* what's left has gone past the trained size again). Both badges
read off the same `tone` (`composer-footer.tsx`'s `meterTone`) as the bar and
the count beside them, so either turning red means the same thing.

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

Everything else in `Settings` is stored on the account, not the browser —
`components/settings/settings-provider.tsx` fetches `GET /user/me/settings` on
login and writes back with `PUT` on every change, debounced 500ms so a dragged
slider or a typed system prompt isn't one request per tick. Logged-out state is
*derived* (`isAuthenticated ? settings : DEFAULT_SETTINGS`) rather than written
in the effect that notices the session changed, same reasoning as
`ChatsProvider`'s empty list — a synchronous `setState` in that effect is what
the `react-hooks/set-state-in-effect` lint rule catches. A fetched blob is
merged over `DEFAULT_SETTINGS`, so a setting added after an account last saved
comes back with its default rather than `undefined`. See
[`../backend/README.md`](../backend/README.md) for where it lives server-side.

### The composer's own choice

`model` and `budget` — what to send with, not how to behave — are not in
`Settings` and never render in the ⌘K modal; they stay `ComposerChoiceProvider`
state (`components/chat/composer-choice-provider.tsx`), read and set through
`useComposerChoice()`, exactly as before. What changed is that the choice now
persists too, through its own resource, `GET`/`PUT /user/me/composer-choice` —
a separate blob from `Settings` rather than two more fields on it, because a
model or budget pick writes immediately (it's one discrete action, not a
dragged slider) while a settings change debounces; sharing a column would mean
whichever write lands second clobbering the other's field. Same
account-not-browser reasoning, same derive-on-logout shape as `Settings` above.

A persisted model id can go stale between sessions — the model got deleted or
unpublished — so `ComposerPicker` corrects an id that isn't in the loaded list
back to the first available one, the same effect that already handled "no
choice yet"; that's the one place with the list, so it's the one place that
needs to know.

**The stored id may also name a checkpoint**, which is in no model list by
design, and so is indistinguishable from a stale one until asked about.
`hooks/use-checkpoint-parent.ts` is what asks — one request, cached against the
id it answered for — and the correction above waits on it, or every checkpoint
choice would be discarded the moment it was made. It also supplies the model a
checkpoint belongs to, which is what the trigger names and what the list marks
as active. Nothing on the chat path ever learned what a checkpoint is: it is a
model id and a directory, exactly like the finished weights
([`../decisions/checkpoints-are-models.md`](../decisions/checkpoints-are-models.md)).

**Model, checkpoint and thinking budget are one popover**,
`components/chat/composer-picker.tsx`, replacing the footer's model trigger.
Earlier this was two-to-three separate controls — a model dropdown, a
checkpoint dropdown rendered as its sibling only when the model had
intermediates, and a wholly separate thinking-budget dropdown — with the
checkpoint control deliberately kept as "a sibling and not a flyout off the
model list, so that list stays a list of models." **That rationale is
superseded here, at the user's request**: the three are all answers to "what
should this turn send with," and one control that opens to three legible
sections reads better in the footer than several buttons in a row. The
checkpoint section still only appears nested under the model it belongs to,
and only when that model's run left intermediates, listing its finished
weights and each checkpoint by index and step — the same list
`components/training/resume-picker.tsx`'s `ResumePicker` shows, off the same
`useCheckpoints` hook. The thinking-budget options are their own section,
under a separator, rather than blended into the model list — a budget and a
model are different kinds of choice and are shown as such. `ComposerPicker`
owns none of the state itself: it composes `useModels`, `useCheckpointParent`
and `useCheckpoints` exactly as before, and reads and writes `model` and
`budget` through `useComposerChoice()` like everything else in this section.

### Slash commands & fuzzy finder

Typing a leading `/` in the composer brings up the floating fuzzy finder popover
(`components/chat/slash-command-menu.tsx`, driven by `lib/commands.ts`).

- **Stage 1 (Command Discovery)**: Fuzzy-matches command names, aliases, and
  descriptions (e.g. `/model`, `/thinking`, `/show-thinking`, `/temperature`,
  `/top-k`, `/top-p`, `/context-multiplier`, `/prune`, `/system-prompt`,
  `/notify`, `/theme`, `/settings`, `/help`, `/clear`, `/new`).
- **Stage 2 (Argument Selection & Range Guides)**:
  - `/model <name>`: Fuzzy-filters and selects available models.
  - `/thinking <budget>`: Sets the thinking budget (`auto | low | med | high`).
  - `/show-thinking` / `/notify`: Toggles reasoning visibility and notifications.
  - `/temperature`, `/top-k`, `/top-p`, `/context-multiplier`: Displays allowed
    ranges, current values, and clickable preset buttons.
  - `/prune`: No argument. A no-op toast at `context_multiplier: 1`; otherwise
    asks the engine for this chat's floor and toasts what it cut — see the
    note on `/prune` above.
  - `/system-prompt <text>`: Previews current prompt and offers templates or `clear`.
  - `/theme <light | dark | system>`: Switches visual theme.
  - `/new`: No argument. Navigates to `/`, the empty composer — the same
    destination the sidebar's own "New" link goes to, not a route this command
    invented. Available from any composer mount, including the new-chat screen
    itself, where it's a no-op navigation to where you already are.
- **Keyboard navigation**: Seamless `↑` / `↓` navigation, `Tab` autocomplete,
  `Enter` execution / selection, and `Esc` dismissal. Standalone commands execute
  locally with toast feedback and are never dispatched over the chat socket.

### The admin panel

Everything behind the admin flag is reached through **one row** in the sidebar
footer, beside Settings — not a labelled nav group. The row is a mode of the app
you have access to, which is what the footer holds; the content above it is the
subject you are reading.

- `/admin` is the panel, in three parts: `QueueView` as the hero,
  `AdminDirectory` under it, and `HistoryView` last. **The directory sits
  between the two halves of the queue on purpose** — history is unbounded, so
  anything below it is reached only by scrolling past every run ever made.
  **Add an admin destination to the directory, not to the sidebar** — a
  directory has room for a sentence saying what a page is, and a nav row does
  not. The two bands are what puts work on the GPU (`WORKBENCH`) and what only
  renders a fixture (`INTERFACE`).
- **`/data` is a `WORKBENCH` destination**, not a second nav row. It reads the
  data engine through `lib/data.ts`; the cleaning workbench previews a pipeline
  over the first few documents on a debounce, which is why the sample is always
  taken by ordinal — a moving sample would compare two different texts as the
  user tunes a parameter. The pipeline is a **horizontal rail** above the
  before/after and the open step's parameters sit under it: the steps are an
  order, and a column that wraps reads as a second pipeline. The before/after
  is a split diff (`lib/diff.ts` — an LCS over lines, no dependency), with
  untouched stretches folded away so what a step did is not buried in what it
  left alone.
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

- **The board lives at `/admin`**, as the hero of the panel — `QueueView` and
  `HistoryView` are mounted there, and `/jobs` is a redirect kept only so
  bookmarked URLs work. `/jobs/{id}` and `/jobs/new/*` are unchanged. The two
  are separate components only so the panel can put its directory between them;
  they read from the same `JobsProvider` and are one ledger.

#### Composing a training run

The same fields are rendered two ways, and neither may drift from the other:

- `components/training/train-board.tsx` at `/jobs/new/train` — the decisions in
  panels down the left, and a **build sheet** beside them stating what the run
  *is* rather than what was typed. That sheet is the spec-sheet idiom
  `/models/[id]` uses for a model that exists, on purpose: composing a run and
  reading a model back are one object at two moments.
- `components/training/params-rail.tsx` inside a follow-up step, where the same
  fields have a rail's width and no submit of their own.

Both take their field list, their derived figures and their validation from
`components/training/run-config.ts`, which holds no JSX. A rule about what a
run may be belongs there, never in a layout.

**A run that continues from a checkpoint states no architecture at all.** The
engine takes the shape from the state it resumes and *refuses* a request that
restates it differently — it cannot let the request win, because the weights
would win anyway and the run would report an architecture it did not train. So
`RESUMED_ARCH_FIELDS` (`lib/training.ts`) is dropped from the payload in
`lib/jobs.ts`, exactly as a fine-tune's `base_model_id` is translated there, and
the same list is what `fieldsFor` hides and `validate` skips. Hiding without
dropping is the failure mode to watch for: pydantic reads a field left at the
form's default as one the request *stated*, so a control nobody can see is still
enough to have the run rejected.

#### Reading a run back — the diagnostics panel

`components/training/diagnostics-panel.tsx` renders `DiagPoint[]`
(`lib/training.ts`), which is the engine's `Diagnostics` dataclass **verbatim**
— nothing between the training loop and the chart renames a field, so a rename
on either side arrives as `undefined`. The engine's side is documented in
[`../engine/telemetry.md`](../engine/telemetry.md).

- **An optional field means "not measured", and must never render as 0.** The
  engine drops what it did not measure instead of sending a zero, because a zero
  reads as a real measurement. Every fine-tuning field on `DiagPoint` is
  therefore optional; `opt()` in the panel maps `undefined` to `null` so recharts
  drops the point rather than plotting it at the bottom of the axis.
- **The fine-tuning figures are their own presence check.** A frame carries no
  mode flag: an SFT run sends `loss_masked` and a pretrain run sends nothing, so
  `diag.some(d => d.loss_masked !== undefined)` is what gates them. A pretrain
  run must show no empty SFT charts at all.
- **Every figure is `Figure` + recharts + `chart-theme.ts`.** Colour is chosen
  by what the series *is*: per-layer, per-head and by-position are magnitudes by
  index and take a single-hue sequential ramp; the three fine-tuning losses are
  categories and take distinct hues. Do not add a palette to a component.
- **Every array derived from `diag` is built in a `useMemo` keyed on `diag`,
  with no exception for a one-off passed straight into JSX.** The embedding
  biplot and globe key their own expensive work — `alignFrames`'s sign-matching
  across every stored report, `EmbeddingGlobe`'s icosphere/convex-hull rebuild —
  off the *reference* of the array they're handed, not its contents. A `diag.map(...)`
  written inline in the render body hands them a new array on every render of
  the panel regardless of whether `diag` itself changed, which silently defeats
  that memoization: the full rebuild reruns on every render rather than once per
  new report. A live run reports every few seconds and hides this; a finished
  run — reloaded with its whole history already in state — does not, and a page
  stuck re-running it on every render reads as frozen.

- **The board polls.** It moves when a run is queued, starts or ends — rare
  enough that a 3s refetch of `GET /runs` says the same thing a nudge would,
  without a second live connection to keep alive across sleeps and restarts.
- `hooks/use-jobs.ts` owns the board; `hooks/use-job.ts` owns one job — its
  durable row plus whatever it is broadcasting. **Panels render from the row
  first and treat live state as an enrichment**, because a queued or finished
  job has no frames at all.
- **The appendix's checkpoint list is built from `checkpoint` frames**, not from
  `TrainStats.checkpoint_path`. A checkpoint interval is a multiple of the report
  interval, so the epoch-boundary save is usually deduplicated against the
  interval one and its frame carries no path — reading the path listed almost
  nothing, and nothing at all on a single-epoch run. The frame is the engine's
  `Checkpoint` dataclass verbatim (`CheckpointFrame` in `lib/training.ts`) and
  announces every save as its weights land, so the list is complete and each row
  can say where in the run it was taken.
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

#### What a training run's phase 1 shows

`components/training/telemetry.tsx` holds the phase-1 checklist — which build
step is running, and what each produced — and under it
`components/training/build-metrics.tsx`, what the build actually *produced*:
characters per token by source, how much of the vocabulary the corpus never
uses, the share of training windows that span a document boundary and a ribbon of
what those windows actually look like, document lengths against the window, and
the head of the merge list in the order BPE learned it.

Three things about it are load-bearing:

- **It outlives phase 1.** The checklist is replaced by the loss overlay the
  moment training starts; these stay, because they are facts about the corpus the
  run is training on, and a finished run's page is where they are read.
- **An absent field is "—", never 0.** The engine omits what a phase did not
  measure (see [`../engine/README.md`](../engine/README.md)), and a corpus packed
  before a metric existed reports none of it. Drawing that as a zero would be a
  number that reads as a real count of nothing.
- **`ctx_window` comes from the run's config, not from the frame.** The window is
  a property of the run, and one packed corpus can be trained at two of them —
  which is also why the engine measures the length histogram at report time
  rather than baking it into the build.

##### The span ribbon

`components/training/span-ribbon.tsx` is the picture behind the straddle stat,
rendered directly under it: a sample of the run's actual training windows, one
bar each, split into the documents inside it and coloured by dataset. The number
cannot tell a corpus of books (a window sitting mid-chapter) from a corpus of
tweets (forty unrelated things per window), and those want opposite decisions.
The engine's half — the sampling strategy, the frame bound, the cost — is in
[`../engine/telemetry.md`](../engine/telemetry.md).

- **SVG, not canvas2d, and the element count is not why.** The engine caps the
  frame at `RIBBON_WINDOWS × RIBBON_SEGMENTS` rects, which either would draw
  without noticing. What decides it is that the fills are the theme's own values
  applied as attributes (canvas would need `getComputedStyle` and a redraw on
  every theme switch), a per-window tooltip is a `<title>` child rather than a
  hit-test, and the file needs no `'use client'`, no ref and no effect — a canvas
  is a client component by construction (R14). Lifting the cap is the signal to
  revisit this, and nothing else is.
- **A bar's own edges are not document boundaries.** The hairlines are drawn only
  where a document actually ends inside the window (`closes`), never at the
  window's edge. Drawing the two alike is what turns one book into nine unrelated
  documents, which is precisely the reading the figure exists to correct.
- **The elided tail is drawn thin and muted, never at bar height.** A window past
  the engine's per-window cap reports `overflow` and stops short; filling that
  space at full height would claim one long document where there are dozens of
  short ones.
- **A dataset is a category, so it takes a hue and not a ramp** — the same wheel
  `lossStageColor` uses (`datasetColor` in `chart-theme.ts`), because a dataset
  and the curriculum stage it was packed into are one subject seen twice. As
  everywhere else here: do not add a palette to a component.

### The attention viewer

`components/models/attention-viewer.tsx` on `/models/[id]`: pick a layer and a
head, give it a text, get that head's attention matrix back. `lib/attention.ts`
is the wire (base64 fp32 in, a `Float32Array` out), `hooks/use-attention.ts` the
state, `components/models/attention-canvas.tsx` the figure. The engine side —
including why nothing records attention until someone asks — is
[`../engine/README.md`](../engine/README.md).

- **Canvas2d, and measured rather than assumed.** One backing pixel per cell,
  filled through a 256-entry ramp into one `ImageData`, one `putImageData`, and
  CSS scales the element up with `image-rendering: pixelated`. A cell is a pixel
  and not a shape, so a `fillRect` per cell would be 262,144 draw calls for
  something the compositor does as a blit. Measured in headless Chromium with
  **no GPU at all** (software raster, the pessimistic case): 512×512 — the
  largest matrix the engine will return — **median 6.3 ms, p95 12.7 ms**;
  256×256 1.6 ms; 128×128 0.3 ms. That is a one-off cost per fetched matrix,
  behind a network round trip and a forward pass that each cost more, and
  hovering redraws nothing. Canvas2d is comfortably enough here and there is no
  case for a GPU path; the panel reports its own last draw time so the next
  person can check that on their own machine rather than take this on trust.
- **The crosshair is not drawn on the canvas.** Two absolutely-positioned rules
  at a percentage offset, so following the pointer costs a style recalculation
  instead of a redraw of the matrix.
- **Colour lives in `components/models/attention-ramp.ts`, not in
  `chart-theme.ts`.** A weight is a magnitude, so it takes that file's
  single-hue sequential rule — but a *cell* at t=0 has to vanish into the
  surface where a *line* at t=0 still has to be visible, which inverts the
  bottom of the ramp. One hue at varying alpha, so the panel's own surface
  carries the low end and the figure is theme-aware without sampling a token or
  keeping a second dark-mode ladder. The hue is chart-theme's per-head violet,
  because attention entropy in the diagnostics panel is per head too.
- **The ramp is gamma'd (0.4) and the readout is not.** Rows sum to 1 over up to
  512 keys, so a linear ramp draws a diffuse head as an empty square. The figure
  is a display transform; the hovered cell always states the exact weight, and
  the tokens on both axes, which is the whole point of labelling them.

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

- the chat thread (loading, and the thinking/tool/generating status row),
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
