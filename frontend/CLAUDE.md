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
│   ├── files/
│   ├── usage/
│   └── profile/
├── components/
│   ├── ui/          shadcn/ui primitives (button, input, card, sidebar, etc.)
│   ├── auth/
│   ├── chat/
│   ├── conversations/
│   ├── files/
│   ├── models/
│   ├── profile/
│   ├── sidebar/
│   ├── training/
│   └── usage/
├── hooks/           Custom React hooks (use-chat, use-models, use-auth, etc.)
├── lib/             Pure utilities, API helpers, socket clients
│   ├── api.ts           HTTP helpers, auth token management
│   ├── chat-socket.ts   WebSocket chat streaming
│   ├── training-socket.ts
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

### Stream protocol (chat)

The chat WebSocket lives in `lib/chat-socket.ts` and follows this wire protocol:

```
client → { token }                            (auth frame)
server → { type: "ready" }
client → { chat_id, content, model_id }       (chat_id null = new chat)
server → { type: "queued", job_id, chat_id, ahead }
server → { type: "token", value }             (repeated)
server → { type: "done", chat_id, id_message }
server → { type: "error", message }
client → { action: "cancel" }
```

A reply is a job on the GPU queue, so `queued` always arrives before the first
token — with `ahead: 0` when the turn starts immediately, so there is one code path
rather than two. Because the job outlives the socket, closing the tab no longer
loses the reply: the runner persists it either way.

### The jobs queue

Everything that uses the GPU — training, benchmarks, chat turns — is a job. See
`lib/jobs.ts` (types + REST) and `lib/jobs-socket.ts` (one socket, two
subscriptions: the whole board, or one job's frames).

- **Board events are nudges, not payloads.** The socket sends `{ type: "changed" }`
  and the client refetches `GET /jobs`, so there is exactly one description of a job
  rather than a REST shape and a socket shape that drift apart.
- `hooks/use-jobs.ts` owns the board; `hooks/use-job.ts` owns one job — its durable
  row plus whatever it is broadcasting. **Panels render from the row first and treat
  live state as an enrichment**, because a queued or finished job has no frames at
  all (the socket answers `inactive`).
- Composers live at `/jobs/new/*` and are *only* forms: submitting queues a job and
  hands off to `/jobs/{id}`, where the telemetry lives. `?from=<job id>` seeds a
  composer from a previous run's stored config.
- `/benchmark` is history and analysis only — it no longer starts runs.
- The throbber's per-kind modes (`training` / `benchmarking` / `generating`) are how
  a running job says what it is. Give **one** instance a changing `mode` so it morphs
  between them as the queue advances rather than cutting.

### Scripts

```sh
bun run dev         # dev server
bun run build       # production build
bun run lint        # ESLint
bun run format      # Prettier
bun run typecheck   # tsc --noEmit
```
