# Frontend development

## Conventions

## Components

Read the complete [coding conventions](../conventions.md) before edits. Every
handwritten named function, hook and component has brief JSDoc describing its
purpose, inputs and outputs. No comments inside implementation bodies.


- **Server-first**: App Router pages are async server components by default. Only add `'use client'` when hooks, browser APIs, or event handlers are required.
- **One primary concern per file**; cohesive private components may stay together. Use named component exports; App Router pages/layouts follow Next.js export requirements.
- **Props interface**: Define a named `interface` or `type` in the same file, exported, prefixed with the component name.
  ```tsx
  export interface ChatThreadProps { chatId: string }
  /** Renders a conversation thread for the supplied chat id. */
  export function ChatThread({ chatId }: ChatThreadProps) { ... }
  ```
- **Business components** live in `components/<domain>/` and compose UI primitives from `components/ui/`.
- **Don't re-invent UI primitives**: Use the shadcn/ui components from `components/ui/` for buttons, inputs, dialogs, cards, etc. If you need a variant the existing component doesn't support, add a prop to it — don't hand-roll a replacement.

## Failure states

Use the notification owner in [`components/ui/toast.tsx`](../../frontend/components/ui/toast.tsx)
and [Notifications](notifications.md). Errors and confirmations do not get a new
boxed paragraph. `FormError` and `LoadError` retain their call-site contracts but
publish explanations through notifications with deduplication. Keep invalid-field
semantics and useful retry/recovery controls beside the affected content. A timed
popup disappearing is distinct from dismissing a retained notification.

## Selects

Use `components/ui/select.tsx` — never a native `<select>`. Two things about
the Base UI primitive underneath, both learned the hard way:

- `<SelectValue />` on its own renders the **raw value**. Give it a render
  function that maps the value to its label, or the trigger will read `0` while
  the list it opens reads `Layer 0`.
- An option meaning "nothing chosen" needs a sentinel value of its own; the
  empty string is how a Select says *no selection*. Map it back at the edge
  (see `LEARN_WITH_RUN` in `components/training/run-config.ts`).

## Hooks

- Domain hooks live in `hooks/` and are named `use-<domain>.ts`. Each file exports one primary hook.
- Hooks call `lib/api.ts` helpers for HTTP, and `lib/` socket functions for WebSocket. They do not inline `fetch()` or `new WebSocket()`.
- **State is local to hooks.** Components consume state and callbacks from hooks; they don't own `useState` for domain data.

## Data fetching

- **Server pages** fetch data available to their server authentication context. Browser-local bearer tokens are unavailable to server components; authenticated client data loads through domain hooks and `lib/` adapters.
- **Client components** fetch via domain hooks (e.g. `useChat`, `useModels`) that call centralized `lib/` HTTP/socket adapters.
- Mutations (create chat, send message, etc.) go through the socket or through POST/PUT/DELETE via `fetch` with `authHeaders`.

## Routing

- Page folders match the route: `app/chat/[id]/page.tsx` → `/chat/:id`.
- Use Next.js `<Link>` for navigation, not raw `<a>` tags (avoids full page reloads).

Sidebar hover/focus warms conversation messages as well as the route; composer
ghosts preserve its footprint while loading. See [Chat loading](chat-loading.md)
for cache freshness, session isolation, and selection readiness.

## Sessions and forms

`lib/api.ts` clears only the session that authorized a rejected request, so a
late 401 from a replaced account cannot log out the newer one; sockets pass the
token they opened with for the same reason. A mounted `SessionExpiryGuard`
answers expiry with a client navigation to `/login`, and repeated rejections on
one path redirect once. `/docs`, `/login` and `/signup` keep rendering: they
discard a stale token without leaving the page. Without a mounted shell the
adapter falls back to a document navigation.

Forms are submitted by JavaScript. Before hydration the browser would perform a
native GET on submit and reload the page, so `app/layout.tsx` installs a capture
listener that cancels the default for forms without an `action`. React's handler
still runs after hydration.

## Styling

- Use the shared `PageHeader` for page titles, back links and actions. Keep its
  shared identity texture and omit decorative header/list/stat separators.
  Headers contain no decorative description prose. Keep functional metadata
  such as task counts, status or thread/token figures; use header meta for figures
  beside actions. Stored resource descriptions and editable resource fields remain
  content, not introductions. Settings section titles stand alone. Secondary text
  carries measurements or instructions needed to use a control. Admin directory
  links use the grouped Workbench structure.
- Tailwind utility classes in JSX. No separate `.module.css` files.
- Use `cn()` from `@/lib/utils` for conditional class merging.
- Follow the existing design: Geist font (sans), the shadcn/ui color tokens, the sidebar layout.
- **Reach for a role, not a raw value.** `rounded-(--radius-sheet)`, not
  `rounded-lg`; `shadow-overlay`, not `shadow-md`; `eyebrow`, not `text-2xs
  font-medium tracking-[0.14em] uppercase`. An arbitrary value (`text-[11px]`,
  `rounded-[4px]`) means either the role is missing or you want the wrong thing —
  see [Sizing](appearance.md#sizing).

## Where the data lives

People, sessions, presets, conversations, files and usage are owned by Rust;
runs/models by the LLM engine,
and scores by bench-engine. Public paths remain `/chats`, `/models`, `/runs`
and `/benchmarks`; the backend resolves ownership.

Two names are adapted at the wire rather than renamed through every component,
each in one place: the LLM engine's `run_id` is this side's `id_job` (`lib/jobs.ts`),
and bench-engine’s `benchmark_id` is `id_run` (`lib/benchmark.ts`). Nothing else is
reshaped — a field the LLM engine adds arrives without a change here.

## Scripts

```sh
bun run dev         # dev server
bun run build       # production build
bun run lint        # ESLint
bun run format      # Prettier
bun run typecheck   # tsc --noEmit
```
