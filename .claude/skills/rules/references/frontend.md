# Frontend rules (TypeScript / React) — full text

### R14 — Server-first rendering
Pages and layouts are server components by default. Only add `'use client'` when
browser APIs, React hooks (`useState`, `useEffect`, etc.), or DOM event handlers
are required.
*Violation:* a page that fetches static data at build time using `'use client'`
with a `useEffect` instead of an `async` server component.

### R15 — Centralized API and WebSocket layer
HTTP calls use `lib/api.ts` helpers (`authHeaders`, configured fetch wrapper).
WebSocket connections use the socket helpers in `lib/` (`chat-socket.ts`,
`training-socket.ts`). Don't inline raw `fetch()` or `new WebSocket()` in
components.
*Violation:* a chat component calling `fetch("/api/...")` directly instead of
going through `lib/api.ts`; a component creating its own WebSocket connection
from scratch.

### R16 — UI primitives from `components/ui/`
Use the prebuilt shadcn/ui components in `components/ui/` for buttons, inputs,
dialogs, cards, dropdowns, etc. Don't hand-roll equivalents with raw HTML and
ad-hoc Tailwind classes.
*Violation:* a `<button>` with custom Tailwind classes scattered across the
codebase when `components/ui/button.tsx` already exists; a custom dialog built
with `<div>` + state when `components/ui/alert-dialog.tsx` already exists.

### R17 — One component per file, explicit props
Each file exports one primary component. Props are defined as a dedicated
TypeScript `interface` or `type` (named, not inlined in the function signature
when more than 2–3 fields).
*Violation:* a file exporting two unrelated components; props inlined as
`(props: { a: string; b: number; c: boolean })` instead of a named interface.

### R18 — Strict TypeScript, no `any`
Every function and component is fully typed. `any` is forbidden unless a
comment justifies why the type is genuinely unreachable (rare — almost always
`unknown` + type guard is the right choice instead).
*Violation:* a prop typed as `any`; a `@ts-ignore` without a comment explaining
the limitation.
