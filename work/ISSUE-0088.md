---
id: ISSUE-0088
type: issue
title: Composer scrolls away in long conversations — unbounded height chain lets the document scroll instead of the thread
scope: frontend
priority: P2
status: done
rule: none
location: frontend/components/ui/sidebar.tsx:141 frontend/app/layout.tsx:42-47 frontend/components/chat/chat-thread.tsx:31-32
blocked-by: []
blocks: []
---

## Description

Scrolling up through a past conversation scrolls the input composer off-screen.
The composer is already correctly outside the thread's scroll container
(`chat-thread.tsx:77`), but the height chain above it is unbounded: the
`SidebarProvider` wrapper is `min-h-svh` (grow-only, `ui/sidebar.tsx:141`) and
`SidebarInset` is `flex-1` with no fixed height (`ui/sidebar.tsx:310`), so
`ChatThread`'s `flex-1 overflow-y-auto` (`chat-thread.tsx:31-32`) never gets a
bounded height. The inner scroll area never activates; the **document body**
scrolls instead, taking the composer with it.

Shape of the fix: bound the height chain so the inner `overflow-y-auto` takes
over — e.g. make the chat layout wrapper `h-svh`/`h-dvh` + `overflow-hidden`
(root `app/layout.tsx:42-47`), or constrain `SidebarInset` there. Prefer
bounding at the layout usage site over editing the shadcn `ui/sidebar.tsx`
primitive; if the primitive must change, verify every other page (training,
conversations, files, usage, profile) still scrolls correctly, since
`min-h-svh` is global.

Done when: in a conversation longer than the viewport, the composer stays
pinned at the bottom while the messages scroll, on the chat page and with no
scrolling regression on the other pages.

## Comments

_No comments yet._

### Builder · 2026-07-17
Root cause confirmed: no ancestor bounded the height (`SidebarProvider` is
`min-h-svh`, grow-only), so ChatThread's `flex-1 overflow-y-auto` resolved
against content and the document body scrolled. Fix in `app/layout.tsx` at the
usage site (shadcn `ui/sidebar.tsx` untouched): `SidebarInset` is now
`h-svh overflow-hidden`, and `{children}` is wrapped in
`<main class="flex min-h-0 flex-1 flex-col overflow-y-auto">`. On /chat the
thread's own scroll area activates and the composer stays pinned; every other
page (none set their own heights — verified by grep) scrolls inside <main>
instead of the body, visually unchanged. Ran `bun run typecheck` (clean) and
`bun run build` (succeeds).

### Verifier · 2026-07-17 · pass

Confirmed fix in `app/layout.tsx`: `SidebarInset` is `h-svh overflow-hidden`
(line 47), bounding the height chain. `<main>` is `flex min-h-0 flex-1 flex-col
overflow-y-auto` (line 51), hosting the scroll. `SidebarProvider` still has
`min-h-svh` (line 141) — primitive untouched. Chat thread's inner
`overflow-y-auto` (line 44) now has a bounded height and activates correctly.
All other pages (conversations, training, files, usage, profile, models) use
simple `flex-1` divs with no custom overflow — they scroll inside `<main>`
instead of the body, visually identical. `bun run build` succeeds with zero
warnings.

### VCS · 2026-07-17
Committed at `e769d50` on branch `dev`. Staged only `frontend/app/layout.tsx`
and `work/ISSUE-0088.md`.
