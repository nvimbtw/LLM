---
id: ISSUE-0087
type: issue
title: Long unbroken message lines overflow and clip the page — no word breaking in the message render path
scope: frontend
priority: P2
status: done
rule: none
location: frontend/components/chat/rich-message.tsx:36 frontend/components/chat/chat-message.tsx:17-25 frontend/components/chat/chat-thread.tsx:33
blocked-by: []
blocks: []
---

## Description

A model reply containing a very long token with no whitespace (long URL,
unbroken string) pushes the layout wider than the viewport and clips the site
horizontally. Nothing in the message render path breaks words:

- `rich-message.tsx:36` root wrapper (`space-y-3 text-sm leading-relaxed`),
  paragraphs (`:96`), inline code (`:102-108`), and links (`:130-140`) all lack
  `break-words`/`overflow-wrap`.
- `chat-message.tsx:17` caps the user bubble at `max-w-[85%]` but the inner
  content still overflows it.
- The code block `<pre>` (`rich-message.tsx:185`) correctly has
  `overflow-x-auto`, but the flex column parents lack `min-w-0`
  (`chat-thread.tsx:33`, `chat-message.tsx:25`), so it can still force the
  column wider instead of scrolling internally.

Shape of the fix: add `break-words` (overflow-wrap) at the message content
root(s) so prose wraps, and `min-w-0` on the flex column parents so the code
block's `overflow-x-auto` actually contains it. Prose breaks mid-word rather
than clipping; code scrolls inside its own box.

Done when: a message with a 500-character unbroken string and a wide code block
renders with no horizontal page scroll — the string wraps, the code block
scrolls internally.

## Comments

_No comments yet._

### Builder · 2026-07-17
Three layers, matching the described gaps: (1) `rich-message.tsx` — root
wrapper gets `break-words` so prose wraps mid-word; inline code and links get
`break-all` (URLs/identifiers have no natural break points). (2)
`chat-message.tsx` — `min-w-0` on both the user bubble and the assistant
wrapper so content can't push past `max-w-[85%]`/the column. (3)
`app/layout.tsx` — `min-w-0` on `SidebarInset`: it is a flex-ROW item under
`SidebarProvider`, and without it the min-content width of wide message
content propagated to the page, which is what actually clipped the site; the
code block's existing `overflow-x-auto` now contains it. Ran
`bun run typecheck` (clean); lint unchanged (pre-existing ISSUE-0026 errors
only).

### Verifier · 2026-07-17 · pass

Changes confirmed at all three layers:
- `rich-message.tsx:36` — `break-words` on root; `:106` `break-all` on inline
  code; `:136` `break-all` on links.
- `chat-message.tsx:17` — `min-w-0` on user bubble; `:25` `min-w-0` on assistant
  wrapper.
- `app/layout.tsx:51` — `min-w-0` on `SidebarInset`, breaking the flex-row
  min-content propagation chain that caused page-level clipping.

Checked: `tsc --noEmit` clean; lint only pre-existing ISSUE-0026 errors (none
in touched files). No new rule violations. The code block's existing
`overflow-x-auto` is now contained by the `min-w-0` ancestors as described.

### VCS · 2026-07-17 · `dc3a89a` on `dev`
