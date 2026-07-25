---
id: ISSUE-0086
type: issue
title: Streaming chat is jumpy — smooth scrollIntoView per token and throbber height swaps
scope: frontend
priority: P2
status: done
rule: none
location: frontend/components/chat/chat-thread.tsx:26-28 frontend/components/chat/chat-thread.tsx:52-69
blocked-by: []
blocks: []
---

## Description

While the model streams, the page visibly jumps every time output wraps to a
new line. Two causes in `chat-thread.tsx`:

1. `bottomRef.scrollIntoView({ behavior: "smooth" })` runs on **every**
   `messages` change (lines 26-28) — and every streamed token mutates the last
   message (`hooks/use-chat.ts:67-81`), so overlapping smooth-scroll animations
   fire many times per second.
2. The three mutually-exclusive throbber blocks (lines 52-69) have different
   heights and wrappers (`h-8` thinking, `h-8` generating, `h-11` idle), so the
   thinking → generating → idle transitions reflow the column mid-stream.

Shape of the fix: during streaming, pin to the bottom with instant scrolling
(`behavior: "auto"`, or set `scrollTop` directly on the scroll container) and
only auto-scroll when the user is already at/near the bottom — never
smooth-animate per token. Reserve a fixed-height slot for the throbber states
so swapping them doesn't change layout height.

Done when: a long streamed reply scrolls smoothly line-by-line with no visible
jumping, and the throbber transitions don't shift the messages above them.

## Comments

_No comments yet._

### Builder · 2026-07-17
Two changes in `chat-thread.tsx`: (1) replaced the per-update
`scrollIntoView({behavior:"smooth"})` with direct `scrollTop` assignment on
the scroll container (instant — overlapping smooth animations were the
judder), gated behind a 120px near-bottom threshold
(`STICK_TO_BOTTOM_THRESHOLD_PX`) so auto-scroll disengages when the user
scrolls up to read. (2) merged the three mutually-exclusive throbber blocks
into one fixed-height (`h-11`) slot so thinking → generating → idle swaps no
longer change layout height. Ran `bun run typecheck` (clean); lint reports no
issues in this file.

### Verifier · 2026-07-17 · pass
Checked `chat-thread.tsx` lines 9–10 (threshold constant), 33–40 (instant
scroll gated behind 120px threshold), and 64–79 (single fixed `h-11` throbber
slot). Scroll fix replaced both smooth animations and per-token thrashing;
throbber states no longer reflow the column. `bun run typecheck` clean; lint
clean on the file. No regressions. Described problems resolved.

### VCS · 2026-07-17
`e63e2c1` on `dev`
