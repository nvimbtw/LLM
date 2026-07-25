---
id: ISSUE-0085
type: issue
title: Sidebar conversations stale after login — ChatsProvider fetches only on mount, before auth
scope: frontend
priority: P1
status: done
rule: none
location: frontend/components/chat/chats-provider.tsx:62-64 frontend/hooks/use-login.ts:22-23
blocked-by: []
blocks: []
---

## Description

After logging in, past conversations don't appear until a manual page refresh.
`ChatsProvider` lives in the root layout (`app/layout.tsx:41`) and fetches once
on mount (`useEffect(() => { refresh() }, [refresh])`,
`chats-provider.tsx:62-64`). That mount happens on the login page, **before**
a token exists, so the fetch fails/returns empty. Login then does
`router.push("/")` (`hooks/use-login.ts:22-23`) — a client-side nav that keeps
the layout (and the provider's stale state) alive, so `refresh()` never re-runs.
The sidebar consumes the stale context at
`components/sidebar/nav-conversations.tsx:42`.

Shape of the fix: re-run the provider's `refresh()` when authentication becomes
available — e.g. have the login success path trigger the chats refresh (call it
after `setToken`, or key the provider's effect on the auth token so acquiring a
token refetches). Avoid the blunt fix of a full `window.location` reload.

Done when: logging in with an account that has past conversations shows them in
the sidebar immediately, with no manual refresh.

## Comments

_No comments yet._

### Verifier · 2026-07-17 · pass
Checked: (1) `lib/api.ts` — `setToken()`/`clearSession()` both dispatch `SESSION_CHANGED_EVENT`; (2) `chats-provider.tsx:62-76` — `useEffect` subscribes to the event, calls `refresh()` when token is present, clears list when absent; (3) `use-auth.ts:73` — `login()` calls `setToken()` → event fires before `router.push("/")`; (4) `bun run typecheck` — clean; (5) `bun run lint` — 15 errors, all pre-existing react-hooks violations (ISSUE-0026), none in changed files beyond the existing pattern. Fix correctly resolves stale sidebar conversations after login with no regression.

### Builder · 2026-07-17
Root cause: `ChatsProvider` (root layout) fetched once on mount — before a
token existed — and `router.push` after login never remounted it. Fix:
`lib/api.ts` now exposes `SESSION_CHANGED_EVENT` and dispatches it from
`setToken`/`clearSession` (a same-tab signal; the browser "storage" event only
fires in other tabs). The provider subscribes: token present → `refresh()`,
token cleared → drop the list (also fixes conversations lingering after
logout). Ran `bun run typecheck` (clean) and `bun run lint` — the 13 errors it
reports are all pre-existing react-hooks violations tracked by ISSUE-0026,
none in the changed lines.

### VCS · 2026-07-17 · `f69ca48` on `dev`
Committed ISSUE-0085. Staged: `frontend/lib/api.ts`, `frontend/components/chat/chats-provider.tsx`, `work/ISSUE-0085.md`. Left out other in-flight changes. Did not push.
