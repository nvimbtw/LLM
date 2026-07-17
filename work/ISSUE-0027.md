---
id: ISSUE-0027
type: issue
title: Centralized API/WebSocket layer and UI primitive violations (R15, R16)
scope: frontend
priority: P2
status: todo
rule: R15, R16
location: frontend/components/chat/chats-provider.tsx:50,62,81,101,117 frontend/hooks/use-signup.ts:21 frontend/hooks/use-auth.ts:56 frontend/components/chat/rich-message.tsx:176
blocked-by: []
blocks: []
---

## Description
R15 violations: `chats-provider.tsx` has 4 inline `fetch` calls directly in a
component file instead of through hooks or `lib/api.ts` helpers; `use-signup.ts` and
`use-auth.ts` use raw `fetch` with manually-constructed headers instead of
`authHeaders()` from `lib/api.ts`. R16 violation: `rich-message.tsx:176` uses a
plain `<button>` with ad-hoc Tailwind classes instead of the `Button` component from
`components/ui/button`.

## Comments
_No comments yet._
