---
id: ISSUE-0026
title: Frontend lint errors — react-hooks violations across hooks and components
scope: frontend
priority: P2
status: todo
rule: none
location: frontend/hooks/use-auth.ts:50 frontend/hooks/use-chat.ts:35 frontend/hooks/use-mobile.ts:14 frontend/hooks/use-model.ts:42 frontend/hooks/use-models.ts:106 frontend/hooks/use-usage.ts:44 frontend/components/chat/chat-composer.tsx:44,92,102 frontend/components/chat/chats-provider.tsx:63 frontend/components/chat/flux-throbber.tsx:219 frontend/components/models/model-edit-sheet.tsx:45
blocked-by: []
blocks: []
---

## Description
`bun run lint` reports 12 errors and 1 warning. 9 errors are
`react-hooks/set-state-in-effect` (calling `setState` synchronously in `useEffect`
body, which triggers cascading renders) across `use-auth`, `use-chat`, `use-mobile`,
`use-model`, `use-models`, `use-usage`, `chats-provider`, `flux-throbber`,
`model-edit-sheet`. 3 errors are `react-hooks/refs` in `chat-composer` (mutating
`ref.current` during render instead of in a `useEffect`). 1 warning is unused
variable `_details` in `use-auth.ts:82`. Fix per React 19 best practices.

## Comments
_No comments yet._
