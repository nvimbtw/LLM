---
id: ISSUE-0075
type: issue
title: "Raw fetch() calls in chats-provider.tsx instead of using centralized API helpers"
scope: frontend
priority: P2
status: todo
rule: R15
location: frontend/components/chat/chats-provider.tsx:50 frontend/components/chat/chats-provider.tsx:81 frontend/components/chat/chats-provider.tsx:101 frontend/components/chat/chats-provider.tsx:117
blocked-by: []
blocks: []
---

## Description

`components/chat/chats-provider.tsx` calls raw `fetch()` directly in four places instead of using a configured API helper from `lib/api.ts`. Violates **R15 (Centralized API and WebSocket layer)**.

While the file correctly imports `authHeaders()` from `@/lib/api`, it still calls bare `fetch()` with manual URL construction instead of going through a configured wrapper. This couples the component to:
- The hard-coded `API_URL` constant (imported from `lib/api.ts` but repeated in every call)
- Error-handling logic that should live in one place
- The `authHeaders()` boilerplate on every call

### Affected calls

| Line | Call |
|------|------|
| 50 | `fetch(\`${API_URL}/chat\`, { headers: authHeaders() })` — GET list |
| 81 | `fetch(\`${API_URL}/chat\`, { method: "POST", headers: authHeaders(), body: ... })` — create chat |
| 101 | `fetch(\`${API_URL}/chat/${id}\`, { method: "PUT", headers: authHeaders(), body: ... })` — rename chat |
| 117 | `fetch(\`${API_URL}/chat/${id}\`, { method: "DELETE", headers: authHeaders() })` — delete chat |

### Shape of the fix

The fix has two layers:

1. **Add a configured fetch wrapper to `lib/api.ts`** (if one doesn't already exist) that centralizes:
   - Base URL resolution (`API_URL`)
   - Auth header injection (`authHeaders()`)
   - Error handling (non-2xx responses, network errors)
   - Content-Type header
   - JSON serialization/deserialization

   Something like:
   ```ts
   export async function apiFetch<T>(
     path: string,
     options?: RequestInit,
   ): Promise<T> { ... }
   ```

2. **Replace raw `fetch()` calls in `chats-provider.tsx`** with the `apiFetch` wrapper. See the existing `hooks/` files (`use-chat.ts`, `use-models.ts`) for the pattern — they already use `authHeaders()` but would also benefit from a wrapper; this issue only concerns the component file (`chats-provider.tsx`).

### Verification
- `chats-provider.tsx` contains no raw `fetch()` calls
- All four HTTP operations (list, create, rename, delete) still work
- `bun run build` passes

## Comments
_No comments yet._
