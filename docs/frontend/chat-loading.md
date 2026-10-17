# Chat loading

Hovering or focusing a sidebar conversation prefetches both its Next.js route
and its messages. Route prefetch alone cannot warm a client hook's API request.
`lib/chat-history.ts` shares a pending request with `useChat`, and a completed
prefetch supplies the first render so the thread opens without a loading flash.
The same response supplies the open thread's title and token total. The hook
still folds stored rows through the same tool-turn parser.

The signed-in frame initially requests only five recent conversation summaries.
The full archive and its token totals load when `/conversations` opens or the
project conversation picker opens. Mutations refresh the recent slice and any
archive already opened in the current session. Account changes keep responses
from the previous account out of both lists.

The in-memory cache holds at most ten conversations for thirty seconds. It is
cleared on session changes, including logout and cross-tab storage events, and
checked against the current token before every read. It never persists messages
in browser storage. A failed speculative request is discarded silently; opening
the conversation retries and reports a normal load error if that fails too.
Sending, rating a message, and deleting a chat invalidate its entry. A completed
turn always fetches fresh persisted messages, bypassing the prefetched snapshot.

The composer reserves the editor's normal minimum height even before Tiptap
mounts. Skeletons occupy the editor, model trigger, and context-meter positions
while their data is unavailable; their animation respects reduced motion.
The footer keeps its minimum row height when a model has no context window.

Account choice readiness is exposed by `ComposerChoiceProvider`. The picker
waits for both that choice and the model list before correcting a missing or
stale model id, then also waits for checkpoint-parent resolution. This prevents
a fast model-list response from replacing a saved choice still arriving.
Failures settle the loading state and allow the existing fallback selection.
Drafts remain editable while loading, but sending waits for a loaded model
choice. Standalone slash commands can still run locally during that wait.

Verify cache reuse, expiry, invalidation, retries, and session isolation with
`cd frontend && bun test lib/chat-history.test.ts`.
