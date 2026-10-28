# Private projects

`/projects` lists the signed-in account's projects. New project opens
`/projects/new-project`, where the user fills in the page's editable fields.
`/projects/{id}` presents one project's name, description, notes and instructions,
conversation list and real new-chat composer. Project data is private to its owner;
the browser loads it through Rust using the actual bearer session identity.

The sidebar's Projects navigation item sits directly below Conversations. A flat
section of up to eight project links appears above recent conversations, with an
All projects link when more exist. Project rows share conversation rows' text
alignment and hover/selected marker. The five most recent conversations appear
regardless of project membership. The full `/conversations` archive includes every
chat and exposes membership management alongside rename and delete.

Active chat threads have no Conversation / Add to project header. Manage their
project membership from the conversation archive; the thread stays devoted to its
messages and composer.

## Ownership and lifecycle

[`lib/projects.ts`](../../frontend/lib/projects.ts) validates interpreted project
metadata, text limits, pagination and thread membership at the HTTP boundary.
[`lib/chats.ts`](../../frontend/lib/chats.ts) adapts chat summaries, preserving
compatibility with older responses that omit the project field or token total.
The browser sends no project prompt or memory objects; Rust coordinates project
notes and relevant past-message retrieval when a turn is sent.

[`ProjectsProvider`](../../frontend/components/projects/projects-provider.tsx)
coordinates project mutations with the existing chat archive. Its
[`ProjectCatalog`](../../frontend/lib/project-catalog.ts) owns abortable reads and
paginated private rows; account replacement immediately hides earlier rows and
ignores obsolete completions. The index reads 50 projects per page. Individual
project views read 50 conversations per page and return to the last existing page
if moving or deleting chats empties a later page.

Project names are nonblank and at most 120 Unicode characters; descriptions allow
2,000 characters and notes allow 8,000. Title, description and notes are directly
editable on both the creation and existing-project pages; no metadata modal is
used. Fields retain the page's heading/prose style with no resting border or note
box. Create project or Save changes explicitly persists all three fields; failed
saves preserve the draft. Existing projects expose Cancel and an unsaved state,
and [tab-local recovery](editing.md#draft-recovery) preserves edits while browsing.
The privacy/conversation-count line sits underneath the
description, and Delete project appears beside New chat in the page header. Deleting
a project preserves its conversations and messages, removes their association and
refreshes both project navigation and the chat archive. Moving a chat uses the same
identity and history; selecting No project removes its association. Membership
changes do not affect eligibility for the recent conversations sidebar.

## Composer and session boundaries

New chat on a project page reveals the existing `ChatComposer`, with the account's
real model/checkpoint and thinking-budget picker, attachments and local plans.
[`useStartChat`](../../frontend/hooks/use-start-chat.ts) passes the owning project
identity when creating the conversation. The ordinary New navigation action and
home composer omit it. Uploaded attachment rows, the opening text and the exact
chosen model and budget use the same one-shot navigation handoff in
[`PendingChatMessages`](../../frontend/lib/pending-chat.ts); account replacement
discards pending messages from the previous session.

The active conversation boundary is keyed by bearer session and chat identity.
Switching accounts or chats disposes its sockets, draft and local attachment view.
Composer-choice reads and writes also belong to the actual token; default values
cannot overwrite a saved choice before its load settles. A late 401 can clear
only the session that authorized its request.

Project forms preserve failed drafts for retry. Failed reads use `LoadError`, and
failed submissions use `FormError`. Changing account or leaving a form prevents
its obsolete completion from closing a replacement dialog or navigating another
account.

## Verification

From `frontend/`, run `bun run typecheck`, `bun run lint` and `bun test`. Focused
tests in `lib/projects.test.ts`, `lib/project-catalog.test.ts`,
`lib/project-session.test.ts` and `lib/pending-chat.test.ts` cover wire validation,
field budgets, pagination, ownership races, cancellation, stale 401 responses,
membership writes and first-message handoff. Production builds remain a separate
check, as do real browser interactions and backend/PostgreSQL integration.
