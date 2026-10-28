# Composer plan

The task list is part of `ChatComposer`, through its optional `plan` prop.
`lib/plan.ts` defines the presentation snapshot: a plan id, title, revision,
and ordered tasks with stable ids. The supplied order decides pending-task
priority; the active task appears first and finished tasks last. The component
does not change task status or infer completion from the end of a reply.

## Presentation

`components/chat/composer-plan.tsx` shows the first three unfinished tasks.
Completing a task gives its strike-through a short beat before it slides down
out of the preview; subsequent rows move into place. Completed and cancelled
tasks remain in the full list with distinct marks. `sending` controls the
active task's warm mark; an interrupted turn shows it as paused.

The preview and full disclosure occupy the composer's shared textured upper
region within its outline. Hover, click, keyboard focus and touch expand the
full list naturally; Escape folds it. Expanded content scrolls within a
viewport-relative cap. Inactive copies are inert and hidden from assistive
technology. Reduced motion removes movement while retaining task states.

Slash command discovery temporarily replaces the plan region and restores it
when the menu closes, including any open editing draft. An all-completed plan
keeps a compact completion header; a plan containing cancellations says finished
rather than claiming every task succeeded. Neither replies nor disclosure
transitions change task status. Motion lives in `components/chat/composer-plan.css`.

## User editing

`/plan add <text>` appends a pending task, creating a plan if needed.
`/plan edit` opens a draft with inline titles, removal, and an add row.
Drag a handle or use its arrow keys to reorder within a status group: current
stays first and finished tasks stay last. `/plan add` during editing changes
only the draft. `/plan save` or Save plan validates and commits it; Cancel
discards it. A changed source revision or a storage failure keeps the draft
open and reports the error rather than silently losing edits.

Live conversations store saved plans locally in this browser, keyed by chat
id. A new conversation holds its plan in memory until chat creation transfers
it to that key. Unsaved drafts do not survive navigation. These are frontend
plans, not server records or model context; no backend or model sync is implied.

## Preview and current boundary

`/markup#composer-plan` renders the real composer and list underneath the
existing markup conversation. `hooks/use-plan-preview.ts` supplies local
fixtures for working, paused, complete, long, and absent plans. Complete next
advances one step; Replay advances automatically and stops at completion.
Sending text echoes it locally rather than starting model generation.

Below it, the existing `Trace` component shows a `todo` call running, saved,
and failed. `lib/tools.ts` owns its English labels and argument summary, with
the raw argument retained as the fallback for an unknown or malformed shape.

The preview supports the same editing commands without persisting its fixtures.
Entering editing stops Replay; scenario controls stay disabled until the draft
is saved or cancelled. Backend synchronization, model training, and a gym
remain later work.
