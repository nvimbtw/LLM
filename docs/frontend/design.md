# Slate design

Slate is an editorial, minimal LLM workspace with the quietness of a notebook.
Content, typography and meaningful state carry the hierarchy. Warm surfaces,
restrained corners and faint character texture give it identity. Controls feel
part of the page; their relationships stay clear without drawing a box or line
around every fact. This is the active design contract, settled in
[FEAT-6](https://discord.com/channels/1538659898383011961/1565025604586438841).
[Appearance](appearance.md) defines the tokens and implementation details.

## Page composition

Use `PageShell` and `PageHeader`. Collection pages share the dashboard board
width and title alignment, including Projects, Models and Conversations.
Technical resource details and operational workspaces use the board too.
Constrain prose and short forms inside that frame when line length needs it;
a few editable fields do not make a collection a centered reading page.
Chat retains its focused thread/composer layout; authentication retains its
separate composition. Do not add page-specific width overrides to fix a list.

Keep headings, content and actions aligned. Useful metadata can sit beside a
heading or action. Resource descriptions are authored content and remain
editable where the domain allows it. Omit generic introductory subtitles.

Admin uses queue/history beside a grouped, collapsible Workbench. Preserve its
usable data, queue operations and navigation as it responds to narrower widths.
Model detail uses the declared architecture to organize its measurements and a
compact training record. Do not spread unrelated facts across oversized tiles,
ruled bands or a sparse collection of label/value widgets. Do not draw a legacy
transformer for a modular architecture that does not declare that structure.
Benchmark presentation remains outside this redesign.

## Shared lists

Use the composable `ResourceList`, `ResourceRow`, `ResourceLink`,
`ResourceTitle`, `ResourceMetadata`, `ResourceFacts` and `ResourceActions` in
`components/ui/resource-list.tsx`. The list owns rhythm and interaction;
the domain supplies meaningful content, measurements and actions.

- Use **open** density for catalogues and **compact** density for operational
  queue/history rows. Keep domain differences explicit rather than adding
  unrelated mode flags to one universal item record.
- Resource names use the shared editorial hierarchy. Metadata stays secondary;
  operational rows can use interface type for quicker comparison.
- Keep the main destination generous and predictable. Secondary controls remain
  separate, correctly labelled targets; do not nest a button inside a link.
- Group with space, alignment and headings. Draw no separators between entries
  or around the list. Hover/focus can add a restrained neutral surface.
- Keep quiet actions accessible to keyboard and coarse pointers. An action must
  not depend on hover to exist or work. Preserve selection, confirmation,
  pagination, pending, empty and failure behavior.
- Use `EmptyState` for a bounded warm-clay character field, a clear empty title
  and a useful action. Supporting copy needs the same justification as elsewhere.

## Hairline rule

**Decorative rules are absent by default.** Do not add a border below a page or
section heading, between ordinary list items, around a statistics band, between
form columns, or around each summary value. Adding data does not authorize
adding separators. Solve hierarchy through composition, space and type.

Keep functional boundaries and marks:

- The existing sidebar frame and its header/footer boundaries.
- Persistent editor/pane boundaries and the settings/notification modal rail.
- Composer outlines, input boundaries, keyboard focus and invalid/selection cues.
- The single thin underline identifying a search field.
- Diagram connections, chart axes/data, progress indicators, semantic quote/tree
  rails and the functional exchange-navigation strokes.

One owner draws each retained boundary. Use its semantic token at one CSS pixel;
do not compose two edges at the same join or invent per-component border colours.
Horizontal page tabs have no underline or dot, including under the active tab.

## Titles, prose fields and searches

Editable identity and prose remain page-native. `EditableTitle` and
`Textarea variant="inline"` share the small **Edit** cue, visible only when that
field is hovered or keyboard-focused. Keep the editing caret, accessible labels,
clear read-only/permission distinctions and the existing Save/Cancel contract.
Do not add a persistent pencil, filled input, focus underline or ring to imitate
editability. The keyboard cue must remain visible. Ordinary outlined controls
retain their functional focus indication.

Use `SearchField` for searches: magnifier, accessible name and one thin underline.
A search must not look like editable page prose. Use a placeholder only when it
helps identify the input. [Editing](editing.md) defines draft/retry behavior.

## Tabs and menus

Horizontal tabs are plain labels with muted counts. Active text uses steel blue
and a small weight change, without a pill, underline, dot or raised tile.
Retain the vertical primary marker in settings-style modal navigation.
Use the accessible primitives and their keyboard behavior; a route link is not
an in-page tab.

Menus are compact neutral overlays with aligned labels and restrained icons.
Separate meaningful action groups with space. Keep destructive actions explicit
and theme-aware. Selects and overflow menus share the same surface language;
composer-related popovers use its softer corner exception. Do not create a new
menu style at a call site.

## Colour, surfaces and texture

Keep the existing palette and steel-blue primary. Use semantic roles in both
themes; palette hexes belong only in the palette layer of `globals.css`.

| Role | Meaning |
| --- | --- |
| Primary steel blue | Actions, active selection and focus |
| Warm ochre (`kind-model`, `kind-tokenize`) | Model/vocabulary identity |
| Muted mauve (`kind-chat`) | Conversation identity |
| Aqua (`kind-rl`) | Practice identity |
| Sage (`kind-benchmark`, `status-success`) | Benchmark identity or explicitly labelled success |
| Amber (`status-warning`) | Warnings and limits |
| Soft red (`destructive`) | Failure and destructive actions |
| Warm clay (`texture-warm`) | Decorative empty-state texture |

Domain accents belong in small labels or marks, not whole-page fills or coloured
body text. Pair status colour with a word or icon. A quiet empty state is not an
error. Authored content and ordinary headings keep neutral ink.

Floating menus, dialogs, tooltips and toasts use the shared crisp shadow:
**1px right, 2px down, zero blur and spread**. Light/dark opacity is token-owned.
Do not add glowing halos, diffuse shadows or bespoke elevation. Keep general
corners restrained. Composer-related surfaces and chat bubbles retain their
existing softer shapes.

Every page header and the sidebar Slate brand header use `SectionTexture`'s
faint grey character field. It also belongs in the composer's shared command/plan
upper area. Only empty states use its warm tone. Do not repeat it behind every
section heading, behind populated lists, or across ordinary whole-page content.
Keep it decorative, static, muted, clipped and unable to intercept interaction.
The existing landing mountain/wordmark and pointer treatment remain special.

## Composer and conversation navigation

The existing composer outline contains its editor and a shared textured upper
region for commands/plans. Do not stack another outlined drawer above it.
Model selection sits on the footer left, separate thinking selection immediately
right, and context usage on the right. Remove the redundant conversation title;
sidebar/archive renaming remains available. The model picker is searchable,
starts with published choices and offers all eligible models only to admins.
Checkpoint choices do not belong in this menu.

Retain upload/dictation, command ordering, plan status/editing and context
semantics. Visual changes do not authorize treating local plans as model state.
See [Composer](composer.md) and [Plans](composer-plan.md).

The exchange rail uses dense, short functional strokes. Proximity/focus smoothly
extends nearby strokes; the current exchange stays steel blue. A compact adjacent
neutral preview shows the user prompt and reply excerpt, with restrained corners
and the crisp shadow. Keep scroll tracking, keyboard navigation and viewport
bounds; reduced motion preserves information. Do not use the rejected numbered
margin/counter alternatives or a pale inverse speech-bubble tooltip.

## Feedback and interface copy

Use [Notifications](notifications.md) for failure explanations and event feedback.
The separate sidebar-footer bell opens the settings-style modal; its count means
retained, undismissed notifications. Opening does not clear it. Severity and
lifetime are separate. Retention is browser-local and account-scoped.

Remove standalone status dots, including navigation badges and dots beside
already-labelled activity. Use concise state words, counts or actual progress.
The meaningful live throbber/diagram remains a separate figure with its existing
motion contract; do not substitute a new spinner at each call site.

**Default to no explanatory subtext.** Add secondary copy only when it helps the
current decision, identifies meaningful content or communicates a real current
state. Do not narrate background mechanics, reassure users that ordinary features
work, repeat a heading or add a subtitle to fill space.

| Remove | Keep when applicable |
| --- | --- |
| “Drafts save in this browser.” | Unsaved changes |
| “Manage all your models in one place.” | Paused |
| “Configure the settings below.” | Legacy |
| An automatic subtitle for every section | 2 waiting |

User-authored descriptions, units, consequential field hints, recovery actions
and genuine failure explanations remain useful. Put failure explanations in the
notification system instead of another boxed paragraph.

## Extending Slate

Choose an existing page/list/field/menu contract first. Add a shared variant only
for a demonstrated semantic difference. Keep storage, networking and domain
calculations out of presentation primitives. Check keyboard/touch, narrow widths,
light/dark, long content, pending/failure and reduced motion. Keep topic docs and
the module map current when a public behavior changes.

The original `frontend/slate-system-design.html` contains historical exploration;
it does not override this contract or authorize additional product features.
