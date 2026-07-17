---
id: ISSUE-0074
type: issue
title: "R17 violations: multi-export files and inline props in frontend components"
scope: frontend
priority: P2
status: todo
rule: R17
location: frontend/components/training/stat-run.tsx:16 frontend/components/training/stat-run.tsx:38-48 frontend/components/chat/rich-message.tsx:16 frontend/components/chat/rich-message.tsx:28 frontend/components/models/model-edit-sheet.tsx:18-35 frontend/components/auth/auth-card.tsx:12-22 frontend/components/chat/flux-throbber.tsx:184-202
blocked-by: []
blocks: []
---

## Description

Six violations of **R17 (one component per file, explicit props)** in the frontend:

### Multi-export files

1. **`components/training/stat-run.tsx`** exports three components: `StatRun` (line 16), `Figure` (line 38), and `FigurePlaceholder` (line 63). `Figure` and `FigurePlaceholder` are imported independently by `telemetry.tsx`.

2. **`components/chat/rich-message.tsx`** exports two components: `RichMessage` (line 16) and `RichContent` (line 28). `RichContent` is imported independently by `chat-message.tsx`.

### Inline props (3+ fields, no named interface)

3. **`stat-run.tsx:38-48`**: `Figure` has 4 inline props (`caption`, `aside`, `minHeight`, `children`) — no named interface.

4. **`components/models/model-edit-sheet.tsx:18-35`**: `ModelEditSheet` has **7 inline props** (`open`, `onOpenChange`, `initialName`, `initialDescription`, `initialFamily`, `families`, `onSave`) — no named interface. This is the most egregious; 7 props inlined makes the signature hard to read.

5. **`components/auth/auth-card.tsx:12-22`**: `AuthCard` has 4 inline props (`title`, `description`, `children`, `footer`) — no named interface.

6. **`components/chat/flux-throbber.tsx:184-202`**: `FluxThrobber` has **8 inline props** (`mode`, `size`, `width`, `height`, `className`, `speed`, `direction`, `colors`) — no named interface. This is the second most egregious after `ModelEditSheet`.

### Shape of the fix

**For multi-export files:**
- Split `stat-run.tsx`: move `Figure` and `FigurePlaceholder` into their own files (`components/training/figure.tsx` and `components/training/figure-placeholder.tsx`, or a single `components/training/figure.tsx` if `Figure` is the primary and `FigurePlaceholder` is a sub-component).
- Split `rich-message.tsx`: move `RichContent` into `components/chat/rich-content.tsx`.

**For inline props:**
- Define a named `interface` for each component with 3+ inline props, exported and prefixed with the component name:
  - `FigureProps` for `Figure`
  - `ModelEditSheetProps` for `ModelEditSheet`
  - `AuthCardProps` for `AuthCard`
  - `FluxThrobberProps` for `FluxThrobber`

### Verification
- Each file exports exactly one primary component
- No component with 3+ props uses inline type annotations
- `bun run build` (or `bun run typecheck`) passes
- All imports updated to reflect file splits

## Comments
_No comments yet._
