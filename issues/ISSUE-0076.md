---
id: ISSUE-0076
title: "Raw HTML elements in business components instead of shadcn/ui primitives"
scope: frontend
priority: P3
status: todo
rule: R16
location: frontend/components/training/preset-controls.tsx:97 frontend/components/training/appendix.tsx:29 frontend/components/chat/rich-message.tsx:176 frontend/components/files/file-dropzone.tsx:61
blocked-by: []
blocks: []
---

## Description

Four instances of raw `<button>` and `<input>` HTML elements in business components instead of using the prebuilt shadcn/ui primitives from `components/ui/`. Violates **R16 (UI primitives from components/ui/)**.

### Locations

| File | Line | Element | Suggested replacement |
|------|------|---------|---------------------|
| `components/training/preset-controls.tsx` | 97 | `<button type="button" aria-label="Delete..." ...>` | `<Button variant="ghost" size="icon">` |
| `components/training/appendix.tsx` | 29 | `<button type="button" onClick={() => setOpen(...)} ...>` | `<Button variant="ghost">` |
| `components/chat/rich-message.tsx` | 176 | `<button type="button" onClick={copy} ...>` | `<Button variant="ghost" size="sm">` |
| `components/files/file-dropzone.tsx` | 61 | `<input type="file" ... className="sr-only">` | `<Input type="file" className="sr-only">` |

### Why it matters

Using raw HTML elements bypasses the project's design system. The shadcn/ui `Button` component provides consistent styling (focus rings, active states, disabled appearance, size variants, loading state) that raw `<button>` lacks. A global style change (e.g., button border radius or font) only needs to update `components/ui/button.tsx`, not every raw `<button>` scattered across the codebase.

### Shape of the fix

Replace each raw element with its shadcn/ui counterpart:
1. In `preset-controls.tsx`: import `Button` from `@/components/ui/button` and replace `<button ...>` with `<Button variant="ghost" size="icon" aria-label={...} onClick={...}>`
2. In `appendix.tsx`: replace `<button ...>` with `<Button variant="ghost" ...>`
3. In `rich-message.tsx`: replace `<button ...>` with `<Button variant="ghost" size="sm" ...>`
4. In `file-dropzone.tsx`: import `Input` from `@/components/ui/input` and replace `<input type="file" ...>` with `<Input type="file" className="sr-only" ...>`

For the file-dropzone's visually-hidden file input, verify that the sr-only Tailwind class works correctly with the shadcn `Input` component (the `<Input>` might need `ref` forwarding which it already supports).

### Verification
- No raw `<button>` or `<input>` elements remain in business components under `components/` (excluding `components/ui/` itself)
- `bun run build` passes
- Visual appearance is consistent with the rest of the app (verify button styling, focus rings)
- File dropzone still works (hidden input still triggers file picker)

## Comments
_No comments yet._
