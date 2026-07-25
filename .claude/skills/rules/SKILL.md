---
name: rules
description: Index of the coding rules (R1–R21) every artifact is judged against. Load the full text of a cited rule from references/backend.md (Rust) or references/frontend.md (TS) — only the rules you need.
---

# Coding rules — index

The shared standard: the Auditor files ISSUEs against these, the Builder must
satisfy them, the Verifier bounces violations. They layer on top of
`backend/CLAUDE.md` and `frontend/CLAUDE.md`. Ids are stable; numbering is not
sectionally contiguous.

**Load full rule text on demand**: backend rules in
[`references/backend.md`](references/backend.md), frontend in
[`references/frontend.md`](references/frontend.md). For a single cited rule,
read just that rule's section — not the whole file.

## Backend (Rust) — references/backend.md

| Id | Rule |
|---|---|
| R1 | Bounded function length (~60-line body; FFI judged by logical steps) |
| R2 | Bounded loops — obvious, finite exit |
| R3 | Handle every fallible `Result`/`Option`; no bare `unwrap` |
| R4 | `unsafe` minimal and justified with a SAFETY comment |
| R5 | Smallest scope for data; no wide reused `mut` |
| R6 | No duplicated logic — factor shared blocks |
| R7 | Zero warnings (fix or justified `#[allow]`) |
| R8 | No magic numbers — named `const`s |
| R9 | Doc comments state non-obvious contracts only, never restate names |
| R10 | Validate at boundaries (lengths, shapes, ranges) |
| R11 | No panics on external input — return `Result` |
| R12 | Respect the layering (`routes`/`db`/`crypto`, one-directional crates) |
| R13 | Self-documenting naming/structure; comments say *why*, never *what* |
| R19 | *(folded into R13 — existing `rule: R19` citations judged under R13)* |
| R20 | Cohesive modules and honest crate boundaries; no grab-bag `utils` |
| R21 | Names don't stutter; every path segment adds meaning |

## Frontend (TypeScript/React) — references/frontend.md

| Id | Rule |
|---|---|
| R14 | Server-first rendering; `'use client'` only when required |
| R15 | Centralized API/WebSocket layer (`lib/api.ts`, socket helpers) |
| R16 | UI primitives from `components/ui/`, no hand-rolled equivalents |
| R17 | One component per file; named props interface |
| R18 | Strict TypeScript, no `any` |
