# Slash Commands & Fuzzy Finder

- **Date**: 2026-08-18
- **Status**: Implemented

## Context

Users adjust session parameters (model choice, thinking budget, sampling temperature, top-k, top-p, system steering prompts, theme) frequently while chatting. Previously, some were toggled only in the ⌘K settings modal while thinking budget and model selection had dedicated dropdowns.

## Decision

Provide a unified, keyboard-driven slash command system with an Editorial Minimalist fuzzy finder popover floating above the composer:
1. Two-stage fuzzy search: command discovery (`/`) followed by parameter/argument suggestions (`/<command> <query>`).
2. Immediate inline feedback with range indicators for numeric knobs (`/temperature`, `/top-k`, `/top-p`), model family grouping for `/model`, and toggle states for `/show-thinking` / `/notify`.
3. Standalone execution: recognized slash commands run locally with instant toast feedback without dispatching messages to the model.

## Rejected alternatives

- *Modal-only settings*: Required disrupting chat flow to open the ⌘K modal for every prompt steering or parameter change.
- *Chatbot interpreted commands*: Sending slash commands over the wire to be interpreted by the LLM wastes tokens, adds latency, and creates unpredictable execution compared to client-side local interception.
