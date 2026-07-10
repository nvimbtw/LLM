---
id: ISSUE-0028
title: Widespread inline props types instead of named interfaces (R17)
scope: frontend
priority: P3
status: todo
rule: R17
location: frontend/components/chat/chat-thread.tsx:12 frontend/components/chat/chat-message.tsx:8 frontend/components/chat/chat-composer.tsx:35 frontend/components/chat/rich-message.tsx:19,31 frontend/components/chat/flux-throbber.tsx:193 frontend/components/conversations/conversation-item.tsx:37 frontend/components/models/model-card.tsx:15 frontend/components/models/model-detail.tsx:37,49,58 frontend/components/usage/usage-stats.tsx:10,32 frontend/components/usage/usage-by-model.tsx:14 frontend/components/usage/usage-chart.tsx:33 frontend/components/training/tokenize-panel.tsx:26,37 frontend/components/training/train-panel.tsx:43,54 frontend/components/training/section.tsx:32 frontend/components/training/config-form.tsx:82
blocked-by: []
blocks: []
---

## Description
The convention (per CLAUDE.md) is to define named exported `interface`/`type` for
props, prefixed with the component name. Across ~20 component files, props are typed
inline in the function signature instead. Additionally, `rich-message.tsx` exports
two primary components (`RichMessage` and `RichContent`), violating "one primary
export per file." Convert to named exported interfaces and split multi-export files.

## Comments
_No comments yet._
