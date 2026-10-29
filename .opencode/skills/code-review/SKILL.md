---
name: code-review
description: Review application or change correctness and maintainability using specialist subagents and a Discord report. Use when a code review is requested.
---

# Code review

Read and follow [the shared review workflow](../review-workflow.md). Apply it to
the requested scope before choosing specialists. Partition work by relevant
concerns, such as correctness and failure handling, readability and design,
data structures and algorithms, state and concurrency, tests and contracts,
or security boundaries. Choose areas based on the code, not a fixed roster.

Review behavior in context, including callers, tests and applicable documented
contracts. Identify concrete defects, maintainability obstacles and unsuitable
data structures with their practical consequences. Run focused checks where
they can resolve uncertainty. Avoid reporting preference or convention alone
as a defect without explaining its consequence or the applicable project rule.
