---
name: tester
description: Test application behavior or changes with specialist subagents and report results in Discord. Use when an independent testing pass is requested.
---

# Tester

Read and follow [the shared review workflow](../review-workflow.md). Apply it to
the requested scope before choosing specialists. Assign relevant behavior and
test layers, such as unit, integration, cross-module contracts, frontend flows,
and failure or recovery scenarios. Divide ownership so specialists do not
interfere with each other's test resources.

Read applicable module documentation, test instructions, callers and existing
tests. Identify expected behavior and risks, then run the relevant documented
test commands and targeted checks. Exercise meaningful boundaries and failure
transitions, including cancellation, retries and concurrency where applicable.
Use temporary tests or isolated test data when needed to investigate a gap;
do not add permanent tests or change application code unless separately asked.
Avoid destructive operations or shared production data; respect test resource
budgets and clean up resources created by the run.

Reproduce failures where practical and distinguish product defects, test defects,
environment failures and unverified hypotheses. Report commands, environment,
expected and actual behavior, reproduction steps, relevant locations, and
passed, failed, skipped or blocked checks. Describe consequential coverage gaps
and suggested tests. A passing build or unit suite does not establish that
integration or end-to-end behavior was tested.
