---
name: performance-review
description: Review application or change performance using specialist subagents, measurements, and a Discord report. Use when a performance review is requested.
---

# Performance review

Read and follow [the shared review workflow](../review-workflow.md). Apply it to
the requested scope before choosing specialists. Partition work by relevant
performance risks, such as hot-path algorithms and data structures, database and
network I/O, memory and concurrency, frontend rendering, or model execution.
Choose areas based on the actual application and workload, not a fixed roster.

Trace likely hot paths and resource costs against realistic inputs and execution
frequency. Run appropriate existing benchmarks, tests and profilers when feasible;
create temporary benchmarks when they can meaningfully test a hypothesis. Use
bounded, safe workloads and preserve user data and the working tree. Record
commands, workload, environment, baseline and results, including variance when
relevant. Never present an estimated improvement as a measured one. Explain
when infrastructure, time, dependencies or representative workloads prevent
measurement, and report the finding as a hypothesis with a way to validate it.
