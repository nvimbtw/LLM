import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import {
  ValidationObserver,
  type ValidationTransport,
} from "./validation-observer"
import { decodeArchitecture, identityDigest } from "./decoder"
import { ArchitectureApiError } from "./http"
import { requireValidationReport } from "../model-validation"
import {
  jobFromValidation,
  jobHref,
  jobStatusLabel,
  formatClock,
} from "../jobs"
import type {
  ArchitectureContracts,
  ValidationReport,
  ValidationRun,
} from "./contracts.generated"

const cases = JSON.parse(
  readFileSync(
    new URL(
      "../../../contracts/modular-architecture/v1/api/cases.json",
      import.meta.url
    ),
    "utf8"
  )
).cases

/** Copies shared wire fixtures for synthetic lifecycle tests; these fixtures do not constitute numerical evidence. */
function fixture<K extends keyof ArchitectureContracts>(
  model: K,
  name = `${model}-positive`
): ArchitectureContracts[K] {
  return decodeArchitecture(
    model,
    structuredClone(
      cases.find((item: { name: string }) => item.name === name).value
    )
  )
}

/** Aligns the synthetic report's input identity with its synthetic queue request to exercise cross-response checks. */
function execution(status: ValidationRun["status"] = "running"): {
  run: ValidationRun
  report: ValidationReport
} {
  const run = fixture("ValidationRun")
  const report = fixture("ValidationReport")
  report.receipt!.identity.input_profile_digest = identityDigest({
    tensor_inputs: run.config.tensor_inputs,
    domains: run.config.profile!.domains,
  })
  report.receipt!.identity_hash = identityDigest(report.receipt!.identity)
  run.status = status
  run.result = status === "done" ? report : null
  return { run, report }
}

/** Supplies harmless default transports, with individual tests overriding actual observed lifecycle/effect seams. */
function transport(
  run: ValidationRun,
  report: ValidationReport,
  overrides: Partial<ValidationTransport> = {}
): ValidationTransport {
  return {
    run: async () => run,
    report: async () => report,
    cancel: async () => {},
    retry: async () => {
      throw new Error("Unconfigured retry")
    },
    remove: async () => {},
    ...overrides,
  }
}

test("concurrent reads coalesce; disposal aborts observation and ignores late state without cancelling work", async () => {
  const { run, report } = execution()
  let complete!: (value: ValidationRun) => void
  let readSignal: AbortSignal | undefined
  let calls = 0
  let cancellations = 0
  const observer = new ValidationObserver(
    run.run_id,
    transport(run, report, {
      run: (signal) => {
        calls++
        readSignal = signal
        return new Promise((resolve) => {
          complete = resolve
        })
      },
      cancel: async () => {
        cancellations++
      },
    })
  )
  const first = observer.refresh()
  assert.equal(observer.refresh(), first)
  const before = observer.getSnapshot()
  observer.dispose()
  complete(run)
  await first
  assert.equal(calls, 1)
  assert.equal(cancellations, 0)
  assert.equal(readSignal?.aborted, true)
  assert.equal(observer.getSnapshot(), before)
})

test("terminal socket hints reconcile queued/running lifecycle; only the report endpoint publishes an authoritative verdict", async () => {
  const { run, report } = execution("queued")
  let row = run
  let reports = 0
  let release!: (value: ValidationReport) => void
  const observer = new ValidationObserver(
    run.run_id,
    transport(run, report, {
      run: async () => structuredClone(row),
      report: async () => {
        reports++
        return new Promise((resolve) => {
          release = resolve
        })
      },
    })
  )
  try {
    await observer.refresh()
    assert.equal(reports, 0)
    row = { ...run, status: "running" }
    await observer.refresh()
    row = { ...run, status: "done", result: report }
    observer.event(fixture("ValidationEvent"))
    const pending = observer.refresh()
    await Promise.resolve()
    assert.equal(observer.getSnapshot().report, null)
    release(report)
    await pending
    assert.equal(observer.getSnapshot().report?.verdict, "valid")
    assert.equal(observer.getSnapshot().run?.status, "done")
    assert.equal(reports, 1)
  } finally {
    observer.dispose()
  }
})

test("missing publication does not promote the queue result to a verified report", async () => {
  const { run, report } = execution("done")
  const observer = new ValidationObserver(
    run.run_id,
    transport(run, report, {
      report: async () => {
        throw new Error("Published report unavailable")
      },
    })
  )
  try {
    await observer.refresh()
    assert.equal(observer.getSnapshot().report, null)
    assert.match(observer.getSnapshot().error!, /unavailable/)
  } finally {
    observer.dispose()
  }
})

test("report pairing refuses a different run, source, tokenizer, numerical profile or synthetic input domain", () => {
  const { run, report } = execution("done")
  requireValidationReport(run, report)
  const changed = [
    { ...run, run_id: "30000000-0000-4000-8000-000000000099" },
    {
      ...run,
      config: {
        ...run.config,
        subject: { ...run.config.subject, content_hash: "b".repeat(64) },
      },
    },
    { ...run, config: { ...run.config, tokenizer: null } },
    {
      ...run,
      config: {
        ...run.config,
        profile: { ...run.config.profile!, numerical_seed: 5 },
      },
    },
    {
      ...run,
      config: {
        ...run.config,
        tensor_inputs: {
          hidden: { shape: [1, 2, 4], dtype: "float32" as const },
        },
      },
    },
  ]
  changed.forEach((row) =>
    assert.throws(
      () => requireValidationReport(row, report),
      /another execution|different source or input profile/
    )
  )
})

test("discovered strict export guards extend the prospective request while supplied guard expectations must match exactly", () => {
  const { run, report } = execution("done")
  report.receipt!.identity.profile.export_guards = [
    "eval/full_prefix/domain-0/part-0: singleton batch=1 sequence=1",
  ]
  report.receipt!.identity_hash = identityDigest(report.receipt!.identity)
  assert.doesNotThrow(() => requireValidationReport(run, report))
  run.config.profile!.export_guards = ["different expected capture"]
  assert.throws(
    () => requireValidationReport(run, report),
    /different source or input profile/
  )
})

test("stage replay ignores old sequences and keeps diagnostics bounded without trusting a terminal hint", () => {
  const { run, report } = execution()
  const observer = new ValidationObserver(run.run_id, transport(run, report))
  try {
    for (let sequence = 0; sequence < 100; sequence++)
      observer.event({
        schema_version: 1,
        run_id: run.run_id,
        sequence,
        type: "validation_stage",
        stage: { stage: "capture", outcome: "running" },
        diagnostic: {
          code: "capture_notice",
          message: `Notice ${sequence}`,
          phase: "capture",
        },
      })
    const current = observer.getSnapshot()
    assert.equal(current.diagnostics.length, 64)
    assert.equal(current.stages.length, 1)
    assert.equal(current.sequence, 99)
    observer.event({
      schema_version: 1,
      run_id: run.run_id,
      sequence: 1,
      type: "validation_stage",
      stage: { stage: "numerical", outcome: "passed" },
    })
    assert.equal(observer.getSnapshot(), current)
    assert.throws(
      () =>
        observer.event({
          schema_version: 1,
          run_id: "30000000-0000-4000-8000-000000000099",
          sequence: 101,
          type: "done",
          verdict: "valid",
        }),
      /another execution/
    )
    assert.equal(current.report, null)
  } finally {
    observer.dispose()
  }
})

test("cancel acknowledgement remains nonterminal and cancellation errors survive successful observation refresh", async () => {
  const { run, report } = execution()
  let calls = 0
  const observer = new ValidationObserver(
    run.run_id,
    transport(run, report, {
      cancel: async () => {
        calls++
        if (calls === 1) throw new ArchitectureApiError(409, null)
      },
    })
  )
  try {
    await observer.refresh()
    await observer.cancel()
    assert.match(observer.getSnapshot().actionError!, /409/)
    await observer.refresh()
    assert.match(observer.getSnapshot().actionError!, /409/)
    await observer.cancel()
    await observer.cancel()
    assert.equal(calls, 2)
    assert.equal(observer.getSnapshot().cancelRequested, true)
    assert.equal(observer.getSnapshot().run?.status, "running")
  } finally {
    observer.dispose()
  }
})

test("retry preserves the original snapshot and requires a fresh execution and request; ambiguous retries cannot duplicate", async () => {
  const { run, report } = execution("failed")
  let calls = 0
  const nextId = "30000000-0000-4000-8000-000000000099"
  const next = {
    ...run,
    run_id: nextId,
    ref_id: nextId,
    output: { ...run.output, id: nextId },
    status: "queued" as const,
    config: { ...run.config, request_id: nextId },
  }
  const observer = new ValidationObserver(
    run.run_id,
    transport(run, report, {
      retry: async () => {
        calls++
        return calls === 1
          ? next
          : {
              ...next,
              config: { ...next.config, parameters: { different: 3 } },
            }
      },
    })
  )
  try {
    await observer.refresh()
    assert.equal((await observer.retry())?.run_id, nextId)
    assert.equal(await observer.retry(), undefined)
    assert.equal(observer.getSnapshot().retryUncertain, true)
    await observer.retry()
    assert.equal(calls, 2)
    assert.equal(observer.getSnapshot().run?.run_id, run.run_id)
  } finally {
    observer.dispose()
  }
})

test("queue labels show actual invalid or inspection verdicts rather than treating done as numerical success", () => {
  const { run } = execution("done")
  run.result = fixture("ValidationReport", "invalid-completed-report")
  assert.equal(jobStatusLabel(jobFromValidation(run)), "Invalid")
  run.config = fixture("ValidationRequest", "inspection-no-constructor-profile")
  run.result = fixture("ValidationReport", "inspection-report")
  const job = jobFromValidation(run)
  assert.equal(jobStatusLabel(job), "Inspected")
  assert.equal(jobHref(job), `/arch/validations/${run.run_id}`)
  assert.equal(formatClock(null), "—")
})
