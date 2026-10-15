import assert from "node:assert/strict"
import { test } from "node:test"
import { jobHref } from "./jobs"
import { resolveExecutionTarget, withRouteQuery } from "./routes"

const id = "11111111-1111-4111-8111-111111111111"

test("durable jobs use domain URLs while tokenizer artifact IDs remain separate from build IDs", () => {
  assert.equal(jobHref({ id_job: id, kind: "train" }), `/train/${id}`)
  assert.equal(jobHref({ id_job: id, kind: "benchmark" }), `/bench/${id}`)
  assert.equal(jobHref({ id_job: id, kind: "rl" }), `/gyms/${id}`)
  assert.equal(
    jobHref({ id_job: id, kind: "tokenize" }),
    `/tokenizers/builds/${id}`
  )
})

test("redirects preserve repeated query values and encoded parent context", () => {
  const result = withRouteQuery(`/train/${id}`, {
    from: id,
    seed: ["4", "5"],
    parent: "/bench?mode=all",
    missing: undefined,
  })
  const query = new URL(result, "http://localhost").searchParams
  assert.deepEqual(query.getAll("seed"), ["4", "5"])
  assert.equal(query.get("parent"), "/bench?mode=all")
  assert.equal(query.get("from"), id)
})

test("a shared gym URL falls back to evaluation only on an execution 404", async () => {
  const previous = globalThis.fetch
  const requested: string[] = []
  globalThis.fetch = async (input) => {
    const path = new URL(String(input)).pathname
    requested.push(path)
    return new Response("{}", { status: path.startsWith("/runs/") ? 404 : 200 })
  }
  try {
    assert.deepEqual(await resolveExecutionTarget(id, "rl"), {
      type: "gym",
      id,
    })
    assert.deepEqual(requested, [`/runs/${id}`, `/gym/runs/${id}`])
    requested.length = 0
    globalThis.fetch = async (input) => {
      requested.push(String(input))
      return new Response("{}", { status: 503 })
    }
    await assert.rejects(resolveExecutionTarget(id, "rl"), /Failed to load/)
    assert.equal(requested.length, 1)
  } finally {
    globalThis.fetch = previous
  }
})

test("existing executions determine the canonical domain instead of being mistaken for an evaluation", async () => {
  const previous = globalThis.fetch
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ run_id: id, kind: "rl", status: "done" }))
  try {
    const result = await resolveExecutionTarget(id, "rl")
    assert.equal(result.type, "job")
    if (result.type === "job") assert.equal(jobHref(result.job), `/gyms/${id}`)
  } finally {
    globalThis.fetch = previous
  }
})

test("historical benchmark report IDs remain readable without a matching execution and retain cancellation", async () => {
  const previous = globalThis.fetch
  const controller = new AbortController()
  const requested: string[] = []
  globalThis.fetch = async (input, init) => {
    const path = new URL(String(input)).pathname
    requested.push(path)
    assert.equal(init?.signal, controller.signal)
    return path.startsWith("/runs/")
      ? new Response("{}", { status: 404 })
      : new Response(
          JSON.stringify({
            benchmark_id: id,
            suite: "language",
            status: "done",
            report: null,
          })
        )
  }
  try {
    const target = await resolveExecutionTarget(
      id,
      "benchmark",
      controller.signal
    )
    assert.equal(target.type, "report")
    if (target.type === "report") assert.equal(target.run.id_run, id)
    assert.deepEqual(requested, [`/runs/${id}`, `/benchmarks/${id}`])
    requested.length = 0
    globalThis.fetch = async () => {
      requested.push("invalid execution")
      return new Response(
        JSON.stringify({ run_id: "wrong", kind: "benchmark" })
      )
    }
    await assert.rejects(
      resolveExecutionTarget(id, "benchmark"),
      /invalid execution/
    )
    assert.equal(requested.length, 1)
  } finally {
    globalThis.fetch = previous
  }
})
