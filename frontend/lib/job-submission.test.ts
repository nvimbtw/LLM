import { test } from "node:test"
import assert from "node:assert/strict"
import { JobSubmission } from "./job-submission"
import {
  JobSubmissionError,
  queueTraining,
  queueReinforcement,
  retryJob,
  recoverJob,
  type Job,
  type JobSubmissionRequest,
} from "./jobs"
import { DEFAULT_CONFIG } from "@/hooks/use-training"

const job: Job = {
  id_job: "30000000-0000-4000-8000-000000000003",
  kind: "train",
  status: "queued",
  label: "Frozen selection",
  config: {},
  progress: 0,
  stage: null,
  ref_id: null,
  output: { type: "model", id: null },
  result: null,
  depends_on: null,
  error: null,
  created_at: null,
  started_at: null,
  finished_at: null,
}

test("unknown outcomes reuse the same request UUID only for unchanged inputs, including 5xx and decode failures", async () => {
  const owner = new JobSubmission("session-a")
  const requests: JobSubmissionRequest[] = []
  for (const error of [
    new TypeError("network lost"),
    new JobSubmissionError(503, "unavailable"),
    new SyntaxError("invalid response"),
  ]) {
    assert.equal(
      await owner.submit("same", async (request) => {
        requests.push(request)
        throw error
      }),
      undefined
    )
    assert.equal(owner.snapshot().uncertain, true)
  }
  const next = await owner.submit("same", async (request) => {
    requests.push(request)
    return job
  })
  assert.equal(next, job)
  assert.equal(new Set(requests.map((request) => request.requestId)).size, 1)
  assert.ok(requests.every((request) => request.token === "session-a"))
  assert.equal(
    await owner.submit("same", async () => {
      throw new Error("duplicate")
    }),
    job
  )
  owner.acknowledge()
  await owner.submit("same", async (request) => {
    requests.push(request)
    return job
  })
  assert.notEqual(requests.at(-1)?.requestId, requests[0].requestId)
})

test("changed inputs and definite rejections get fresh identities; disposal suppresses late success and never cancels the server job", async () => {
  const owner = new JobSubmission("session-a")
  const requests: JobSubmissionRequest[] = []
  await owner.submit("first", async (request) => {
    requests.push(request)
    throw new Error("unknown")
  })
  await owner.submit("changed", async (request) => {
    requests.push(request)
    throw new JobSubmissionError(422, "Fix the field")
  })
  assert.equal(owner.snapshot().uncertain, false)
  assert.equal(owner.snapshot().error, "Fix the field")
  let resolve!: (job: Job) => void
  const pending = owner.submit("changed", (request) => {
    requests.push(request)
    return new Promise<Job>((complete) => {
      resolve = complete
    })
  })
  assert.equal(await owner.submit("changed", async () => job), undefined)
  assert.equal(new Set(requests.map((request) => request.requestId)).size, 3)
  owner.dispose()
  assert.equal(requests.at(-1)?.signal?.aborted, true)
  resolve(job)
  assert.equal(await pending, undefined)
  assert.equal(owner.snapshot().job, null)
})

test("malformed successful queue acknowledgements remain uncertain and cannot authorize navigation", async () => {
  const owner = new JobSubmission("session-a")
  assert.equal(
    await owner.submit("same", async () => ({
      ...job,
      id_job: "invalid-execution",
    })),
    undefined
  )
  assert.equal(owner.snapshot().uncertain, true)
  const requestId = owner.snapshot().requestId
  const result = await owner.submit("same", async (request) => {
    assert.equal(request.requestId, requestId)
    return job
  })
  assert.equal(result, job)
})

test("training, SFT, RL, retry and recovery preserve request identity/session through Rust query routes", async () => {
  const original = globalThis.fetch
  const request = {
    token: "captured-session",
    requestId: "30000000-0000-4000-8000-000000000001",
    signal: new AbortController().signal,
  }
  const paths: string[] = []
  globalThis.fetch = (async (url, init) => {
    const target = new URL(String(url))
    paths.push(target.pathname)
    assert.equal(target.searchParams.get("request_id"), request.requestId)
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer captured-session"
    )
    assert.equal(init?.signal, request.signal)
    return Response.json({ ...job, run_id: job.id_job })
  }) as typeof fetch
  try {
    await queueTraining(DEFAULT_CONFIG, [], request)
    await queueTraining(
      { ...DEFAULT_CONFIG, mode: "sft", base_model_id: "saved" },
      [],
      request
    )
    await queueReinforcement({ base_model: "saved" }, [], request)
    await retryJob(job, request)
    await recoverJob(job.id_job, request)
    assert.deepEqual(paths, [
      "/runs/train",
      "/runs/sft",
      "/runs/rl",
      `/runs/${job.id_job}/retry`,
      `/runs/${job.id_job}/recover`,
    ])
  } finally {
    globalThis.fetch = original
  }
})
