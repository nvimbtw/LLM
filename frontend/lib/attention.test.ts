import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import {
  catalogAfterMeasurement,
  decodeAttention,
  fetchAttention,
  fetchInspectionPoints,
} from "./attention"
import { decodeArchitecture } from "./architecture/decoder"
import { ArchitectureReadOwner } from "./architecture/read-owner"

const fixtures = JSON.parse(
  readFileSync(
    new URL(
      "../../contracts/modular-architecture/v1/api/cases.json",
      import.meta.url
    ),
    "utf8"
  )
)
/** Reads a private copy of shared observation data for controlled identity and numeric failures. */
function fixture(name: string): Record<string, unknown> {
  return structuredClone(
    fixtures.cases.find((item: { name: string }) => item.name === name).value
  )
}

test("bounded decoding retains little-endian probabilities and actual legacy axes", () => {
  const matrix = decodeAttention(fixture("actual legacy matrix"))
  assert.deepEqual([...matrix.weights], [1, 0, 0.25, 0.75])
  assert.equal(matrix.size, 2)
  assert.equal(matrix.layer, 1)
  assert.equal(matrix.pointId, null)
})

test("invalid probabilities and shape/byte disagreement cannot reach the figure", () => {
  for (const value of [NaN, Infinity, -1, 1.5]) {
    const raw = fixture("actual legacy matrix")
    const bytes = Buffer.alloc(16)
    bytes.writeFloatLE(value)
    raw.weights = bytes.toString("base64")
    assert.throws(() => decodeAttention(raw))
  }
  assert.throws(() =>
    decodeAttention({ ...fixture("actual legacy matrix"), shape: [2, 1] })
  )
})

test("worker replacement retires old proofs before exposing the new measurement", () => {
  const catalog = decodeArchitecture(
    "InspectionCatalog",
    fixture("discovered actual occurrences")
  )
  const point = catalog.points![0]
  catalog.points!.push({
    ...point,
    id: "e".repeat(64),
    validated_sequence_max: 2,
    evidence_hash: "f".repeat(64),
  })
  const matrix = decodeAttention({
    ...fixture("actual modular matrix"),
    runtime_id: "00000000-0000-4000-8000-000000000002",
  })
  const current = catalogAfterMeasurement(catalog, matrix)
  assert.equal(current.runtime_id, matrix.runtimeId)
  assert.equal(current.points![0].evidence_hash, matrix.evidenceHash)
  assert.equal(current.points![1].evidence_hash, null)
  assert.equal(catalog.points![1].evidence_hash, "f".repeat(64))
  assert.throws(() =>
    catalogAfterMeasurement(catalog, {
      ...matrix,
      definitionId: "0".repeat(64),
    })
  )
  assert.throws(() =>
    catalogAfterMeasurement(catalog, { ...matrix, pointId: "0".repeat(64) })
  )
})

test("transport captures session/selection identity and refuses unrelated results", async () => {
  const original = globalThis.fetch
  const requests: RequestInit[] = []
  let result: unknown = fixture("discovered actual occurrences")
  globalThis.fetch = (async (_url, init) => {
    requests.push(init!)
    return Response.json(result)
  }) as typeof fetch
  const controller = new AbortController()
  const request = { token: "actual-session", signal: controller.signal }
  try {
    assert.equal(
      (await fetchInspectionPoints("fixture", request)).model_name,
      "fixture"
    )
    assert.equal(
      new Headers(requests[0].headers).get("Authorization"),
      "Bearer actual-session"
    )
    assert.equal(requests[0].signal, request.signal)
    result = fixture("actual modular matrix")
    assert.equal(
      (
        await fetchAttention(
          "fixture",
          { text: "a b", point_id: "a".repeat(64), head: 0 },
          request
        )
      ).size,
      2
    )
    await assert.rejects(
      fetchAttention(
        "fixture",
        { text: "a b", point_id: "e".repeat(64) },
        request
      ),
      /selection/
    )
  } finally {
    globalThis.fetch = original
  }
})

test("selection changes cancel transport and suppress late noncooperative matrices", async () => {
  const owner = new ArchitectureReadOwner({
    token: "session",
    resource: "attention:model",
  })
  let complete!: (value: string) => void
  let signal: AbortSignal | undefined
  const pending = owner.read((requestSignal) => {
    signal = requestSignal
    return new Promise<string>((resolve) => {
      complete = resolve
    })
  })
  owner.cancel()
  complete("old point")
  assert.equal(signal?.aborted, true)
  assert.equal(await pending, undefined)
  assert.equal(await owner.read(async () => "new point"), "new point")
  owner.dispose()
})
