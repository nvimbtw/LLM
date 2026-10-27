import { test } from "node:test"
import assert from "node:assert/strict"
import { getRetainedSource } from "./model-graph"
import { draftDigest } from "./architecture/draft"
import type { RetainedSource } from "./architecture/contracts.generated"

const revision = "00000000-0000-4000-8000-000000000001"
const model = "00000000-0000-4000-8000-000000000002"
const definitionId = "a".repeat(64)
const source = "# Frozen source: ž 🎯\nprint('retained')\n"
const retained: RetainedSource = {
  revision_id: revision,
  definition_id: definitionId,
  content_hash: draftDigest(source),
  file: "sources/frozen.py",
  language: "python",
  source,
}

test("retained source transport preserves exact UTF-8 bytes, captured session and abort ownership", async () => {
  const previousFetch = globalThis.fetch
  const controller = new AbortController()
  globalThis.fetch = (async (url, init) => {
    assert.equal(
      url,
      `http://localhost:8080/models/${model}/sources/${revision}`
    )
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer captured"
    )
    assert.equal(init?.signal, controller.signal)
    return Response.json(retained)
  }) as typeof fetch
  try {
    const value = await getRetainedSource(
      "models",
      model,
      revision,
      { token: "captured", signal: controller.signal },
      definitionId,
      retained.content_hash
    )
    assert.equal(value.source, source)
    assert.equal(value.content_hash, draftDigest(value.source))
  } finally {
    globalThis.fetch = previousFetch
  }
})

test("source responses from a different immutable revision or definition cannot be displayed", async () => {
  const previousFetch = globalThis.fetch
  try {
    for (const replacement of [
      { revision_id: model },
      { definition_id: "b".repeat(64) },
      { content_hash: draftDigest("different"), source: "different" },
    ]) {
      globalThis.fetch = (async () =>
        Response.json({ ...retained, ...replacement })) as typeof fetch
      await assert.rejects(
        getRetainedSource(
          "models",
          model,
          revision,
          { token: "captured" },
          definitionId,
          retained.content_hash
        ),
        /another frozen revision or definition/
      )
    }
    globalThis.fetch = (async () =>
      Response.json({ ...retained, source: "corrupt" })) as typeof fetch
    await assert.rejects(
      getRetainedSource(
        "models",
        model,
        revision,
        { token: "captured" },
        definitionId
      ),
      /hash mismatch/
    )
  } finally {
    globalThis.fetch = previousFetch
  }
})

test("validation source has no saved model identity and frozen builtin descriptors remain JSON", async () => {
  const previousFetch = globalThis.fetch
  const descriptor = '{"frozen_builtin":true}'
  globalThis.fetch = (async () =>
    Response.json({
      ...retained,
      definition_id: null,
      source: descriptor,
      content_hash: draftDigest(descriptor),
      file: "sources/builtin.builtin.json",
      language: "json",
    })) as typeof fetch
  try {
    const value = await getRetainedSource("validations", model, revision, {
      token: "captured",
    })
    assert.equal(value.definition_id, null)
    assert.equal(value.language, "json")
    assert.equal(value.source, descriptor)
    await assert.rejects(
      getRetainedSource(
        "models",
        model,
        revision,
        { token: "captured" },
        definitionId
      ),
      /another frozen revision or definition/
    )
  } finally {
    globalThis.fetch = previousFetch
  }
})
