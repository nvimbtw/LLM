import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { readCopiedDraft } from "./copy"
import { draftDigest } from "./draft"
import { decodeArchitecture, type ContractName } from "./decoder"

const cases = JSON.parse(
  readFileSync(
    new URL(
      "../../../contracts/modular-architecture/v1/api/cases.json",
      import.meta.url
    ),
    "utf8"
  )
).cases

/** Reads a private shared fixture through the same interpreted boundary used for real authoring responses. */
function fixture<K extends ContractName>(name: K) {
  return decodeArchitecture(
    name,
    structuredClone(
      cases.find((item: { name: string }) => item.name === `${name}-positive`)
        .value
    )
  )
}

test("custom block and recipe copies preserve exact immutable source and dependency references", async () => {
  const original = globalThis.fetch
  try {
    for (const kind of ["block", "architecture"] as const) {
      const view =
        kind === "block"
          ? fixture("BlockRevisionView")
          : fixture("ArchitectureRevisionView")
      const source = "source" in view ? view.source : view.yaml
      const reference = {
        ...view.accepted.reference,
        content_hash: draftDigest(source),
      }
      view.accepted = { ...view.accepted, reference }
      const detail = fixture("ResourceDetail")
      detail.summary = {
        ...detail.summary,
        resource_id: reference.resource_id,
        kind,
        origin: "custom",
      }
      globalThis.fetch = (async (url, init) => {
        assert.equal(
          new Headers(init?.headers).get("Authorization"),
          "Bearer author-session"
        )
        return Response.json(
          String(url).includes("/revisions/") ? view : detail
        )
      }) as typeof fetch
      const copied = await readCopiedDraft(
        kind,
        reference.resource_id,
        reference.revision_id,
        { token: "author-session" }
      )
      assert.equal(copied.source, source)
      assert.equal(copied.name, `${detail.summary.name} copy`)
    }
  } finally {
    globalThis.fetch = original
  }
})

test("builtin copies request the eligible wrapper of the exact accepted revision", async () => {
  const original = globalThis.fetch
  const detail = fixture("ResourceDetail")
  detail.summary = { ...detail.summary, kind: "block", origin: "builtin" }
  const reference = fixture("BlockRevisionView").accepted.reference
  const urls: string[] = []
  globalThis.fetch = (async (url) => {
    urls.push(String(url))
    return Response.json(
      String(url).endsWith("copy-source")
        ? { name: "Eligible copy", source: "eligible wrapper" }
        : detail
    )
  }) as typeof fetch
  try {
    const copied = await readCopiedDraft(
      "block",
      detail.summary.resource_id,
      reference.revision_id,
      { token: "session" }
    )
    assert.equal(copied.source, "eligible wrapper")
    assert.ok(
      urls[1].endsWith(`/revisions/${reference.revision_id}/copy-source`)
    )
  } finally {
    globalThis.fetch = original
  }
})

test("mismatched resource or source bytes cannot seed a new custom resource", async () => {
  const original = globalThis.fetch
  const detail = fixture("ResourceDetail")
  const view = fixture("BlockRevisionView")
  detail.summary = {
    ...detail.summary,
    kind: "block",
    origin: "custom",
    resource_id: view.accepted.reference.resource_id,
  }
  globalThis.fetch = (async (url) =>
    Response.json(
      String(url).includes("/revisions/")
        ? { ...view, source: "changed source" }
        : detail
    )) as typeof fetch
  try {
    await assert.rejects(
      readCopiedDraft(
        "block",
        detail.summary.resource_id,
        view.accepted.reference.revision_id,
        { token: "session" }
      ),
      /differs/
    )
    await assert.rejects(
      readCopiedDraft(
        "block",
        "00000000-0000-4000-8000-000000000002",
        view.accepted.reference.revision_id,
        { token: "session" }
      ),
      /another resource/
    )
  } finally {
    globalThis.fetch = original
  }
})
