import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { architectures } from "../architectures"
import { decodeArchitecture } from "./decoder"
import { blocks } from "../blocks"
import { getArchitectureTokenizerBinding } from "../model-validation"
import { draftDigest } from "./draft"

const root = new URL(
  "../../../contracts/modular-architecture/v1/",
  import.meta.url
)
const cases = JSON.parse(
  readFileSync(new URL("api/cases.json", root), "utf8")
).cases

/** Reads an isolated shared prospective metadata fixture without treating it as numerical validation evidence. */
function fixture(name: string): unknown {
  return structuredClone(
    cases.find((item: { name: string }) => item.name === name).value
  )
}

test("prospective metadata decodes the actual complete generated recipe schema", () => {
  const value = decodeArchitecture(
    "ArchitectureSchema",
    fixture("prospective installed runtime")
  )
  const recipe = JSON.parse(
    readFileSync(new URL("schemas/Recipe.schema.json", root), "utf8")
  )
  const current = decodeArchitecture("ArchitectureSchema", { ...value, recipe })
  assert.equal(current.recipe.title, "Recipe")
  assert.equal(current.default_profile.device, "cpu")
  assert.equal("receipt" in current, false)
})

test("selected revision lookup verifies exact immutable bytes and never falls back to latest", async () => {
  const original = globalThis.fetch
  const view = decodeArchitecture(
    "ArchitectureRevisionView",
    fixture("ArchitectureRevisionView-positive")
  )
  view.accepted.reference.content_hash = draftDigest(view.yaml)
  globalThis.fetch = (async (url) => {
    assert.equal(
      String(url),
      `http://localhost:8080/architectures/revisions/${view.accepted.reference.revision_id}`
    )
    return Response.json(view)
  }) as typeof fetch
  try {
    await architectures.selected(view.accepted.reference.revision_id, {
      token: "actual-session",
    })
    view.yaml += "\nchanged bytes"
    await assert.rejects(
      architectures.selected(view.accepted.reference.revision_id, {
        token: "actual-session",
      }),
      /immutable revision/
    )
  } finally {
    globalThis.fetch = original
  }
})

test("prospective profile replies remain bound to the selected batch, seed and actual tokenizer", async () => {
  const original = globalThis.fetch
  const response = decodeArchitecture(
    "TrainingArchitecturePreview",
    fixture("TrainingArchitecturePreview-positive")
  )
  const request = decodeArchitecture("TrainingArchitectureRequest", {
    selection: {
      revision_id: response.architecture.revision_id,
      parameters: {},
    },
    batch_size: response.profile.domains[0].batch_max,
    seed: response.profile.numerical_seed,
    tokenizer_id: response.tokenizer!.tokenizer_id,
  })
  globalThis.fetch = (async () => Response.json(response)) as typeof fetch
  try {
    await architectures.trainingPreview(request, { token: "actual-session" })
    response.profile.numerical_seed = (response.profile.numerical_seed ?? 0) + 1
    await assert.rejects(
      architectures.trainingPreview(request, { token: "actual-session" }),
      /numerical inputs/
    )
  } finally {
    globalThis.fetch = original
  }
})

test("immutable history forwards bounded opaque cursors without altering their bytes", async () => {
  const original = globalThis.fetch
  const cursor = "opaque+/=history cursor"
  globalThis.fetch = (async (url) => {
    const query = new URL(String(url)).searchParams
    assert.equal(query.get("limit"), "50")
    assert.equal(query.get("cursor"), cursor)
    return Response.json({ items: [], next_cursor: null })
  }) as typeof fetch
  try {
    await blocks.revisions(
      "30000000-0000-4000-8000-000000000001",
      { token: "actual-session" },
      cursor
    )
  } finally {
    globalThis.fetch = original
  }
})

test("final binding transport rejects another tokenizer and missing requested reserved names", async () => {
  const original = globalThis.fetch
  const binding = decodeArchitecture(
    "FinalTokenizer",
    fixture("FinalTokenizer-positive")
  )
  globalThis.fetch = (async () => Response.json(binding)) as typeof fetch
  try {
    await assert.rejects(
      getArchitectureTokenizerBinding("30000000-0000-4000-8000-000000000099", {
        token: "actual-session",
      }),
      /selected vocabulary/
    )
    await assert.rejects(
      getArchitectureTokenizerBinding(
        binding.tokenizer_id,
        { token: "actual-session" },
        ["missing_id"]
      ),
      /reserved names/
    )
  } finally {
    globalThis.fetch = original
  }
})

test("saved declaration transport retains exact subject identity and rejects a stale response", async () => {
  const original = globalThis.fetch
  const expected = decodeArchitecture(
    "RecipeDraftInspection",
    fixture("saved recipe declarations only")
  )
  let response = expected
  let body: unknown
  globalThis.fetch = (async (_url, init) => {
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer actual-session"
    )
    body = JSON.parse(String(init?.body))
    return Response.json(response)
  }) as typeof fetch
  try {
    const current = await architectures.declaration(
      expected.subject.resource_id,
      expected.subject,
      { token: "actual-session" }
    )
    assert.deepEqual(body, expected.subject)
    assert.equal(current.verdict, "inspection_only")
    response = {
      ...expected,
      subject: {
        ...expected.subject,
        draft_version: expected.subject.draft_version + 1,
      },
    }
    await assert.rejects(
      architectures.declaration(
        expected.subject.resource_id,
        expected.subject,
        { token: "actual-session" }
      ),
      /different saved draft/
    )
  } finally {
    globalThis.fetch = original
  }
})
