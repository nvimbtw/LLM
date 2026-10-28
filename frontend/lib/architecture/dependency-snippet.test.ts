import { test } from "node:test"
import assert from "node:assert/strict"
import { dependencySnippet } from "./dependency-snippet"
import type { ResourceSummary, RevisionRef } from "./contracts.generated"

const reference: RevisionRef = {
  kind: "block",
  resource_id: "30000000-0000-4000-8000-000000000001",
  revision_id: "30000000-0000-4000-8000-000000000002",
  content_hash: "a".repeat(64),
}
const summary: ResourceSummary = {
  kind: "block",
  resource_id: reference.resource_id,
  name: "Display title",
  description: "",
  origin: "custom",
  archived: false,
  metadata_version: 1,
  draft_version: 1,
  accepted_revision: reference,
}

test("dependency snippets retain the explicitly selected revision when the resource publishes a newer head", () => {
  const next = {
    ...reference,
    revision_id: "30000000-0000-4000-8000-000000000099",
    content_hash: "b".repeat(64),
  }
  const snippet = dependencySnippet(
    "chosen_module",
    { ...summary, accepted_revision: next },
    reference
  )
  assert.match(snippet, /modules:\n  chosen_module:/)
  assert.ok(snippet.includes(reference.revision_id))
  assert.ok(snippet.includes(reference.content_hash))
  assert.ok(!snippet.includes(next.revision_id))
})

test("archived resources, foreign revisions and injected module names cannot produce new-selection snippets", () => {
  assert.throws(
    () =>
      dependencySnippet("chosen", { ...summary, archived: true }, reference),
    /unarchived/
  )
  assert.throws(
    () =>
      dependencySnippet("chosen", summary, {
        ...reference,
        resource_id: "30000000-0000-4000-8000-000000000099",
      }),
    /accepted revisions/
  )
  assert.throws(
    () => dependencySnippet("name\ninjected:", summary, reference),
    /module name/
  )
})
