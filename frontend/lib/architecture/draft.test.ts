import { test } from "node:test"
import assert from "node:assert/strict"
import { ArchitectureDraft, draftDigest } from "./draft"
import { ArchitectureApiError } from "./http"
import type { DraftSaved } from "./contracts.generated"

const resource = "00000000-0000-4000-8000-000000000001"
/** Supplies an immutable acknowledgement for the exact saved bytes. */
function saved(text: string, version = 1): DraftSaved {
  return {
    resource_id: resource,
    draft_version: version,
    content_hash: draftDigest(text),
  }
}

test("save acknowledges submitted bytes while newer typing stays dirty", async () => {
  let complete!: (value: DraftSaved) => void
  const calls: Array<[number, string]> = []
  const draft = new ArchitectureDraft(
    saved("initial"),
    "initial",
    (version, text) => {
      calls.push([version, text])
      return new Promise((resolve) => {
        complete = resolve
      })
    }
  )
  draft.edit("invalid python(")
  const saving = draft.save()
  draft.edit("newer still invalid(")
  await draft.save()
  complete(saved("invalid python(", 2))
  await saving
  const current = draft.getSnapshot()
  assert.deepEqual(calls, [[1, "invalid python("]])
  assert.equal(current.text, "newer still invalid(")
  assert.equal(current.savedText, "invalid python(")
  assert.equal(current.saved.draft_version, 2)
  assert.equal(current.editRevision, 2)
  assert.equal(current.saving, false)
  draft.dispose()
})

test("concurrent server edits refuse overwrite and preserve invalid source", async () => {
  let calls = 0
  const draft = new ArchitectureDraft(saved("initial"), "initial", async () => {
    calls++
    throw new ArchitectureApiError(409, null)
  })
  draft.edit("local invalid(")
  await draft.save()
  assert.equal(draft.getSnapshot().conflict, true)
  assert.equal(draft.getSnapshot().text, "local invalid(")
  draft.edit("newer local(")
  assert.match(draft.getSnapshot().error!, /changed on the server/)
  await draft.save()
  assert.equal(calls, 1)
  assert.equal(draft.getSnapshot().conflict, true)
  draft.dispose()
})

test("replacement aborts writes and ignores late acknowledgements", async () => {
  let complete!: (value: DraftSaved) => void
  let signal: AbortSignal | undefined
  const draft = new ArchitectureDraft(
    saved("initial"),
    "initial",
    (_version, _text, requestSignal) => {
      signal = requestSignal
      return new Promise((resolve) => {
        complete = resolve
      })
    }
  )
  draft.edit("old session")
  const saving = draft.save()
  const before = draft.getSnapshot()
  draft.dispose()
  complete(saved("old session", 2))
  await saving
  assert.equal(signal?.aborted, true)
  assert.equal(draft.getSnapshot(), before)
})

test("wrong acknowledgements and multibyte overruns cannot become saved drafts", async () => {
  let calls = 0
  const draft = new ArchitectureDraft(saved("initial"), "initial", async () => {
    calls++
    return saved("wrong text", 2)
  })
  draft.edit("local")
  await draft.save()
  assert.equal(draft.getSnapshot().saved.draft_version, 1)
  assert.match(draft.getSnapshot().error!, /differs/)
  draft.edit("é".repeat(131073))
  await draft.save()
  assert.equal(calls, 1)
  assert.match(draft.getSnapshot().error!, /UTF-8/)
  draft.dispose()
})
