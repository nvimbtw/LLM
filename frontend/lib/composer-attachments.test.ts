import { test } from "node:test"
import assert from "node:assert/strict"
import { ComposerAttachments, readyAttachments } from "./composer-attachments"
import type { Attachment } from "./attachments"

/** Creates a draft with deterministic upload completion and observable preview/discard effects. */
function harness() {
  const pending: {
    file: File
    resolve: (stored: Attachment) => void
    reject: (error: Error) => void
  }[] = []
  const discarded: string[] = []
  const revoked: string[] = []
  const draft = new ComposerAttachments({
    upload: (file) =>
      new Promise((resolve, reject) => {
        pending.push({ file, resolve, reject })
      }),
    discard: async (id) => {
      discarded.push(id)
    },
    preview: (file) => `blob:${file.name}`,
    revoke: (url) => {
      revoked.push(url)
    },
  })
  return { draft, pending, discarded, revoked }
}

/** Completes a fake upload using its picked file metadata, then lets the draft continuation run. */
async function finish(
  upload: { file: File; resolve: (stored: Attachment) => void },
  id: string
): Promise<void> {
  upload.resolve({
    id,
    filename: upload.file.name,
    mediaType: upload.file.type,
    size: upload.file.size,
  })
  await Promise.resolve()
}

test("rapid picks share the cap; failed excess files stay visible and ready metadata retains pick order", async () => {
  const view = harness()
  view.draft.addFiles([new File(["a"], "a.png", { type: "image/png" })])
  view.draft.addFiles(
    Array.from(
      { length: 5 },
      (_, index) => new File(["text"], `${index}.txt`, { type: "text/plain" })
    )
  )
  assert.equal(view.pending.length, 5)
  assert.equal(view.draft.getSnapshot().length, 6)
  assert.equal(view.draft.getSnapshot()[5].status, "failed")
  await finish(view.pending[1], "second")
  await finish(view.pending[0], "first")
  assert.deepEqual(
    readyAttachments(view.draft.getSnapshot()).map((item) => item.id),
    ["first", "second"]
  )
  view.draft.release()
  assert.deepEqual(view.revoked, ["blob:a.png"])
  assert.deepEqual(view.discarded, [])
  assert.deepEqual(view.draft.getSnapshot(), [])
})

test("removing a pending upload revokes its preview once and deletes its row once when the upload arrives", async () => {
  const view = harness()
  view.draft.addFiles([
    new File(["image"], "pending.png", { type: "image/png" }),
  ])
  const item = view.draft.getSnapshot()[0]
  view.draft.remove(item.key)
  view.draft.remove(item.key)
  assert.deepEqual(view.revoked, ["blob:pending.png"])
  await finish(view.pending[0], "late-row")
  assert.deepEqual(view.discarded, ["late-row"])
  assert.deepEqual(view.draft.getSnapshot(), [])
})

test("abandon deletes ready and future uploads while sending retains ready rows", async () => {
  const view = harness()
  view.draft.addFiles([
    new File(["a"], "ready.txt"),
    new File(["b"], "pending.txt"),
  ])
  await finish(view.pending[0], "ready-row")
  view.draft.discard()
  await finish(view.pending[1], "pending-row")
  assert.deepEqual(view.discarded, ["ready-row", "pending-row"])
  view.draft.addFiles([new File(["sent"], "sent.png", { type: "image/png" })])
  await finish(view.pending[2], "claimed-row")
  view.draft.release()
  view.draft.dispose()
  assert.deepEqual(view.discarded, ["ready-row", "pending-row"])
  assert.deepEqual(view.revoked, ["blob:sent.png"])
})

test("failed uploads retain their error and repeated snapshot reads have no effects", async () => {
  const view = harness()
  view.draft.addFiles([new File(["a"], "failed.txt")])
  view.pending[0].reject(new Error("Too large"))
  await Promise.resolve()
  const snapshot = view.draft.getSnapshot()
  assert.equal(snapshot[0].status, "failed")
  assert.equal(snapshot[0].error, "Too large")
  assert.equal(view.draft.getSnapshot(), snapshot)
  assert.equal(view.draft.getSnapshot(), snapshot)
  assert.deepEqual(view.discarded, [])
})
