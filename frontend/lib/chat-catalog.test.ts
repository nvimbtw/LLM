import assert from "node:assert/strict"
import { test } from "node:test"
import { ChatCatalog, RECENT_CHAT_LIMIT } from "./chat-catalog"
import type { ChatSummary } from "./chats"

/** Builds an owned chat summary for controlled metadata and request-race tests. */
function chat(id: string, title = id, tokens = 0): ChatSummary {
  return {
    id,
    title,
    tokens,
    projectId: null,
    createdAt: "2026-10-17T00:00:00Z",
  }
}

/** Creates a catalog with independently controllable HTTP reads, including adapters that ignore abort. */
function harness() {
  const reads: {
    token: string
    signal?: AbortSignal
    limit?: number
    resolve: (rows: ChatSummary[]) => void
    reject: (error: Error) => void
  }[] = []
  const catalog = new ChatCatalog(
    (token, signal, limit) =>
      new Promise((resolve, reject) => {
        reads.push({ token, signal, limit, resolve, reject })
      })
  )
  return { catalog, reads }
}

/** Flushes the catalog's promise notifications without relying on elapsed-time assertions. */
async function settle(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve))
}

test("startup requests five recent chats and a closed archive never refreshes in the background", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  assert.equal(reads.length, 1)
  assert.equal(reads[0].limit, RECENT_CHAT_LIMIT)
  reads[0].resolve([chat("recent")])
  await settle()
  const closeFirst = catalog.retainArchive()
  const closeSecond = catalog.retainArchive()
  assert.equal(reads.length, 2)
  assert.equal(reads[1].limit, undefined)
  reads[1].resolve([chat("recent"), chat("older")])
  await settle()
  closeFirst()
  assert.equal(catalog.getSnapshot().archive?.length, 2)
  closeSecond()
  assert.equal(catalog.getSnapshot().archive, null)
  const refresh = catalog.refresh()
  assert.equal(reads.length, 3)
  assert.equal(reads[2].limit, RECENT_CHAT_LIMIT)
  reads[2].resolve([chat("recent")])
  await refresh
  catalog.dispose()
})

test("a mutation supersedes an initial archive read and refresh waits for the mounted archive", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  reads[0].resolve([chat("current")])
  await settle()
  catalog.rememberSummary(chat("older", "Before", 8), catalog.getVersion())
  const release = catalog.retainArchive()
  catalog.updateSummary(chat("older", "Renamed", 8))
  let settled = false
  const refresh = catalog.refresh().then(() => {
    settled = true
  })
  assert.equal(reads[1].signal?.aborted, true)
  reads[1].resolve([chat("older", "Before")])
  reads[2].resolve([chat("current")])
  await settle()
  assert.equal(settled, false)
  assert.equal(catalog.getSummary("older")?.title, "Renamed")
  reads[3].resolve([chat("current"), chat("older", "Renamed", 8)])
  await refresh
  assert.equal(catalog.getSnapshot().archive?.[1].title, "Renamed")
  release()
  catalog.dispose()
})

test("a write during a read forces a new snapshot even before its caller requests refresh", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  const release = catalog.retainArchive()
  catalog.updateSummary(chat("older", "Committed", 12))
  reads[1].resolve([chat("older", "Obsolete")])
  await settle()
  assert.equal(reads.length, 3)
  reads[2].resolve([chat("older", "Committed", 12)])
  await settle()
  assert.equal(catalog.getSnapshot().archive?.[0].title, "Committed")
  release()
  catalog.dispose()
})

test("chat metadata has one owner across history, composer rename and sidebar rename", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  reads[0].resolve([chat("recent")])
  await settle()
  const beforeRename = catalog.getVersion()
  catalog.rememberSummary(chat("older", "Original", 20), beforeRename)
  catalog.updateSummary(chat("older", "Composer name", 20))
  catalog.updateSummary(chat("older", "Sidebar name", 20))
  catalog.rememberSummary(chat("older", "Stale", 1), beforeRename)
  assert.equal(catalog.getSummary("older")?.title, "Sidebar name")
  assert.equal(catalog.getSummary("older")?.tokens, 20)
  const beforeOtherRename = catalog.getVersion()
  catalog.updateSummary(chat("recent", "Other chat renamed"))
  catalog.rememberSummary(chat("older", "Sidebar name", 24), beforeOtherRename)
  assert.equal(catalog.getSummary("older")?.tokens, 24)
  assert.equal(catalog.getSummary("recent")?.title, "Other chat renamed")
  catalog.remove("older")
  catalog.rememberSummary(chat("older", "Deleted"), beforeOtherRename)
  assert.equal(catalog.getSummary("older"), undefined)
  catalog.dispose()
})

test("account replacement and cleanup ignore late success, errors and history metadata", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  const oldVersion = catalog.getVersion()
  const release = catalog.retainArchive()
  catalog.setSession("account-b")
  assert.equal(reads[0].signal?.aborted, true)
  assert.equal(reads[1].signal?.aborted, true)
  reads[0].resolve([chat("private-a")])
  reads[1].reject(new Error("old account failed"))
  catalog.rememberSummary(chat("private-a"), oldVersion)
  reads[2].resolve([chat("private-b")])
  reads[3].resolve([chat("private-b")])
  await settle()
  assert.equal(catalog.getSnapshot().token, "account-b")
  assert.deepEqual(
    catalog.getSnapshot().recent.map((row) => row.id),
    ["private-b"]
  )
  assert.equal(catalog.getSummary("private-a"), undefined)
  assert.equal(catalog.getSnapshot().archiveError, null)
  release()
  catalog.dispose()
})

test("archive errors settle loading, retry succeeds and closing aborts an unfinished read", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  const release = catalog.retainArchive()
  reads[1].reject(new Error("Archive offline"))
  await settle()
  assert.equal(catalog.getSnapshot().archiveLoading, false)
  assert.equal(catalog.getSnapshot().archiveError, "Archive offline")
  const retry = catalog.loadArchive()
  reads[2].resolve([chat("restored")])
  await retry
  assert.equal(catalog.getSnapshot().archiveError, null)
  release()
  const closeAgain = catalog.retainArchive()
  closeAgain()
  assert.equal(reads[3].signal?.aborted, true)
  reads[3].resolve([chat("closed")])
  await settle()
  assert.equal(catalog.getSnapshot().archive, null)
  catalog.dispose()
})

test("effect cleanup and reattachment restart an aborted recent read for the same account", async () => {
  const { catalog, reads } = harness()
  catalog.setSession("account-a")
  catalog.dispose()
  catalog.setSession("account-a")
  assert.equal(reads.length, 2)
  reads[0].resolve([chat("stale")])
  reads[1].resolve([chat("current")])
  await settle()
  assert.deepEqual(
    catalog.getSnapshot().recent.map((row) => row.id),
    ["current"]
  )
  catalog.dispose()
})
