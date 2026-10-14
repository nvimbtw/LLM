import { test } from "node:test"
import assert from "node:assert/strict"
import { ProjectCatalog } from "./project-catalog"
import type { ProjectPage, Project } from "./projects"

/** Creates an explicitly settled async result for account, refresh and cancellation race tests. */
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (failure: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}

/** Returns realistic private metadata whose name identifies its owning session in assertions. */
function project(name: string): Project {
  return {
    project_id: "00000000-0000-4000-8000-000000000001",
    name,
    notes: "private",
    description: "",
    thread_count: 0,
    created_at: "2026-10-14T12:00:00Z",
    updated_at: "2026-10-14T12:00:00Z",
  }
}

/** Wraps one account's row in its initial catalogue page. */
function page(name: string): ProjectPage {
  return { projects: [project(name)], total: 1, limit: 50, offset: 0 }
}

test("replacement sessions immediately hide private rows and ignore old success and failure completions", async () => {
  const requests: ReturnType<typeof deferred<ProjectPage>>[] = []
  const signals: (AbortSignal | undefined)[] = []
  const owner = new ProjectCatalog((_token, _offset, signal) => {
    const request = deferred<ProjectPage>()
    requests.push(request)
    signals.push(signal)
    return request.promise
  })
  owner.setSession("account-a")
  requests[0].resolve(page("A"))
  await requests[0].promise
  assert.equal(owner.getSnapshot().projects[0].name, "A")
  const oldRefresh = owner.refresh()
  owner.setSession("account-b")
  assert.deepEqual(owner.getSnapshot().projects, [])
  assert.equal(signals[1]?.aborted, true)
  requests[2].resolve(page("B"))
  await requests[2].promise
  requests[1].reject(new Error("A failed late"))
  await oldRefresh
  assert.equal(owner.getSnapshot().projects[0].name, "B")
  assert.equal(owner.getSnapshot().error, null)
  owner.setSession(null)
  assert.deepEqual(owner.getSnapshot().projects, [])
  assert.equal(owner.getSnapshot().loading, false)
})

test("a newer refresh wins even when an aborted loader ignores cancellation", async () => {
  const requests: ReturnType<typeof deferred<ProjectPage>>[] = []
  const owner = new ProjectCatalog(() => {
    const request = deferred<ProjectPage>()
    requests.push(request)
    return request.promise
  })
  owner.setSession("a")
  const refresh = owner.refresh()
  requests[1].resolve(page("Fresh"))
  await refresh
  requests[0].resolve(page("Old"))
  await requests[0].promise
  assert.equal(owner.getSnapshot().projects[0].name, "Fresh")
})

test("catalogue failures preserve loaded rows, allow retry and never publish after disposal", async () => {
  let fail = false
  const owner = new ProjectCatalog(async () => {
    if (fail) throw new Error("offline")
    return page("Available")
  })
  owner.setSession("a")
  await owner.refresh()
  fail = true
  await owner.refresh()
  assert.equal(owner.getSnapshot().projects[0].name, "Available")
  assert.equal(owner.getSnapshot().error, "offline")
  fail = false
  await owner.refresh()
  assert.equal(owner.getSnapshot().error, null)
  const request = deferred<ProjectPage>()
  const disposable = new ProjectCatalog(() => request.promise)
  disposable.setSession("a")
  disposable.dispose()
  request.resolve(page("Too late"))
  await request.promise
  assert.deepEqual(disposable.getSnapshot().projects, [])
})

test("project pagination appends unique rows and has one active read", async () => {
  const offsets: number[] = []
  const owner = new ProjectCatalog(async (_token, offset) => {
    offsets.push(offset)
    const row = project(offset === 0 ? "First" : "Second")
    if (offset !== 0) row.project_id = "00000000-0000-4000-8000-000000000002"
    return { projects: [row], total: 2, limit: 50, offset }
  })
  owner.setSession("a")
  await owner.refresh()
  await Promise.all([owner.loadMore(), owner.loadMore()])
  assert.deepEqual(
    owner.getSnapshot().projects.map((row) => row.name),
    ["First", "Second"]
  )
  assert.equal(offsets.filter((offset) => offset === 1).length, 1)
  await owner.loadMore()
  assert.equal(offsets.filter((offset) => offset === 1).length, 1)
})

test("effect cleanup and reactivation restart an aborted load for the same session", async () => {
  const requests: ReturnType<typeof deferred<ProjectPage>>[] = []
  const owner = new ProjectCatalog(() => {
    const request = deferred<ProjectPage>()
    requests.push(request)
    return request.promise
  })
  owner.setSession("a")
  owner.dispose()
  owner.setSession("a")
  assert.equal(requests.length, 2)
  requests[1].resolve(page("Remounted"))
  await requests[1].promise
  assert.equal(owner.getSnapshot().projects[0].name, "Remounted")
  requests[0].resolve(page("Disposed"))
  await requests[0].promise
  assert.equal(owner.getSnapshot().projects[0].name, "Remounted")
})

test("deduplicating a shifted catalogue page retains its server cursor instead of repeating earlier offsets", async () => {
  const offsets: number[] = []
  const owner = new ProjectCatalog(async (_token, offset) => {
    offsets.push(offset)
    return { projects: [project("Repeated")], total: 3, limit: 50, offset }
  })
  owner.setSession("a")
  await owner.refresh()
  await owner.loadMore()
  await owner.loadMore()
  assert.equal(owner.getSnapshot().projects.length, 1)
  assert.equal(owner.getSnapshot().nextOffset, 3)
  assert.deepEqual(offsets.slice(-2), [1, 2])
  await owner.loadMore()
  assert.equal(offsets.length, 4)
})
