import { after, beforeEach, test } from "node:test"
import assert from "node:assert/strict"
import { apiFetch } from "./api"
import { loadProjects, createProject, deleteProject } from "./projects"
import { assignChatProject } from "./chats"
import {
  loadComposerChoice,
  parseComposerChoice,
  persistComposerChoice,
} from "./composer-choice"

const originalFetch = globalThis.fetch
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window")
const originalStorage = Object.getOwnPropertyDescriptor(
  globalThis,
  "localStorage"
)
let token: string | null = "account-a"
const redirects: string[] = []
const projectId = "00000000-0000-4000-8000-000000000001"
const chatId = "00000000-0000-4000-8000-000000000002"
const project = {
  project_id: projectId,
  name: "Research",
  description: "",
  notes: "Instructions",
  thread_count: 0,
  created_at: "2026-10-14T12:00:00Z",
  updated_at: "2026-10-14T12:00:00Z",
}

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: {
    dispatchEvent: () => true,
    location: {
      pathname: "/projects",
      assign: (path: string) => redirects.push(path),
    },
  },
})
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: () => token,
    removeItem: (key: string) => {
      if (key === "auth_token") token = null
    },
  },
})

beforeEach(() => {
  token = "account-a"
  redirects.length = 0
  globalThis.fetch = originalFetch
})
after(() => {
  globalThis.fetch = originalFetch
  if (originalWindow)
    Object.defineProperty(globalThis, "window", originalWindow)
  else Reflect.deleteProperty(globalThis, "window")
  if (originalStorage)
    Object.defineProperty(globalThis, "localStorage", originalStorage)
  else Reflect.deleteProperty(globalThis, "localStorage")
})

test("project reads and writes carry their account token and exact bounded/private API shape", async () => {
  const calls: { url: string; init?: RequestInit }[] = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init })
    return init?.method === "POST"
      ? Response.json(project, { status: 201 })
      : init?.method === "DELETE"
        ? new Response(null, { status: 204 })
        : Response.json({ projects: [project], total: 1, limit: 50, offset: 0 })
  }
  await loadProjects("account-a")
  await createProject("account-a", {
    name: " Research ",
    description: "",
    notes: "Instructions",
  })
  await deleteProject("account-a", projectId)
  assert.equal(calls[0].url, "http://localhost:8080/projects?limit=50&offset=0")
  assert.deepEqual(JSON.parse(String(calls[1].init?.body)), {
    name: "Research",
    description: "",
    notes: "Instructions",
  })
  assert.equal(calls[2].url, `http://localhost:8080/projects/${projectId}`)
  for (const call of calls)
    assert.equal(
      new Headers(call.init?.headers).get("Authorization"),
      "Bearer account-a"
    )
})

test("membership writes explicitly support both moving and removal without sending chat messages", async () => {
  const bodies: unknown[] = []
  globalThis.fetch = async (_url, init) => {
    const body = JSON.parse(String(init?.body))
    bodies.push(body)
    return Response.json({
      chat_id: chatId,
      title: "Keep history",
      created_at: project.created_at,
      tokens: 10,
      project_id: body.project_id,
    })
  }
  assert.equal(
    (await assignChatProject("account-a", chatId, projectId)).projectId,
    projectId
  )
  assert.equal(
    (await assignChatProject("account-a", chatId, null)).projectId,
    null
  )
  assert.deepEqual(bodies, [{ project_id: projectId }, { project_id: null }])
})

test("stale project actions do not write and replacement accounts reject old project completions", async () => {
  let calls = 0
  globalThis.fetch = async () => {
    calls++
    return Response.json(project)
  }
  token = "account-b"
  await assert.rejects(
    createProject("account-a", { name: "Old", description: "", notes: "" }),
    /account changed/
  )
  assert.equal(calls, 0)
  token = "account-a"
  let finish!: (response: Response) => void
  globalThis.fetch = () =>
    new Promise<Response>((resolve) => {
      finish = resolve
    })
  const pending = createProject("account-a", {
    name: "Old",
    description: "",
    notes: "",
  })
  token = "account-b"
  finish(Response.json(project, { status: 201 }))
  await assert.rejects(pending, /account changed/)
})

test("a late unauthorized response from the former account cannot log out its replacement", async () => {
  let finish!: (response: Response) => void
  globalThis.fetch = () =>
    new Promise<Response>((resolve) => {
      finish = resolve
    })
  const request = apiFetch("/projects", {
    headers: new Headers({ Authorization: "Bearer account-a" }),
  })
  token = "account-b"
  finish(new Response("", { status: 401 }))
  await assert.rejects(request, /expired/)
  assert.equal(token, "account-b")
  assert.deepEqual(redirects, [])
})

test("an unauthorized response for the current account still clears it and redirects to login", async () => {
  globalThis.fetch = async () => new Response("", { status: 401 })
  await assert.rejects(apiFetch("/projects"), /expired/)
  assert.equal(token, null)
  assert.deepEqual(redirects, ["/login"])
})

test("composer choices validate readiness data and never persist picks for a replaced session", async () => {
  assert.deepEqual(parseComposerChoice({}), { model: "", budget: "auto" })
  assert.deepEqual(
    parseComposerChoice({ model: "checkpoint-42", budget: "high" }),
    { model: "checkpoint-42", budget: "high" }
  )
  for (const invalid of [null, [], { model: 5 }, { budget: "unbounded" }])
    assert.throws(() => parseComposerChoice(invalid))
  const calls: RequestInit[] = []
  globalThis.fetch = async (_url, init) => {
    calls.push(init ?? {})
    return Response.json({
      composer_choice: { model: "checkpoint-42", budget: "high" },
    })
  }
  assert.equal((await loadComposerChoice("account-a")).model, "checkpoint-42")
  token = "account-b"
  persistComposerChoice("account-a", { model: "old", budget: "low" })
  assert.equal(calls.length, 1)
  persistComposerChoice("account-b", { model: "current", budget: "med" })
  assert.equal(
    new Headers(calls[1].headers).get("Authorization"),
    "Bearer account-b"
  )
  assert.deepEqual(JSON.parse(String(calls[1].body)), {
    composer_choice: { model: "current", budget: "med" },
  })
})
