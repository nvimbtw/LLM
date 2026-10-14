import { test } from "node:test"
import assert from "node:assert/strict"
import {
  parseProject,
  parseProjectPage,
  parseProjectDetail,
  validateProjectFields,
  projectThreadOffset,
} from "./projects"
import { parseChatSummary } from "./chats"

const projectId = "00000000-0000-4000-8000-000000000001"
const otherId = "00000000-0000-4000-8000-000000000002"
const project = {
  project_id: projectId,
  name: "Research",
  description: "Private notes",
  notes: "Keep links\nUse metric units",
  thread_count: 1,
  created_at: "2026-10-14T12:00:00Z",
  updated_at: "2026-10-14T12:00:00Z",
}
const chat = {
  chat_id: otherId,
  title: "First topic",
  created_at: project.created_at,
  tokens: 42,
  project_id: projectId,
}

test("project metadata, pagination and thread membership preserve the agreed wire contract", () => {
  assert.deepEqual(parseProject(project), project)
  assert.deepEqual(
    parseProjectPage({ projects: [project], total: 1, limit: 50, offset: 0 }),
    { projects: [project], total: 1, limit: 50, offset: 0 }
  )
  const detail = parseProjectDetail({
    ...project,
    threads: [chat],
    threads_total: 1,
    threads_limit: 50,
    threads_offset: 0,
  })
  assert.equal(detail.notes, project.notes)
  assert.equal(detail.threads[0].projectId, projectId)
  assert.equal(detail.threads[0].tokens, 42)
  assert.equal(detail.threads[0].id, otherId)
})

test("ordinary and older chat summaries remain compatible while malformed project identities reject", () => {
  assert.equal(parseChatSummary({ ...chat, project_id: null }).projectId, null)
  const older = {
    chat_id: chat.chat_id,
    title: chat.title,
    created_at: chat.created_at,
  }
  assert.equal(parseChatSummary(older).projectId, null)
  assert.equal(parseChatSummary(older).tokens, 0)
  for (const changes of [
    { chat_id: "../../account" },
    { project_id: "bad-id" },
    { tokens: -1 },
    { tokens: Infinity },
    { created_at: "yesterday" },
  ])
    assert.throws(() => parseChatSummary({ ...chat, ...changes }))
})

test("project field validation uses Unicode characters and preserves notes verbatim", () => {
  const fields = {
    name: "  Research  ",
    description: "",
    notes: " leading\nnotes ",
  }
  assert.deepEqual(validateProjectFields(fields), {
    ...fields,
    name: "Research",
  })
  assert.equal(
    validateProjectFields({ ...fields, name: "🧠".repeat(120) }).name.length,
    240
  )
  for (const changes of [
    { name: "  " },
    { name: "x".repeat(121) },
    { description: "x".repeat(2001) },
    { notes: "x".repeat(8001) },
  ])
    assert.throws(() => validateProjectFields({ ...fields, ...changes }))
})

test("malformed private payloads, excessive pages and cross-project thread rows reject", () => {
  for (const invalid of [
    null,
    [],
    1,
    { ...project, name: "" },
    { ...project, notes: 123 },
    { ...project, thread_count: -1 },
    { ...project, project_id: "not-a-uuid" },
    { ...project, updated_at: "bad" },
  ])
    assert.throws(() => parseProject(invalid))
  for (const changes of [
    { limit: 201 },
    { limit: 0 },
    { total: 0 },
    { offset: -1 },
    { projects: "list" },
  ])
    assert.throws(() =>
      parseProjectPage({
        projects: [project],
        total: 1,
        limit: 50,
        offset: 0,
        ...changes,
      })
    )
  assert.throws(
    () =>
      parseProjectDetail({
        ...project,
        threads: [{ ...chat, project_id: otherId }],
        threads_total: 1,
        threads_limit: 50,
        threads_offset: 0,
      }),
    /another project/
  )
})

test("deleting or moving the last row on a later thread page returns to an existing page", () => {
  assert.equal(projectThreadOffset(100, 100), 50)
  assert.equal(projectThreadOffset(50, 50), 0)
  assert.equal(projectThreadOffset(50, 51), 50)
  assert.equal(projectThreadOffset(150, 0), 0)
})
