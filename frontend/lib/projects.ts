import { apiFetch, getToken } from "./api"
import { parseChatSummary, parseResourceId, type ChatSummary } from "./chats"

export const PROJECT_PAGE_SIZE = 50
export const PROJECT_NAME_LIMIT = 120
export const PROJECT_DESCRIPTION_LIMIT = 2000
export const PROJECT_NOTES_LIMIT = 8000

export interface Project {
  project_id: string
  name: string
  description: string
  notes: string
  thread_count: number
  created_at: string
  updated_at: string
}

export interface ProjectFields {
  name: string
  description: string
  notes: string
}

export interface ProjectPage {
  projects: Project[]
  total: number
  limit: number
  offset: number
}

export interface ProjectDetail extends Project {
  threads: ChatSummary[]
  threads_total: number
  threads_limit: number
  threads_offset: number
}

/** Reads a JSON object at the project boundary, rejecting null, arrays and primitive payloads. */
function projectObject(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("The server returned invalid project data.")
  return value as Record<string, unknown>
}

/** Reads a bounded project text field using the backend's Unicode character limits. */
function projectText(value: unknown, limit: number): string {
  if (typeof value !== "string" || Array.from(value).length > limit)
    throw new Error("The server returned invalid project text.")
  return value
}

/** Reads a nonnegative integer count; malformed pagination or token totals reject. */
function projectCount(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)
    throw new Error("The server returned invalid project counts.")
  return value
}

/** Validates editable project fields before a write, preserving notes verbatim and trimming the name. */
export function validateProjectFields(fields: ProjectFields): ProjectFields {
  const name = fields.name.trim()
  if (!name || Array.from(name).length > PROJECT_NAME_LIMIT)
    throw new Error(`Use a project name of 1–${PROJECT_NAME_LIMIT} characters.`)
  if (Array.from(fields.description).length > PROJECT_DESCRIPTION_LIMIT)
    throw new Error(
      `Keep the description within ${PROJECT_DESCRIPTION_LIMIT} characters.`
    )
  if (Array.from(fields.notes).length > PROJECT_NOTES_LIMIT)
    throw new Error(
      `Keep notes and instructions within ${PROJECT_NOTES_LIMIT} characters.`
    )
  return { ...fields, name }
}

/** Keeps a conversation page within the remaining project archive after chats move or are deleted. */
export function projectThreadOffset(offset: number, total: number): number {
  return total === 0
    ? 0
    : Math.min(
        offset,
        Math.floor((total - 1) / PROJECT_PAGE_SIZE) * PROJECT_PAGE_SIZE
      )
}

/** Validates private project metadata, including identity, counts, timestamps and editable field bounds. */
export function parseProject(value: unknown): Project {
  const raw = projectObject(value)
  const name = projectText(raw.name, PROJECT_NAME_LIMIT)
  if (!name.trim())
    throw new Error("The server returned an empty project name.")
  if (
    typeof raw.created_at !== "string" ||
    !Number.isFinite(Date.parse(raw.created_at)) ||
    typeof raw.updated_at !== "string" ||
    !Number.isFinite(Date.parse(raw.updated_at))
  )
    throw new Error("The server returned invalid project dates.")
  return {
    project_id: parseResourceId(raw.project_id),
    name,
    description: projectText(raw.description, PROJECT_DESCRIPTION_LIMIT),
    notes: projectText(raw.notes, PROJECT_NOTES_LIMIT),
    thread_count: projectCount(raw.thread_count),
    created_at: raw.created_at,
    updated_at: raw.updated_at,
  }
}

/** Validates one bounded catalogue page, rejecting inconsistent page metadata. */
export function parseProjectPage(value: unknown): ProjectPage {
  const raw = projectObject(value)
  if (!Array.isArray(raw.projects))
    throw new Error("The server returned an invalid project list.")
  const limit = projectCount(raw.limit)
  const total = projectCount(raw.total)
  const offset = projectCount(raw.offset)
  if (
    limit < 1 ||
    limit > 200 ||
    raw.projects.length > limit ||
    raw.projects.length > Math.max(0, total - offset)
  )
    throw new Error("The server returned invalid project pagination.")
  return { projects: raw.projects.map(parseProject), total, limit, offset }
}

/** Validates a project and its bounded conversation page; summaries retain their project association. */
export function parseProjectDetail(value: unknown): ProjectDetail {
  const raw = projectObject(value)
  const project = parseProject(raw)
  if (!Array.isArray(raw.threads))
    throw new Error("The server returned invalid project conversations.")
  const threads_limit = projectCount(raw.threads_limit)
  const threads_total = projectCount(raw.threads_total)
  const threads_offset = projectCount(raw.threads_offset)
  if (
    threads_limit < 1 ||
    threads_limit > 200 ||
    raw.threads.length > threads_limit ||
    raw.threads.length > Math.max(0, threads_total - threads_offset)
  )
    throw new Error("The server returned invalid conversation pagination.")
  const threads = raw.threads.map(parseChatSummary)
  if (threads.some((thread) => thread.projectId !== project.project_id))
    throw new Error("The server returned conversations from another project.")
  return { ...project, threads, threads_total, threads_limit, threads_offset }
}

/** Makes a session-bound project request, forwarding cancellation and reporting actionable HTTP errors. */
async function projectRequest(
  token: string,
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  if (getToken() !== token)
    throw new Error("Your account changed. Please try again.")
  const response = await apiFetch(path, {
    ...init,
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok)
    throw new Error(
      response.status === 404
        ? "This project is no longer available."
        : "Unable to update or load this project. Please try again."
    )
  if (getToken() !== token)
    throw new Error("Your account changed. Please try again.")
  return response
}

/** Loads one catalogue page for the supplied account, with abortable browser ownership. */
export async function loadProjects(
  token: string,
  offset = 0,
  signal?: AbortSignal
): Promise<ProjectPage> {
  const response = await projectRequest(
    token,
    `/projects?limit=${PROJECT_PAGE_SIZE}&offset=${offset}`,
    { signal }
  )
  return parseProjectPage(await response.json())
}

/** Loads metadata and one conversation page for the selected project; navigation may abort the request. */
export async function loadProject(
  token: string,
  projectId: string,
  offset = 0,
  signal?: AbortSignal
): Promise<ProjectDetail> {
  const response = await projectRequest(
    token,
    `/projects/${parseResourceId(projectId)}?limit=${PROJECT_PAGE_SIZE}&offset=${offset}`,
    { signal }
  )
  return parseProjectDetail(await response.json())
}

/** Creates a private project after validating its editable text; returns validated persisted metadata. */
export async function createProject(
  token: string,
  fields: ProjectFields
): Promise<Project> {
  const response = await projectRequest(token, "/projects", {
    method: "POST",
    body: JSON.stringify(validateProjectFields(fields)),
  })
  return parseProject(await response.json())
}

/** Updates the selected project's editable fields without changing its conversation membership. */
export async function updateProject(
  token: string,
  projectId: string,
  fields: ProjectFields
): Promise<Project> {
  const response = await projectRequest(
    token,
    `/projects/${parseResourceId(projectId)}`,
    { method: "PATCH", body: JSON.stringify(validateProjectFields(fields)) }
  )
  return parseProject(await response.json())
}

/** Deletes only a private project; the backend preserves its conversations in the ordinary archive. */
export async function deleteProject(
  token: string,
  projectId: string
): Promise<void> {
  await projectRequest(token, `/projects/${parseResourceId(projectId)}`, {
    method: "DELETE",
  })
}
