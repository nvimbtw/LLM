import { loadProjects, type Project, type ProjectPage } from "./projects"

export interface ProjectCatalogSnapshot {
  token: string | null
  projects: Project[]
  total: number
  nextOffset: number
  loading: boolean
  error: string | null
  revision: number
}

type ProjectLoader = (
  token: string,
  offset: number,
  signal?: AbortSignal
) => Promise<ProjectPage>

/** Owns the current account's paginated private projects, cancelling obsolete reads and ignoring late completions. */
export class ProjectCatalog {
  private snapshot: ProjectCatalogSnapshot = {
    token: null,
    projects: [],
    total: 0,
    nextOffset: 0,
    loading: false,
    error: null,
    revision: 0,
  }
  private listeners = new Set<() => void>()
  private request: AbortController | null = null
  private requestId = 0

  /** Creates an account catalogue with its HTTP loader; an injected loader exposes request races to tests. */
  constructor(private readonly load: ProjectLoader = loadProjects) {}

  /** Returns the immutable current snapshot for React's external-store subscription. */
  getSnapshot = (): ProjectCatalogSnapshot => this.snapshot

  /** Subscribes one reader and returns its listener cleanup. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** Publishes one coherent snapshot after an owned lifecycle transition. */
  private publish(next: ProjectCatalogSnapshot): void {
    this.snapshot = next
    this.listeners.forEach((listener) => listener())
  }

  /** Replaces account identity, drops private rows immediately and loads the replacement account's first page. */
  setSession(token: string | null): void {
    if (token === this.snapshot.token) {
      if (token && this.request?.signal.aborted) void this.refresh()
      return
    }
    this.request?.abort()
    this.requestId++
    this.publish({
      token,
      projects: [],
      total: 0,
      nextOffset: 0,
      loading: Boolean(token),
      error: null,
      revision: 0,
    })
    if (token) void this.refresh()
  }

  /** Reloads the first page and invalidates mounted project details; callers await settlement without losing prior rows on failure. */
  refresh = async (): Promise<void> => {
    await this.readPage(false)
  }

  /** Appends the next bounded page when available, deduplicating rows if the catalogue changed between requests. */
  loadMore = async (): Promise<void> => {
    if (
      this.snapshot.loading ||
      this.snapshot.nextOffset >= this.snapshot.total
    )
      return
    await this.readPage(true)
  }

  /** Performs one cancellable page read; only its originating session and latest request may publish results. */
  private async readPage(append: boolean): Promise<void> {
    const token = this.snapshot.token
    if (!token) return
    this.request?.abort()
    const controller = new AbortController()
    this.request = controller
    const requestId = ++this.requestId
    const offset = append ? this.snapshot.nextOffset : 0
    this.publish({
      ...this.snapshot,
      loading: true,
      error: null,
      revision: this.snapshot.revision + (append ? 0 : 1),
    })
    try {
      const page = await this.load(token, offset, controller.signal)
      if (requestId !== this.requestId || token !== this.snapshot.token) return
      const projects = append
        ? [
            ...new Map(
              [...this.snapshot.projects, ...page.projects].map((project) => [
                project.project_id,
                project,
              ])
            ).values(),
          ]
        : page.projects
      this.publish({
        ...this.snapshot,
        projects,
        total: page.total,
        nextOffset: page.offset + page.projects.length,
        loading: false,
      })
    } catch (failure) {
      if (
        requestId !== this.requestId ||
        token !== this.snapshot.token ||
        controller.signal.aborted
      )
        return
      this.publish({
        ...this.snapshot,
        loading: false,
        error:
          failure instanceof Error
            ? failure.message
            : "Unable to load projects.",
      })
    }
  }

  /** Cancels the provider's current request and prevents outstanding callbacks from publishing after cleanup. */
  dispose(): void {
    this.request?.abort()
    this.requestId++
  }
}
