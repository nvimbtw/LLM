import { readFileSync, readdirSync, statSync } from "node:fs"
import { dirname, join, normalize, relative, resolve, sep } from "node:path"

/**
 * The `docs/` tree lives outside this module, at the repo root. It is read at
 * build time only — every docs page is statically generated (see
 * `generateStaticParams`), so nothing here runs in a deployed server and no
 * path outside the app needs to be traced into the bundle.
 */
const DOCS_ROOT = resolve(process.cwd(), "..", "docs")
const REPO_ROOT = resolve(process.cwd(), "..")

/**
 * Files outside `docs/` that the docs site still renders, mapped to their slug.
 * The repo's two front doors: `README.md` for a person arriving at the project,
 * `AGENTS.md` for anyone about to change it. Both are linked from the docs and
 * would otherwise be dead links on the site — and rendering them here means
 * their links are covered by the build-time check like everything else.
 */
const EXTERNAL_PAGES: Record<string, { slug: string[]; label: string }> = {
  // Labelled by the file they are, not by their heading: the README's heading
  // is the product's name, which in a sidebar reads as a section rather than as
  // the document it is.
  "README.md": { slug: ["readme"], label: "README" },
  "AGENTS.md": { slug: ["agents"], label: "AGENTS.md" },
}

export interface DocPage {
  /** Route segments after `/docs`. The tree root (`docs/README.md`) is `[]`. */
  slug: string[]
  /** Absolute path of the markdown file. */
  file: string
  /** First `# ` heading, falling back to the filename. */
  title: string
}

export interface NavEntry {
  /** Label in the sidebar. Often shorter than the page's own title. */
  label: string
  slug: string[]
  children?: NavEntry[]
}

export interface DocSection {
  title: string
  entries: NavEntry[]
}

const MARKDOWN = /\.md$/

function listMarkdown(dir: string): string[] {
  return readdirSync(dir)
    .filter((name) => MARKDOWN.test(name))
    .sort()
    .map((name) => join(dir, name))
}

function listDirectories(dir: string): string[] {
  return readdirSync(dir)
    .filter((name) => statSync(join(dir, name)).isDirectory())
    .sort()
    .map((name) => join(dir, name))
}

function readTitle(file: string): string {
  const heading = readFileSync(file, "utf8").match(/^#\s+(.+)$/m)
  if (heading) return heading[1].trim()
  return file.split(sep).pop()!.replace(MARKDOWN, "")
}

/** `docs/backend/README.md` → `["backend"]`; `docs/README.md` → `[]`. */
function slugForFile(file: string): string[] {
  const external = EXTERNAL_PAGES[relative(REPO_ROOT, file)]
  if (external) return external.slug
  const segments = relative(DOCS_ROOT, file).replace(MARKDOWN, "").split(sep)
  if (segments[segments.length - 1] === "README") segments.pop()
  return segments
}

function toPage(file: string): DocPage {
  return { slug: slugForFile(file), file, title: readTitle(file) }
}

/**
 * Every renderable page: the docs tree plus `EXTERNAL_PAGES`. Directories
 * without a `README.md` still get a page — an index listing their children —
 * so a link to the directory is never dead.
 */
export function allPages(): DocPage[] {
  const pages = [
    ...listMarkdown(DOCS_ROOT),
    ...listDirectories(DOCS_ROOT).flatMap(listMarkdown),
    ...Object.keys(EXTERNAL_PAGES).map((path) => join(REPO_ROOT, path)),
  ].map(toPage)

  const covered = new Set(pages.map((page) => page.slug.join("/")))
  const sectionIndexes = listDirectories(DOCS_ROOT)
    .map((dir) => relative(DOCS_ROOT, dir))
    .filter((name) => !covered.has(name))
    .map((name) => ({
      slug: [name],
      file: join(DOCS_ROOT, name),
      title: titleCase(name),
    }))

  return [...pages, ...sectionIndexes]
}

export function findPage(slug: string[]): DocPage | undefined {
  const wanted = slug.join("/")
  return allPages().find((page) => page.slug.join("/") === wanted)
}

/**
 * The sidebar: two groups, `Documentation` and `Modules`.
 *
 * Labels here are deliberately not the pages' own titles. A doc's `# ` heading
 * is written to be a masthead — "LLM engine — PyTorch training, inference,
 * evaluation" — and setting that as a tracked uppercase eyebrow gives you three
 * wrapped lines of shouting. The rail gets the directory name; the page keeps
 * its title.
 *
 * A directory is a *module* when the repo has code under that name, so
 * `docs/llm-engine/` files under Modules and `docs/decisions/` does not. Derived
 * rather than listed: a new module documents itself into the right group the
 * day it exists.
 */
export function navigation(): DocSection[] {
  const pages = allPages()
  const at = (path: string) =>
    pages.find((page) => page.slug.join("/") === path)

  // Files sitting directly in `docs/`, plus the external ones. Directory pages
  // land here too by path — they are listed as entries below instead.
  const loose = pages.filter((page) => {
    const parent = dirname(page.file)
    return !isDirectory(page) && (parent === DOCS_ROOT || parent === REPO_ROOT)
  })

  const directories = listDirectories(DOCS_ROOT).map((dir) =>
    relative(DOCS_ROOT, dir)
  )

  const entryForDirectory = (name: string): NavEntry | undefined => {
    const index = at(name)
    if (!index) return undefined
    const children = pages
      .filter((page) => page.slug.length === 2 && page.slug[0] === name)
      .map((page) => ({ label: page.title, slug: page.slug }))
    return {
      label: titleCase(name),
      slug: index.slug,
      children: children.length > 0 ? children : undefined,
    }
  }

  const isModule = (name: string) => exists(join(REPO_ROOT, name))
  const entries = (names: string[]) =>
    names.map(entryForDirectory).filter((entry) => entry !== undefined)

  return [
    {
      title: "Documentation",
      entries: [
        ...loose.map((page) => ({
          // The tree root is the introduction to the tree, whatever its own
          // heading happens to say.
          label: navLabel(page),
          slug: page.slug,
        })),
        ...entries(directories.filter((name) => !isModule(name))),
      ],
    },
    {
      title: "Modules",
      entries: entries(directories.filter(isModule)),
    },
  ]
}

/**
 * The sidebar's name for a loose page. The tree root is the introduction to the
 * tree whatever its own heading says, and the external pages are named for the
 * file they are.
 */
function navLabel(page: DocPage): string {
  if (page.slug.length === 0) return "Introduction"
  const external = EXTERNAL_PAGES[relative(REPO_ROOT, page.file)]
  return external ? external.label : page.title
}

function titleCase(name: string): string {
  return name.replace(
    /(^|-)([a-z])/g,
    (_, separator, letter) => (separator ? " " : "") + letter.toUpperCase()
  )
}

export function isDirectory(page: DocPage): boolean {
  return statSync(page.file).isDirectory()
}

/** Author notes (`<!-- TODO: … -->`) are invisible in any markdown viewer and
 *  must stay invisible here. */
const HTML_COMMENT = /<!--[\s\S]*?-->\n?/g

export function read(page: DocPage): string {
  if (!isDirectory(page)) {
    return readFileSync(page.file, "utf8").replace(HTML_COMMENT, "")
  }

  // A directory with no README: list what is in it rather than 404.
  const children = listMarkdown(page.file).map(toPage)
  const lines = children.map(
    (child) => `- [${child.title}](${child.slug[child.slug.length - 1]}.md)`
  )
  return [
    `# ${page.title}`,
    "",
    lines.length > 0 ? lines.join("\n") : "_Nothing here yet._",
    "",
  ].join("\n")
}

/**
 * Resolve a relative markdown link against the page that contains it.
 *
 * Returns a route when the target is a rendered page, `null` when the target
 * exists but has no page (a source file, say — rendered as plain text), and
 * throws when it resolves to nothing at all. The throw is deliberate: an
 * unresolvable link fails `next build` rather than shipping a dead link, which
 * is the cheapest guard against docs drifting away from the tree they describe.
 */
export function resolveLink(from: DocPage, href: string): string | null {
  const base = isDirectory(from) ? from.file : dirname(from.file)
  const [path, hash] = href.split("#")
  const target = normalize(resolve(base, path))

  const page = allPages().find((candidate) => candidate.file === target)
  if (page) return ["/docs", ...page.slug].join("/") + (hash ? `#${hash}` : "")

  if (exists(target)) return null

  throw new Error(
    `Dead link in ${relative(REPO_ROOT, from.file)}: "${href}" resolves to ` +
      `${relative(REPO_ROOT, target)}, which does not exist. Fix the link or ` +
      `add the file — docs links are checked at build time.`
  )
}

function exists(path: string): boolean {
  try {
    statSync(path)
    return true
  } catch {
    return false
  }
}

export interface Heading {
  id: string
  text: string
  depth: 2 | 3
}

/** Stable, human-readable anchor ids. Must agree with the renderer's `h2`/`h3`. */
export function headingId(text: string): string {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
}

const FENCE = /^```/

/**
 * The `##`/`###` headings of a doc, for the contents rail. Fenced code is
 * skipped — a `# comment` inside a shell block is not a section.
 */
export function headings(source: string): Heading[] {
  let inFence = false
  const found: Heading[] = []

  for (const line of source.split("\n")) {
    if (FENCE.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue

    const match = line.match(/^(#{2,3})\s+(.+?)\s*$/)
    if (!match) continue

    const text = match[2].replace(/`/g, "").replace(/\*\*/g, "")
    found.push({ id: headingId(text), text, depth: match[1].length as 2 | 3 })
  }
  return found
}

/**
 * Split the leading `# Title` off a doc. The title is set by the page masthead,
 * so leaving it in the body would print it twice.
 */
export function splitTitle(source: string): { title?: string; body: string } {
  const match = source.match(/^#\s+(.+)\n/)
  if (!match) return { body: source }
  return { title: match[1].trim(), body: source.slice(match[0].length) }
}
