# The docs site (`/docs`)

The markdown in [`docs/`](../) is rendered by this module at
[`app/docs/[[...slug]]/page.tsx`](../../frontend/app/docs) — public, outside the
`(app)` group, so reading the docs needs no account and does not depend on the
signed-in chrome.

- [`lib/docs.ts`](../../frontend/lib/docs.ts) walks `../docs` and maps files to
  routes: `docs/README.md` → `/docs`, `docs/backend/README.md` → `/docs/backend`,
  `docs/backend/schema.md` → `/docs/backend/schema`. A directory with no
  `README.md` gets a generated index of its children instead of a 404.
- **Build time only.** `generateStaticParams` prerenders every page, so the
  filesystem outside this module is never touched by a running server. The
  consequence: a docs-only commit needs a frontend rebuild before it shows on
  the deployed site. The frontend container uses the repository root as its
  build context and copies `docs/`, `README.md`, and `AGENTS.md` into the paths
  this renderer expects; Compose mounts those paths in development too.
- **Relative links are rewritten** to routes, so the same markdown reads
  correctly on the site and in an editor. A link that resolves to nothing
  **fails the build** — deliberately; that is what keeps the docs from drifting
  away from the tree they describe.
- `AGENTS.md` at the repo root is rendered too, at `/docs/agents`, via
  `EXTERNAL_PAGES` in `lib/docs.ts`. Add to that map to publish another file
  from outside `docs/`.

- **Prose styling is `.doc-prose` in [`app/globals.css`](../../frontend/app/globals.css)**,
  hand-set against the app's own tokens rather than `@tailwindcss/typography` —
  the plugin ships its own type scale and greys, and a page set in them reads
  like a different product. Headings take the display serif, body copy is capped
  at 68 characters, and tables and code are allowed to break that measure.
- **HTML comments are stripped** before rendering. `<!-- TODO: … -->` is
  invisible in every other markdown viewer and stays invisible here.

Adding a doc requires no frontend change: write the markdown, and the page and
its sidebar entry appear on the next build.
