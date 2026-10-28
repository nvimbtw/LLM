# Appearance

## Colour

The palette is Kanagawa (Dragon in dark, a warm-paper light mode) — *our*
Kanagawa: derived from rebelot's and then moved where a screen full of small
text needed it moved. Upstream is lineage, not a source of truth; do not
"restore" a value to it.

It lives in [`app/globals.css`](../../frontend/app/globals.css) in three layers,
which exist so that what a colour **is** stays separate from what **uses** it:

| layer | tokens | holds |
| --- | --- | --- |
| 1. palette | `--kanagawa-dragon-*`, `--kanagawa-lotus-*` | the hexes, named in Kanagawa's vocabulary. Never referenced by a component. |
| 2. theme | `--background-light` / `--background-dark`, one pair per role | which colour each role takes in each mode, both modes side by side |
| 3. active mode | `--background`, … | `:root`/`.theme-light` resolve the light half, `.dark` the dark one. The only layer components see. |

So trying something is a one-line edit in layer 2 — point a role at a different
palette entry and both the app and the `/colors` fixture follow. A hex belongs
in layer 1 and nowhere else; a component reads layer 3 and nothing else. The
reasoning for a *value* is written on it in the palette, the reasoning for a
*role's choice* on it in the theme.

`.theme-light` is `:root`'s twin, declared in the same rule so the two cannot
drift: it lets a subtree render the light theme while the app is in the dark
one, which is what the `/colors` fixture uses to show both at once.

**Selected text** is a role like any other — `--selection` / `--selection-foreground`,
resolved in `::selection` in the base layer. Both modes take Kanagawa's own
`Visual` colour (waveBlue1 under Dragon, lotusViolet3 under Lotus), which is the
one role where upstream's answer is exactly the right one. The foreground is
pinned rather than inherited, so the ratio is a fact rather than a function of
whatever happens to be selected — which is also why selecting coloured text
drops its colour, the same as in an editor.

Every pair is checked by

```sh
cd frontend && node scripts/audit-contrast.mjs
```

which prints the contrast of every token against every surface it lands on, in
both modes, against three floors: **4.5** for text (the app sets almost nothing
at "large" sizes, and its eyebrows are small and letterspaced), **3.0** for
graphical objects that carry meaning (chart marks, form-control boundaries), and
**1.8** for hairlines, which carry none and only have to survive a bright screen.
Exactly one row fails by design — `--tier-low`, for the reason written on it.

WCAG is the floor, not the whole test, and

```sh
cd frontend && node scripts/audit-theme.mjs
```

is the perceptual audit that sits on top of it: the OKLCH ladder of every token,
APCA Lc (vendored apca-w3) with the minimum font size each body pair supports,
and a chromatic-aberration check on every text pair. That check is why the body
ink is a red-violet *whisper* (#4f4f58) rather than the old full violet: a cool
text hue on a warm ground makes the lens hunt between two focal planes at every
letter edge, and the rule catches it as `energy ≥ 3.0` (hue gap × text chroma).
`--primary` links score high on that measure too and are exempt only because
they are short, underlined and spaced — the flag applies to body-carrying pairs.

Both read the stylesheet through
[`scripts/theme-tokens.mjs`](../../frontend/scripts/theme-tokens.mjs), which
resolves the three layers the way the browser does and hands them one flat
`{ role: "#rrggbb" }` map per mode — so an audit measures what ships, not what
a regex could find in one block.

If you change a token, run both, and keep the hue: the fixes they prompt should
be lightness moves, which is what keeps the scheme recognisably Kanagawa.

A palette change is judged against the old values on the `/colors` fixture
page — both themes, the token ladder, and the frozen pre-change snapshot with
its measured deltas.

## Sizing

Facts ledgers use the default `PageShell` board width, including tokenizer lists,
detail and creation pages. Prose editing surfaces such as project notes retain
reading width. Inline metadata fields do not make a facts ledger a prose page.

The same two-layer split the colour tokens got, applied to everything with a
size. In `globals.css`:

1. **The scales** (`@theme`) — what a size *is*. The type steps `2xs`…`5xl`, each
   carrying its own line-height and letter-spacing; the radius steps derived
   from `--radius`; the 4px spacing unit.
2. **The roles** (`:root`, under `SIZING ROLES`) — what *uses* a size.
   `--type-eyebrow-tracking`, `--space-sheet`, `--radius-sheet`,
   `--elevation-overlay`. Components reference roles, never raw steps.

There is no third layer: a size is a size in both themes.

### The two composite utilities

Two devices appear so often, and drifted so far, that they are classes rather
than sets of tokens to remember:

- **`eyebrow`** — the small uppercase label over a section, a stat, a field.
  Sets size, weight, letter-spacing, word-spacing and the transform. Colour is
  *not* included: most take `text-muted-foreground`, the sidebar's take a
  sidebar token. Use `tag` for its mono cousin inside chat traces, which is
  sized in `em` because it sits in text that is itself scaled.
- **`figure`** — the number in a stat: display face, `font-variant-numeric:
  tabular-nums`, one weight. Deliberately sets **no font-size**; the call site
  picks the step (`figure text-2xl`) so the step's own tracking applies. Never
  add `tracking-tight` to a figure — the scale already tracks by size, and the
  constant overrides it.

Before this pass the eyebrow had 17 distinct spellings across 69 uses, with 7
different letter-spacings for one element; the figure had 10 sizes, half of them
fighting the scale with `tracking-tight`.

### Radius roles

`--radius-sheet`, `--radius-control`, `--radius-control-sm`, `--radius-marker`,
and — pinned off the scale — `--radius-composer` and `--radius-bubble`, which
follows it. Chat is the one place the app is allowed to be round, and those
values are absolute so that squaring the app's corners cannot reach into the
composer.

No component spells `rounded-lg`/`rounded-md` any more. The only bare radius
utilities left are `rounded-full` (pills, avatars), `rounded-none`, and three
micro values on chart bars and inline code where the role step would round them
into blobs.

## Surfaces, elevation and texture

The approved design uses a crisp small offset instead of diffuse elevation.
`shadow-overlay`, `shadow-sheet` and `shadow-sheet-hover` resolve to **1px right,
2px down, zero blur/spread**, mixed from light ink at 24% or the deepest dark
palette ground at 80%. `shadow-raised` is a single crisp lower edge. Keep these
roles in the theme layer; do not add default `shadow-sm`/`md`/`lg`/`xl`, inset
highlights, ambient glows or per-component shadow recipes.

A focused actionable sheet can retain `bg-card`, a restrained functional edge
and the sheet role. Summaries, catalogue entries and statistics stay open: no
surrounding card, row separators or outer band rules. Admin workspaces use
`Panel variant="open"`, aligned serif headings and content. Training inputs and
select triggers use their existing `inline` variants; ordinary outlined controls
keep clear functional boundaries and keyboard focus.

Menus/selects use compact neutral surfaces, aligned labels and small icons.
Meaningful groups use spacing. The composer and toasts use the same frosted
material (`bg-card/75 backdrop-blur-2xl backdrop-saturate-150`); the crisp shadow
separates them without removing that glass. Settings and notification dialogs
share one opaque background/frame, border, pane divider and vertical selection
marker. Composer-related popovers retain its softer corners.

`PageHeader` owns the display title, back link, functional metadata, actions and
faint grey `SectionTexture`, with no decorative bottom rule. The sidebar Slate
brand header and composer's command/plan upper region use the same neutral field.
`EmptyState` uses bounded warm-clay texture. Other section headings, populated
lists and whole ordinary pages do not gain texture. The SVG is decorative and
pointer-transparent. `--texture-opacity` (0.065), `--texture-warm-opacity` (0.16)
and `--texture-glyph-size` own its rendering. The landing artwork below retains
its separate geometry and motion contract.

Colour roles are assigned in [Design](design.md#colour-surfaces-and-texture):
steel blue for action/selection/focus, ochre for models/vocabularies, mauve for
chat, aqua for practice, sage for benchmarks/labelled success, amber for warnings,
soft red for failures/destruction and clay for empty texture. Domain colours
are small accents; text/status labels carry meaning independently of colour.

## New-chat figure

The new-chat heading is an ASCII mountain range above a compact **SLATE**
wordmark. `components/ui/figure-slate.tsx` authors both on one character grid;
glyph density describes rock faces, shadow and distant ridges. The imported
stylesheet uses primary and throbber-spark in dark mode, deeper throbber-base
ink in light mode, and foreground for the wordmark. Two empty character rows
separate the mountains from the lettering.

Only the new-chat page mounts `NewChatScene`. A very muted field of characters
fills its content area, aligned with the artwork's glyph size and spacing.
`SlateTexture` repeats an SVG tile; `ResizeObserver` aligns and scales it to the
figure when the page, sidebar or composer changes size. Occupied artwork cells
are masked out of the field so random characters never overprint the mountains
or lettering. Pointer movement reveals the original artwork colours through a
soft radial mask. Surrounding characters keep the same muted ink during hover.
`usePointerLight` writes coordinates through refs. Leaving the page area or
holding the mouse still for 240ms fades the reveal smoothly over another 900ms.
Up to eight positions leave a wider, 1.8-second trail through the artwork colours;
the frame loop stops when all positions expire and is cancelled on unmount.

On mount, a single CSS mask sweeps left to right across **SLATE**, then settles
back to the muted base. Touch and reduced-motion modes retain a static figure
and disable both the sweep and hover reveal. Existing conversations never mount
this scene. All artwork is decorative and ignores pointer events; the heading
supplies its accessible name, while the composer retains its existing controls.
The figure's aspect ratio is reserved before paint.

The standalone `FigureSlate` retains its six fixed clusters of twelve ridge
characters, shimmering in a 24-second loop. The new-chat scene disables that
loop in favour of the pointer reveal and one-time wordmark sweep.

## Activity figures and state

`components/ui/throbber.tsx` selects the classic trellis or optional circular
token art using the account's **Use live throbber** setting (off by default).
See [Live throbber](live-throbber.md) for its frontend wgpu/WASM renderer, token
contract, motion, fallback, showcase and build. Status text is owned by its
existing components and is unchanged by this selection.
Live art has one fixed 32 × 32 pixel footprint throughout the app, controlled by
`--size-live-throbber`; its fluid detail needs slightly more room than the trellis.

The classic `components/ui/trellis-throbber.tsx` is a **figure**, not an icon: a hexagonal field of
nineteen dots with a **tree grown across it**. Ink leaves the centre, runs along
lattice edges and branches; each dot it reaches lights and swells while the rest
of the field goes on breathing. It is one fixed footprint everywhere,
`THROBBER_W` × `THROBBER_H` (28×25), and callers cannot resize it:
`className` is for margin, placement and colour only. That is deliberate — it
used to be sized per call site, and the result was a different figure in every
row, none of which looked like the same object.

**The tree is grown, not drawn.** A fresh one is grown by random walk at the
start of every pass — every state but `generating`, which walks instead — so the
mark is never the same twice. What is fixed is the *character* of the growth,
and that is what carries the state — see `GROWTH`: `limbs` (directions leaving
the root), `branch` (how far an inner dot splits) and `depth` (how far ink
travels). One route walked without stopping is a token being sampled; a broad
shallow bush is several paths held open; the full spanning net is every
parameter moving at once.

Rules governing changes to it:

- **Geometry is built once at mount.** Every lattice edge is rendered up front
  and a tree is merely a choice of which are visible this pass, so a frame
  writes opacities and one dash length and never path data. An earlier design
  rebuilt paths per frame, and the browser re-parsing those was the whole reason
  it ran differently on different machines.
- **The phase is integrated, never read off the clock.** Each frame adds
  `rate × dt`, and a mode change blends the *rate*, so the figure changes pace
  rather than jumping.
- **Growing allocates, so it happens on a pass boundary** — about once a second
  — and never inside a frame. The PRNG is seeded per instance and per pass, so a
  tree is re-growable rather than merely random.
- **A state is three growth numbers plus an advance rule**, not a bespoke
  animation. Add a state by adding entries to `GROWTH`, `RATES` and `TINT`, and
  a branch to the advance `switch` in `draw` only if the new state genuinely
  cannot be told by how its tree grows. `generating` is the one state that
  bought more than that, and the snake section of the file is what it cost —
  that is the bar for the next one.
- **`generating` is a snake, not a tree.** The head steps to a neighbouring dot
  and the tail lets the oldest one go, so six dots travel the field with no seam
  in them. It grows nothing and never re-starts, because a token stream does
  not: the tree it used to grow put a fresh start in the middle of a reply once
  a second. The walk is planned ahead on the pass boundary and never in a frame,
  and it may not step where its own body will still be standing.
- **A state change is answered at once — except by the snake.** Every tree state
  grows the new one immediately rather than waiting out the pass in progress,
  because repeated growth is its whole language. Leaving `generating` instead
  runs the head home to the centre and drains the body in behind it, half a
  second in which the mark still says *generating*: a stream is put away, not
  cut. What is left standing is a lit dot at the centre, which is where the
  incoming figure's first frame starts anyway, so the two join up.

The `/throbber` page shows every state side by side, and the chat's status row
with a live count behind it — the only place either of the row's movements can
be watched without a GPU at the other end. Earlier candidates — a
rule, a stanza of set text, a comb of stems, arcs over a baseline, a folded
line, a causal-mask triangle, orbiting rings, a square lattice of cells — are
recorded with their reasons in the git history of `components/throbber/`; the
short version is that a figure with no silhouette reads as a picture of an axis,
and a figure with one parameter cannot tell eight states apart.

So use it only where a figure belongs — somewhere the waiting *is* the subject, with
room for it:

- the chat thread (loading, and the thinking/tool/generating status row),
- the training hero corner (`TrainingHero`),
- the queue's now-band,
- the benchmark empty state.

Elsewhere, use concise state words, useful counts or actual progress. Standalone
status dots are removed, including in the sidebar and beside already-labelled
activity. Do not replace them with another arbitrary spinner. This does not
remove semantic chart/diagram marks or the approved exchange rail's strokes.
