# Slate design direction

Slate should feel like **an LLM workspace with the quietness of a notebook**.
It brings model building, data preparation, benchmarks, task gyms and chat into
one place. The interface should make the work legible, make live state visible,
and let the content carry the weight.

This guide records the product and visual direction that has emerged from the
standalone `frontend/slate-system-design.html` sketch, the frontend
implementation docs, and the design issues in Discord. [Appearance](appearance.md)
is the detailed reference for theme, type, sizing and surfaces; this page is the
short guide to the choices those rules should serve.

## The character

Aim for editorial minimalism that feels professional, useful and considered.
The page should feel clean without feeling empty or generic. Preserve the
information and flow people use; remove repeated chrome, decoration and copy
that adds no decision-making value. A visible phrase should be content, a
control label, an instruction, or useful state.

Each page should answer three questions through its hierarchy and content:

- Where am I?
- What is this for?
- What is the next useful action?

## Page structure

- Use the shared `PageHeader` for the title, back link, actions and useful
  metadata. Do not add a sentence under the title just to explain the page.
  Counts, status, elapsed time and conversation figures are useful when they
  help someone act; render them as quiet metadata. Descriptions that belong to
  a resource remain editable content on that resource.
- Give catalogues and facts ledgers the available board width. Reserve reading
  width for prose and text-editing surfaces. A few inline fields do not turn a
  facts page into a prose page.
- Prefer single-column lists and open ledger rows for catalogues and summaries.
  Use hairline rules to group related facts instead of wrapping every region in
  a card. Keep a sheet surface for a focused, actionable composition; controls,
  dialogs and confirmations still need clear interactive boundaries.
- Keep pages and controls aligned. A form, diagram or chart should have a
  readable sequence, consistent edges and spacing, and a clear relationship to
  the data it represents.
- Let durable resources be browsed as a list, opened as a detail page, and
  created on a dedicated page when the workflow has meaningful fields. Prefer
  editing resource fields in place with clear Save/Cancel behavior over a
  second modal form. Keep transient choices and confirmations in dialogs.
- Use URLs that name the domain and distinguish a resource from an execution.
  The canonical route map is in [Browser routes](routing.md).

## Visual language

- The palette follows customized Kanagawa: warm paper in light mode and Dragon
  tones in dark mode. Use semantic theme roles so both modes remain coherent;
  do not put palette hex values directly in components.
- Fraunces gives display headings and figures their editorial voice. Geist is
  for interface text; Geist Mono is for technical traces, tags and metrics.
  Keep labels quiet and consistent, and use tabular numerals for changing
  figures.
- Use restrained typography, open space and thin rules to establish hierarchy.
  Reserve accent colour for meaning and interaction. Pair status colour with
  text or an icon, keep keyboard focus visible, and preserve contrast in both
  themes.
- Distinguish a sheet from a ledger. A sheet has a surface, edge and
  theme-aware elevation because it holds an actionable group. A ledger is
  furniture: open rows and rules that let the facts remain prominent. Do not
  give summaries and controls equal visual weight by boxing both.
- Keep corners restrained across the application. The composer and chat
  bubbles have their own softer shape; that exception should stay recognizable.

## Motion and distinctive detail

Use motion when it explains a real state change or makes an interaction easier
to understand. A training or generation diagram should show what each operation
does, keep its arrows and blocks aligned, and follow the actual process. Make
expressive movement smooth and restrained; keep the interface useful with
reduced motion and on narrow screens.

Slate can have texture and a distinctive mark when they give a surface character.
Keep these treatments muted and local to the page that owns them: the
authentication pages use a quiet field of characters, while the new-chat page
has its own mountain-and-wordmark figure. Do not spread decorative artwork
through ordinary pages or let it compete with controls and data.

## Ideas that remain exploratory

The standalone sketch also proposes a workspace pulse and command search,
model comparison, a dataset lineage rail, undo/recovery affordances and a
mobile bottom-sheet workflow. These are product ideas, not settled requirements
or a mandate for a broad redesign. The Discord discussions also contain
domain-specific proposals for model architecture and processor authoring. Keep
their detailed contracts in those threads, and track each as individual work
before treating it as implementation scope.

## References

- [Appearance](appearance.md) — current theme roles, sizing, surfaces, artwork
  and motion details.
- [Frontend development](development.md) — shared components and implementation
  conventions.
- [Authentication pages](authentication.md) — the focused form and character
  texture treatment.
- [Browser routes](routing.md) — canonical resource and execution paths.
- `frontend/slate-system-design.html` — the standalone original product thesis,
  surface map and exploratory roadmap.
- Related feature and system discussions: [FEAT-1](https://discord.com/channels/1538659898383011961/1559193308117344298)
  on live status art, [FEAT-2](https://discord.com/channels/1538659898383011961/1559882446177304607)
  on project pages and inline editing, [FEAT-3](https://discord.com/channels/1538659898383011961/1559933878020472866)
  on processor authoring, [DOC-5](https://discord.com/channels/1538659898383011961/1560318859095834664)
  on modular model architecture, and [DOC-6](https://discord.com/channels/1538659898383011961/1560320001989152781)
  on data processor design. These are workflow and domain specs, not visual
  language rules.
- Discord discussions: [ISS-16](https://discord.com/channels/1538659898383011961/1557269277659111425),
  [ISS-17](https://discord.com/channels/1538659898383011961/1557274460023951442),
  [ISS-25](https://discord.com/channels/1538659898383011961/1558566415387328574),
  [ISS-26](https://discord.com/channels/1538659898383011961/1559991013852839938),
  [ISS-29](https://discord.com/channels/1538659898383011961/1560305653476499537),
  [ISS-30](https://discord.com/channels/1538659898383011961/1560322898894397565),
  [ISS-31](https://discord.com/channels/1538659898383011961/1560423303447846944),
  [ISS-32](https://discord.com/channels/1538659898383011961/1560423364643000521),
  [ISS-33](https://discord.com/channels/1538659898383011961/1560423431592353825),
  [ISS-34](https://discord.com/channels/1538659898383011961/1560423478975529034),
  [ISS-35](https://discord.com/channels/1538659898383011961/1560423511271411924),
  [ISS-36](https://discord.com/channels/1538659898383011961/1560423568788029481), and
  [ISS-37](https://discord.com/channels/1538659898383011961/1560423627223076987).
