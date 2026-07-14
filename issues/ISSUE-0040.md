---
id: ISSUE-0040
title: "Epic: Redesign the /training page — an editorial-minimalist training desk (pretrain→SFT continuation, param presets)"
scope: cross
priority: P2
status: todo
rule: none
location: frontend/app/training/page.tsx frontend/components/training/ backend/src/routes/ backend/src/db/
blocked-by: [ISSUE-0041, ISSUE-0042, ISSUE-0043, ISSUE-0044, ISSUE-0045, ISSUE-0046, ISSUE-0047, ISSUE-0048, ISSUE-0049]
blocks: []
---

## Description

**Tracking/epic issue — little code lives here directly.** It is `blocked-by` the
child chain (0041–0049); once those are `done`, the Builder closes this one against
the "Definition of done" below. It exists to give the Builder the map before it
picks up any child. Follows the same shape as the SFT epic (ISSUE-0030).

### The problem (user brief)

The `/training` page works and shows the right information, but the **UX and visual
language are wrong**:

1. **Cards pop in and out.** `components/training/section.tsx` is a collapsible
   `Card` that auto-expands when its phase goes active and collapses when it
   finishes, and the page conditionally *mounts* the tokenize/train sections as the
   run progresses (`page.tsx:76`, `page.tsx:100`). The result is layout that grows,
   jumps, and reflows — the opposite of calm.
2. **You have to scroll to the relevant thing.** The live loss chart — the whole
   point while a run is going — sits at the bottom, below config and tokenization.
3. **Wasted screen.** The page is capped at `max-w-4xl` and stacks everything in one
   column. On a wide monitor most of the canvas is empty.
4. **Two features wearing one coat.** Pretraining and SFT are distinct activities
   crammed into one linear form + flow. After a pretrain finishes there is no path
   to *continue that model into SFT* — you start over and hand-pick the base model.
5. **No reusable parameters.** Every run is configured from scratch. There's no way
   to save a good config under a name, nor to pull the config out of a model you
   already trained and tweak it.

### The design direction (decided with the user)

**Hard Editorial Minimalism** — the page should read like a lab-notebook / article,
not a dashboard of boxes. Two decisions were locked in up front:

- **Layout: the "editorial spread"** (chosen over a scroll-spy longform and a
  control-room grid). A wide two-column layout that fills the screen:

  ```
  ┌────────────────────────────────────────────────────────────┐
  │  OPAL-1.1            ● training     [ Pretrain › SFT ]       │  masthead + phase stepper + primary action
  ├──────────────────┬─────────────────────────────────────────┤
  │ PARAMETERS       │  loss 2.41   epoch 3/5   eta 4m   72%     │  stat run — hairline rules, no boxes
  │ preset ▾  save   │  ─────────────────────────────────────── │
  │ ─────────────    │                                          │
  │ name   opal-1.1  │        [ loss chart — the hero figure ]  │  figure + caption
  │ dims       64    │                                          │
  │ ctx        64    │  ─────────────────────────────────────── │
  │ heads       4    │  [ batch spread ]      [ vocab growth ]  │  secondary figures
  │ layers      2    │                                          │
  │ epochs      5    │  ─────────────────────────────────────── │
  │ lr       0.01    │  ▸ Appendix — log · checkpoints          │  persistent collapsible detail
  │ [ Start ]        │                                          │
  └──────────────────┴─────────────────────────────────────────┘
  ```

  - **Left rail** — the run's *identity + parameters*, styled as a hairline
    definition list (`dt`/`dd` rows separated by 1px rules), not a boxed form. It is
    the article's byline/sidebar. Holds the preset controls.
  - **Right column** — live telemetry, top-anchored so the *most relevant thing is
    at a glance without scrolling*: a horizontal **stat run** (key numbers separated
    by hairline rules, no card chrome), then the loss chart as the **hero figure**
    with a caption, then smaller secondary figures, then a collapsed **appendix**.
  - **Phase stepper** in the masthead makes pretrain and SFT *lateral* rather than
    stacked — the two features finally look like two things.

- **No pop-in / no page growth.** Every region owns a fixed slot in the layout and
  is *always rendered*. Before data exists it shows a quiet placeholder (hairline
  frame + muted caption), not nothing. Phases that aren't active are de-emphasized
  (dimmed / collapsed appendix), never unmounted. This is the single rule every
  frontend child must honour.

- **Presets: full backend persistence** (chosen over localStorage-only). Named
  parameter sets live in a new `training_presets` table, per-user, reachable over
  REST — durable across devices. Plus a **"Load from model ▾"** affordance that
  reads an existing model's stored `parameters` (models already persist the full
  training config as JSON — `db/model.rs:6-10`) into the form so you can clone-and-
  tweak a past run's config.

### The pretrain→SFT continuation

Today an SFT run needs you to manually pick a `base_model_id` (`config-form.tsx`).
The redesign makes it a *continuation*: when a pretrain finishes, the phase stepper
advances and a **"Continue to SFT"** action seeds a fresh SFT config with the
just-trained model as its base — no hunting in a dropdown. This needs the training
`done`/`snapshot` frame to carry the trained model's **id + name** (today it only
carries `model_dir` — `training-socket.ts:110`), which is child 0043.

### The child issues

| # | Issue | Scope | Depends on |
| --- | --- | --- | --- |
| 0041 | `training_presets` table + `db/preset.rs` queries | backend | — |
| 0042 | Preset REST routes (list/create/update/delete), per-user | backend | 0041 |
| 0043 | Training `done`/`snapshot` frame carries trained model id + name | backend | — |
| 0044 | Editorial layout shell: masthead + phase stepper + two-column spread; retire the collapsible `Section` cards; reserve every region's slot | frontend | — |
| 0045 | Params rail — config as an inline-editable hairline definition list (refactor `ConfigForm`) | frontend | 0044 |
| 0046 | Telemetry restyle — hairline "stat run" (retire boxed `Stat`) + charts-as-figures with a shared editorial chart theme | frontend | 0044 |
| 0047 | Appendix drawer — log + checkpoints in a persistent collapsible detail region | frontend | 0044 |
| 0048 | Preset client + hook + UI (save/load/delete) and "Load from model" config mapping | frontend | 0042, 0045 |
| 0049 | Pretrain→SFT continuation flow — stepper transition + "Continue to SFT" seeding the base from the finished pretrain | frontend | 0043, 0044 |

The three backend children (0041–0043) are independent of the frontend and of each
other and can land first. The frontend structural child (0044) unblocks the rest of
the UI work; 0045–0047 are parallel once 0044 lands; 0048/0049 are the integration
children that consume the backend work.

### The one idea every frontend child shares

**Reserve the slot; fill it in; never pop.** A region is present in the grid from
first paint with a placeholder, and transitions by *populating*, never by mounting/
unmounting or by collapsing in a way that reflows its neighbours. Any child that
adds a region owns its empty state.

### Non-goals (keep the epic bounded)

- No change to the training *backend algorithm*, the socket wire protocol's progress
  frames, or the job/reattach model — only the `done`/`snapshot` payload gains two
  fields (0043).
- No new charting library; keep Recharts, restyle it (0046).
- Presets store a `TrainConfig`-shaped blob; no server-side validation beyond
  name/ownership (the existing `/training` POST already validates a config on use).

### Definition of done (what the Builder verifies to close THIS issue)

1. All children 0041–0049 are `done`.
2. On a wide viewport the `/training` page shows the editorial spread with **no
   card pop-in and no page-height jump** across a full pretrain run (verified by
   driving a tiny run and watching the layout stay stable — regions fill in place).
2b. The live loss chart is visible **above the fold** while training, without
   scrolling, on a 1440px-wide window.
3. A named preset can be **saved, listed, loaded, and deleted**, surviving a page
   reload (backend persistence); and "Load from model" populates the form from an
   existing model's stored config.
4. Finishing a pretrain surfaces **"Continue to SFT"**, which advances the stepper
   and starts an SFT run whose base is the just-trained model — no manual base pick.
5. `frontend` `bun run typecheck` + `bun run lint` clean; `backend` `cargo build` +
   `cargo clippy` clean.

## Comments
### Auditor · 2026-07-12
Filed as the design/tracking parent for the `/training` UX redesign the user
scoped out. Structured as an epic with a `blocked-by` chain rather than one giant
issue, matching this repo's preference for small, single-location, independently-
verifiable units (cf. ISSUE-0030). Two product decisions were resolved with the
user before filing: the **editorial-spread** layout metaphor and **full backend
preset persistence** — both are baked into the children so the Builder needn't
re-litigate them. Priority P2: net-new UX + feature work, not a latent bug.
</content>
</invoke>
