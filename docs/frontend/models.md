# Models

## Registry, details and inline editing

`/models` presents open entries with serif names, quiet metadata and measured
parameter/context figures, without a registry-wide summary strip. Search matches name, description and family; admins
also have visibility filters and explicit bulk selection. `/models/[id]` uses
open fact ledgers for architecture, training, lineage, checkpoints and attention.

Administrators edit the name, description, family and size directly in the page.
Save changes persists the complete display draft; Cancel restores stored values
and failures retain the draft. Size is an explicit Fast, Balanced or Flagship
category, shown as quiet text. Unassigned models have no category; parameter
counts never assign it. Readers see text without editing, publication or deletion
controls. Rust's admin extractor remains the write-authorization boundary.
Drafts, requests and diagram state reset when session or model identity changes.

## Architecture and counts across training families

The LLM engine resolves `architecture`, `vocab_size` and `parameter_count` for
listings, details and checkpoints. Saved results override requested config;
legacy gaps fall back to metadata on disk and a bounded base/resume ancestry walk.
This supports older SFT/RL and resumed models without copying ancestor training
data into the current run. Missing facts stay unavailable, including when an
ancestor was deleted or a lineage cycles.

`hooks/use-models.ts::architectureOf` consumes the resolved shape, then falls
back to saved results and config for older servers. Recorded parameter counts
take precedence; only when absent does the shared training calculation estimate
from shape and measured vocabulary. A requested `max_vocab` is a learning ceiling,
not the vocabulary actually built. Nested legacy dataset summaries are supported.

`components/training/architecture-flow.tsx` is the single diagram implementation
used by both model details and the training composer. It accepts only shape and
training mode, preserves bounded geometry and responsive layouts, and exposes
Pause/Play while respecting reduced motion. SFT uses assistant-token targets;
RL labels rewards and its policy objective. The attention viewer also uses the
resolved layer/head counts, so it remains available for inherited architectures.

## Lineage: how a model was constructed

`components/models/model-lineage.tsx` on `/models/[id]`, beside the
architecture and training-data panels: the chain of models a model was built
from, and what each one actually trained on.

**Two kinds of edge, and never more than one on a given model.** `baseModelOf`
(a fine-tune's ancestry, unchanged from the section above) and
`resumedFromOf(raw)` — new here, reading `parameters.config.resume_from`,
present only on a pretrain that continued from a model or one of its
checkpoints. A model states at most one of the two, because they belong to
different `family` values: a resumed pretrain is still a pretrain, a fine-tune
is never a pretrain. That is why the lineage is a **list, not a tree** —
`hooks/use-model.ts::useModelLineage` walks it as one, never branching — and
why plain flex markup draws it rather than SVG: there is nothing to lay out
that a vertical stack of rows and a `↓` doesn't already say.

**A checkpoint is deliberately not a node in this chain.** `parentId`/
`checkpointIndex` (see `docs/decisions/checkpoints-are-models.md`) say "one of
this run's own intermediate saves" — the same recipe, mid-run — not "a
different recipe this model was built from," which is what this graph exists to
show. Folding checkpoints in would draw a run's ten saves as though they were
ten ancestors of equal standing with an actual base model or resume point.
`ModelCheckpoints` already owns that relationship, on the same page.

**Walked client-side, not through a new endpoint.** There is no route that
returns a whole ancestry in one call; `useModelLineage` fetches the given
model's own `baseModelId`/`resumedFromId` one `GET /models/{id}` hop at a time,
stopping at the first ancestor with neither. A chain here is rarely more than a
few hops — a training lineage isn't something people build ten deep — so a
handful of small sequential requests reads more simply than adding a
server-side lineage endpoint (`resolve_model_metadata` in
`llm-engine/api/model_metadata.py` resolves inherited facts without returning
the lineage itself) for traffic
this light. A `MAX_LINEAGE_HOPS` bound guards the walk against a cycle the
system does not otherwise allow.

**Dataset names are resolved, not shown as ids.** `datasetsOf(raw)` reads
`config.datasets` (or `config.curriculum`'s stages, flattened and deduped by
id, when the run was staged) — the same `{id, name, count}` shape on both
`TrainConfig` and `SftConfig` — and the panel resolves each id against the data
LLM engine's own dataset list, falling back to the name stored with the selection
at train time, then to the bare id: the same "named when we can" fallback
`baseModelName` already uses one section up.

## The attention viewer

`components/models/attention-viewer.tsx` on `/models/[id]`: pick a layer and a
head, give it a text, get that head's attention matrix back. `lib/attention.ts`
is the wire (base64 fp32 in, a `Float32Array` out), `hooks/use-attention.ts` the
state, `components/models/attention-canvas.tsx` the figure. The LLM engine side —
including why nothing records attention until someone asks — is
[`../llm-engine/README.md`](../llm-engine/README.md).

- **Canvas2d, and measured rather than assumed.** One backing pixel per cell,
  filled through a 256-entry ramp into one `ImageData`, one `putImageData`, and
  CSS scales the element up with `image-rendering: pixelated`. A cell is a pixel
  and not a shape, so a `fillRect` per cell would be 262,144 draw calls for
  something the compositor does as a blit. Measured in headless Chromium with
  **no GPU at all** (software raster, the pessimistic case): 512×512 — the
  largest matrix the LLM engine will return — **median 6.3 ms, p95 12.7 ms**;
  256×256 1.6 ms; 128×128 0.3 ms. That is a one-off cost per fetched matrix,
  behind a network round trip and a forward pass that each cost more, and
  hovering redraws nothing. Canvas2d is comfortably enough here and there is no
  case for a GPU path; the panel reports its own last draw time so the next
  person can check that on their own machine rather than take this on trust.
- **The crosshair is not drawn on the canvas.** Two absolutely-positioned rules
  at a percentage offset, so following the pointer costs a style recalculation
  instead of a redraw of the matrix.
- **Colour lives in `components/models/attention-ramp.ts`, not in
  `chart-theme.ts`.** A weight is a magnitude, so it takes that file's
  single-hue sequential rule — but a *cell* at t=0 has to vanish into the
  surface where a *line* at t=0 still has to be visible, which inverts the
  bottom of the ramp. One hue at varying alpha, so the panel's own surface
  carries the low end and the figure is theme-aware without sampling a token or
  keeping a second dark-mode ladder. The hue is chart-theme's per-head violet,
  because attention entropy in the diagnostics panel is per head too.
- **The ramp is gamma'd (0.4) and the readout is not.** Rows sum to 1 over up to
  512 keys, so a linear ramp draws a diffuse head as an empty square. The figure
  is a display transform; the hovered cell always states the exact weight, and
  the tokens on both axes, which is the whole point of labelling them.
