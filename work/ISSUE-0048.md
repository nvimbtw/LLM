---
id: ISSUE-0048
type: issue
title: "Preset client + hook + UI — save/load/delete named param sets and Load-from-model"
scope: frontend
priority: P2
status: verification
rule: none
location: frontend/lib/training-presets.ts frontend/hooks/use-training-presets.ts frontend/components/training/preset-controls.tsx
blocked-by: [ISSUE-0042, ISSUE-0045]
blocks: []
---

## Description

Part of the `/training` redesign epic (EPIC-0040). Wires the params rail up to the
preset backend (0041/0042): save the current config under a name, list/load/delete
saved presets, and **"Load from model"** — pull the stored config out of an
already-trained model and drop it into the form to tweak.

### The three layers (frontend conventions)

**`lib/training-presets.ts`** — API client, mirroring `lib/training-socket.ts`'s
style (uses `API_URL` + `authHeaders()` from `lib/api.ts`; no inline `fetch` in
components/hooks):

```ts
export interface Preset { id_preset: string; name: string; config: TrainConfig; updated_at: string }
export async function listPresets(): Promise<Preset[]>              // GET  /training/presets
export async function createPreset(name, config): Promise<Preset>   // POST /training/presets
export async function updatePreset(id, patch): Promise<Preset>      // PUT  /training/presets/{id}
export async function deletePreset(id): Promise<void>               // DELETE /training/presets/{id}
```

**`hooks/use-training-presets.ts`** — one primary hook owning preset list state +
CRUD callbacks (load on mount via `useEffect`, optimistic-or-refetch on mutate).
Surfaces `{ presets, isLoading, save, remove, ... }`. No domain state in components.

**`components/training/preset-controls.tsx`** — the UI, dropped into the mount point
the rail left in child 0045:

- A **preset picker** (`▾`) listing saved presets; selecting one loads its `config`
  into the form (calls back up to the rail's config setter).
- A **"Load from model ▾"** option/section: lists the user's models (reuse
  `useModels`) and, on pick, fetches that model's `parameters` (via the existing
  `GET /models/{id}` — `routes/model.rs`) and maps it into a `TrainConfig` (see
  mapping below). This is *clone-and-tweak*: it fills the form but does not select
  the model as an SFT base (that's the continuation flow, 0049).
- A **Save** control: names the current config and `createPreset`s it; offers
  "update existing" when the name matches a loaded preset (the backend `409`/PUT
  from 0042). A small **delete** affordance per preset.

### The model→config mapping

`models.parameters` stores "the full training config + dataset stats as JSON"
(`db/model.rs:6-10`) — i.e. a superset of `TrainConfig`. Write one small pure mapper
(in `lib/training-presets.ts`) that picks the `TrainConfig` fields out of that blob
and falls back to `DEFAULT_CONFIG` values for anything missing, so a model trained
before a field existed still loads cleanly. Do **not** trust the blob's shape
blindly — coerce/validate the numeric fields.

### Rules + integration

- Loading a preset or a model config sets the rail's `cfg` state; the existing
  `validate()` (0045) then runs as normal, so an out-of-range loaded value surfaces
  its error rather than silently starting a bad run.
- Presets/model-config loads are **disabled while a run is active** (mirror the
  rail's `disabled`), since you can't reconfigure a run in flight.
- Keep the picker/save UI quiet and editorial (hairline, no `Card`) to match the
  rail — it lives *in* the rail, not as a floating panel.
- Handle the empty state (no saved presets yet) with a muted caption, not an absent
  control.

### Definition of done

- Save a named preset → it appears in the picker and **survives a page reload**
  (backend persistence, 0042).
- Load a preset → the form populates; edit + start works.
- Delete a preset → it's gone after reload.
- "Load from model" → picking a trained model fills the form from its stored config.
- Duplicate-name save is handled gracefully (offer update, or surface the `409`).
  `bun run typecheck` + `bun run lint` clean.

## Comments
### Auditor · 2026-07-12
Child of EPIC-0040, blocked by the preset routes (0042) and the rail's mount point
(0045). This is the preset half of the epic's "reusable parameters" goal; the
model→SFT-base continuation is the separate flow in 0049.

### Builder · 2026-07-12
Added `lib/training-presets.ts` (typed `list/create/update/delete` + `toTrainConfig` coercing mapper + `loadModelConfig` reading `/model/{id}`.`parameters.config`), `hooks/use-training-presets.ts` (list state + `save`/`remove`, save-over-existing via name match → PUT else POST), and `preset-controls.tsx` (picker with per-item delete, "Load from model" submenu via `useModels`, inline name-and-confirm save). Disabled while a run is active; empty states handled. **Backend CRUD already verified end-to-end (0042)**; used base-ui `render={<Button/>}` on the trigger (see Verifier note below).
