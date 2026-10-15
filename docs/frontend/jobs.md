# Jobs and administration

## The admin panel

Everything behind the admin flag is reached through **one row** in the sidebar
footer, above the profile/settings entry — not a labelled nav group. The row is a mode of the app
you have access to, which is what the footer holds; the content above it is the
subject you are reading.

- `/admin` is the panel, in three parts: `QueueView` as the hero,
  `AdminDirectory` under it, and `HistoryView` last. **The directory sits
  between the two halves of the queue on purpose** — history is unbounded, so
  anything below it is reached only by scrolling past every run ever made.
  **Add an admin destination to the directory, not to the sidebar** — it keeps
  admin destinations discoverable without crowding the main navigation.
  The two bands separate data and model work (`WORKBENCH`) from interface
  fixtures and documentation (`INTERFACE`).
  The panel uses open sections with shared serif headings and inline counts,
  matching settings rather than repeating filled cards. The running job is the
  visual focus: its name, per-kind throbber, progress and elapsed time sit above
  a thin progress rule. Idle and paused states are stated beside the Queue
  heading; an empty queue has no repeated status sentence or instructional
  caption. Pausing holds the next claim while the running job remains visible;
  follow-ups stay separate from numbered queue positions. Directory links use
  icon-and-title rows with hover and keyboard-focus treatment, with a rule only
  between groups. History keeps its date groups, fields, actions and pagination
  in a quiet ledger; on narrow screens row metadata wraps beneath the job name
  and menus remain visible without hovering.
- The subpages carry the same open style: shared `PageHeader` titles and a
  bottom hairline, flush rows and line-style tab bars. A ledger that opens with
  its own rule omits the header rule. The editable Train heading and gym-run
  telemetry header retain their compact composition. Forms keep labels,
  validation, explanations needed for a decision and their existing controls.
  Job details use `PageHeader` with kind, status and elapsed time as provenance,
  lifecycle actions beside the title, and the resolved `useParentTrail` back link.
  Chart tooltips, dialogs and menus retain their overlay surfaces. Benchmark scores sit in an open
  ledger, and row actions are visible on touch screens and keyboard focus.
  Docs remains a reading destination with its existing article layout.
- **`/data` is a `WORKBENCH` destination**, not a second nav row. It reads the
  data engine through `lib/data.ts`; the cleaning workbench previews a pipeline
  over the first few documents on a debounce, which is why the sample is always
  taken by ordinal — a moving sample would compare two different texts as the
  user tunes a parameter. The pipeline is a **horizontal rail** above the
  before/after and the open step's parameters sit under it: the steps are an
  order, and a column that wraps reads as a second pipeline. The before/after
  is a split diff (`lib/diff.ts` — an LCS over lines, no dependency), with
  untouched stretches folded away so what a step did is not buried in what it
  left alone.
- **`NavJobs` stays in the sidebar content** and is not navigation: it is the
  ambient "the GPU is busy" signal, a window onto the panel rather than a way
  in. `NavAdmin`'s `QueueTell` is the same fact when the sidebar is collapsed to
  icons and that group is hidden.
- Admin-ness is a client fact (`useAuth().isAdmin`, read from the stored access
  level), so these are hidden rather than guarded. **The endpoints are the
  boundary** — the LLM engine's queue routes are admin-only, and `JobsProvider` is
  gated on the same flag so a normal session does not spend its life collecting
  403s.

## The jobs queue

Everything that uses the GPU — training and benchmarks — is a run in the
LLM engine's queue. See `lib/jobs.ts` (types + REST) and `lib/jobs-socket.ts` (one
socket per run, relaying the LLM engine's own frames).

A `tokenize` run is the exception that proves the line: it never touches the
device. It is queued anyway because it is minutes of CPU that would fight a
running train for it, and because the queue already has the frames, the cancel
and the job page it wants. `/tokenizers` opens on a vocabulary catalogue with
All, Built and From runs filters. Named builds lead the combined list; automatic
builds remain inspectable and deletable, with their reuse semantics stated below
it. `TokenizerCatalogue` presents a board-width ledger of vocabulary names,
provenance and compact figures. Names open the reading-width detail editor;
**Build vocabulary** opens `/tokenizers/new-tokenizer`, where page-native fields
queue the build and hand off to its durable execution page. See
[tokenizers](tokenizers.md) for metadata persistence and session boundaries.

- **The board lives at `/admin`**, as the hero of the panel — `QueueView` and
  `HistoryView` are mounted there, and `/jobs` is a redirect kept only so
  bookmarked URLs work. Job links use their domain URLs: `/train/{id}`,
  `/bench/{id}`, `/gyms/{id}` and `/tokenizers/builds/{id}`. Legacy detail and
  composer URLs redirect while preserving queries; see [routing](routing.md).
  The two are separate components only so the panel can put its directory between them;
  they read from the same `JobsProvider` and are one ledger.

### Composing a training run

The same fields are rendered two ways, and neither may drift from the other:

- `components/training/train-board.tsx` at `/train/new-train` — one bounded
  form whose model name is the editable page heading, like the project editor.
  Presets sit beside the heading; quiet mode tabs precede compact identity rows,
  then data, architecture, optimization and follow-ups. Identity values and their
  popup menus align to the right without explanatory subtitles. Text actions
  keep a transparent surface during hover. Selectors and numeric
  values use borderless inline controls with visible keyboard focus. Labels sit
  beside their controls on wide screens; numeric settings use two columns.
  Dataset limits, curriculum names/epochs and follow-up fields use the same
  inline treatment; menus retain their normal accessible overlay surfaces.
  Switching between prose training and fine-tuning clears dataset/curriculum
  selections from the other dataset kind, while preserving explicitly chosen
  learning rates. Both editors use `run-config.ts::configForMode` for this policy.
  Training data presents datasets, then settings, then a composition donut with
  document counts and shares. Shuffle, Curriculum and Strip thinking use fully
  clickable checkbox blocks, with a theme-accent tint only when selected and
  visible keyboard focus. `lib/training-data.ts` resolves limits against the
  catalogue and counts repeated curriculum prefixes once at their largest limit;
  epochs do not weight the shares. Unknown sizes or invalid limits leave the
  composition unavailable instead of silently drawing a partial or zero share.
  Below fresh-run architecture fields, `architecture-flow.tsx` explicitly selects
  the **training** view: a forward pass, target supervision, loss, gradients,
  optimizer and weight updates before the next batch. Shape values update from
  the draft. The shared [architecture diagram](architecture.md) uses a bounded
  transformer stack, aligned responsive geometry and operation-specific motion.
  Pause/Play retains the current frame; reduced motion shows a static diagram.
  Fine-tuning and resumed pretraining display the selected model's actual
  architecture and vocabulary as read-only facts through
  `inherited-architecture.tsx`, with the same parameter subtitle and training
  diagram. Reads reset on session/model changes; missing metadata stays
  unavailable instead of displaying draft defaults. The fine-tune diagram
  identifies assistant-token targets and masked cross entropy.
  A quiet parameter count sits directly below the Architecture heading. The
  estimate uses the selected tokenizer's actual vocabulary size,
  or the learning ceiling before a vocabulary exists. It counts tied embeddings
  once, attention/feed-forward weights, biases and layer norms; RoPE has no learned
  positional table. Invalid draft shapes or unavailable vocabulary sizes show no
  estimate. `lib/training-architecture.ts` owns this preview calculation.
  The **build sheet** follows the decisions as a compact Review ledger with the
  derived specification, validation and start action. There is no fixed recap
  column competing with the fields.
- `components/training/params-rail.tsx` inside a follow-up step, where the same
  fields have a rail's width and no submit of their own.

Both take their field list, their derived figures and their validation from
`components/training/run-config.ts`, which holds no JSX. A rule about what a
run may be belongs there, never in a layout.

**A run that continues from a checkpoint states no architecture at all.** The
LLM engine takes the shape from the state it resumes and *refuses* a request that
restates it differently — it cannot let the request win, because the weights
would win anyway and the run would report an architecture it did not train. So
`RESUMED_ARCH_FIELDS` (`lib/training.ts`) is dropped from the payload in
`lib/jobs.ts`, exactly as a fine-tune's `base_model_id` is translated there, and
the same list is what `fieldsFor` hides and `validate` skips. Hiding without
dropping is the failure mode to watch for: pydantic reads a field left at the
form's default as one the request *stated*, so a control nobody can see is still
enough to have the run rejected.

Dropping a field only works while the far side leaves the gap open. It did not:
`POST /train` stored the config as a full dump, defaults and all, so every field
was "set" again by the time the pipeline read it and **every** resume from a
checkpoint that was not the composer's default shape was refused, whatever this
module sent — see [`../llm-engine/checkpoints.md`](../llm-engine/checkpoints.md). What
that means here is that a resumed run's stored config carries no architecture
and no vocabulary, so `components/jobs/config-recap.tsx` names whose shape it is
("The checkpoint's") rather than printing five numbers nobody chose, and the
resolved shape is read back off the model the run produces
(`hooks/use-models.ts::architectureOf`, which falls through to
`result.architecture`).

**The tokenizer is inherited the same way, on its own list.** `resumed_tokenizer`
in the LLM engine refuses a resumed run's *stated* `tokenizer_id` exactly as
`resumed_architecture` refuses a disagreeing shape, so the composer offers no
Tokenizer selector once `resume_from` is set (`run-config.ts::hidesTokenizer`,
read by both `train-board.tsx` and `params-rail.tsx`), and `RESUMED_TOKENIZER_FIELDS`
(`lib/training.ts`) is dropped from the payload alongside `RESUMED_ARCH_FIELDS`
in `lib/jobs.ts`. It is a list of its own rather than folded into
`RESUMED_ARCH_FIELDS` because neither field on it is an architecture field and
`tokenizer_id` is not rendered through `ARCH_FIELDS` — the two are read together
everywhere a resume's payload is stripped. The main editor's inherited Architecture
section shows the checkpoint's resolved shape and vocabulary without making them
editable; a follow-up's rail states the same inheritance in prose.

**`max_vocab` is on that list too, and it is the one to keep an eye on.** It is
neither refused nor honoured: the LLM engine's `tokenizer_for` returns the
checkpoint's vocabulary and returns *before* `max_vocab` is read, so a stated
ceiling is silently ignored — which is why it survived the pass that hid the
architecture knobs. A field that is quietly dropped on the LLM engine's side is
worse than one that is refused, not better: nothing fails, and the run is stored
and recapped carrying a number it never used. It renders through `ARCH_FIELDS`,
so `fieldsFor` had been hiding it only when a *built* vocabulary was named —
a condition a resume never meets, because `hidesTokenizer` means `tokenizer_id`
is never set. `validate` skips it on a resume for the same reason it skips the
shape, and `jobs/config-recap.tsx` prints "Tokenizer — The checkpoint's" instead
of a max-vocab row rather than reporting the LLM engine's default back as a
decision.

### Reading a run back — the diagnostics panel

`components/training/diagnostics-panel.tsx` renders `DiagPoint[]`
(`lib/training.ts`), which is the LLM engine's `Diagnostics` dataclass **verbatim**
— nothing between the training loop and the chart renames a field, so a rename
on either side arrives as `undefined`. The LLM engine's side is documented in
[`../llm-engine/telemetry.md`](../llm-engine/telemetry.md).

- **An optional field means "not measured", and must never render as 0.** The
  LLM engine drops what it did not measure instead of sending a zero, because a zero
  reads as a real measurement. Every fine-tuning field on `DiagPoint` is
  therefore optional; `opt()` in the panel maps `undefined` to `null` so recharts
  drops the point rather than plotting it at the bottom of the axis.
- **The fine-tuning figures are their own presence check.** A frame carries no
  mode flag: an SFT run sends `loss_masked` and a pretrain run sends nothing, so
  `diag.some(d => d.loss_masked !== undefined)` is what gates them. A pretrain
  run must show no empty SFT charts at all.
- **Every figure is `Figure` + recharts + `chart-theme.ts`.** Colour is chosen
  by what the series *is*: per-layer, per-head and by-position are magnitudes by
  index and take a single-hue sequential ramp; the three fine-tuning losses are
  categories and take distinct hues. Do not add a palette to a component.
- **Every array derived from `diag` is built in a `useMemo` keyed on `diag`,
  with no exception for a one-off passed straight into JSX.** The embedding
  biplot and globe key their own expensive work — `alignFrames`'s sign-matching
  across every stored report, `EmbeddingGlobe`'s icosphere/convex-hull rebuild —
  off the *reference* of the array they're handed, not its contents. A `diag.map(...)`
  written inline in the render body hands them a new array on every render of
  the panel regardless of whether `diag` itself changed, which silently defeats
  that memoization: the full rebuild reruns on every render rather than once per
  new report. A live run reports every few seconds and hides this; a finished
  run — reloaded with its whole history already in state — does not, and a page
  stuck re-running it on every render reads as frozen.

- **The board polls.** It moves when a run is queued, starts or ends — rare
  enough that a 3s refetch of `GET /runs` says the same thing a nudge would,
  without a second live connection to keep alive across sleeps and restarts.
- `hooks/use-jobs.ts` owns the board; `hooks/use-job.ts` owns one job — its
  durable row plus whatever it is broadcasting. **Panels render from the row
  first and treat live state as an enrichment**, because a queued or finished
  job has no frames at all.
- **The appendix's checkpoint list is built from `checkpoint` frames**, not from
  `TrainStats.checkpoint_path`. A checkpoint interval is a multiple of the report
  interval, so the epoch-boundary save is usually deduplicated against the
  interval one and its frame carries no path — reading the path listed almost
  nothing, and nothing at all on a single-epoch run. The frame is the LLM engine's
  `Checkpoint` dataclass verbatim (`CheckpointFrame` in `lib/training.ts`) and
  announces every save as its weights land, so the list is complete and each row
  can say where in the run it was taken.
- Composers live at `/train/new-train`, `/bench/new-bench` and `/gyms/new-gym`.
  Submitting queues a run and hands off to its domain execution page, where
  telemetry lives. `?from=<job id>`
  seeds a composer from a previous run's stored config.
- **A chat turn is not a run.** It is answered directly, on the socket that
  asked for it — an interactive turn behind a three-hour training run is a chat
  nobody can use.
- **Queued is a state with a body, not an empty version of the finished one.**
  Every kind renders `JobQueuedPanel` while it waits: the config recap, its
  place in the line, and what the run will do once it starts.
- `ConfigRecap` is shared and switches on `job.kind`. A new kind adds its rows
  there rather than growing a recap of its own.
- A **chain** is a flat list of steps, each running against the model the one
  before it produced. A follow-up waits rather than queueing: its model does not
  exist yet, so a queue position would be a lie about when it runs.
- `/bench` is history and analysis; starting runs uses `/bench/new-bench`.
- The throbber's per-kind modes (`training` / `benchmarking` / `generating`) are
  how a running job says what it is. Give **one** instance a changing `mode` so
  it morphs between them as the queue advances rather than cutting. `NowBand` is
  that instance.

### What a training run's phase 1 shows

Two panels, answering two different questions.
`components/training/build-progress.tsx` is **what the build is doing right
now** — which step, how far into it, how fast, and which source that leaves it
on — and `components/training/build-metrics.tsx` is what the build **produced**:
characters per token by source, how much of the vocabulary the corpus never uses,
the share of training windows that span a document boundary and a ribbon of what
those windows actually look like, document lengths against the window, and the
head of the merge list in the order BPE learned it.

Three things about it are load-bearing:

- **It outlives phase 1.** The build panel is replaced by the loss overlay the
  moment training starts; the metrics stay, because they are facts about the
  corpus the run is training on, and a finished run's page is where they are
  read.
- **An absent field is "—", never 0.** The LLM engine omits what a phase did not
  measure (see [`../llm-engine/README.md`](../llm-engine/README.md)), and a corpus packed
  before a metric existed reports none of it. Drawing that as a zero would be a
  number that reads as a real count of nothing.
- **`ctx_window` comes from the run's config, not from the frame.** The window is
  a property of the run, and one packed corpus can be trained at two of them —
  which is also why the LLM engine measures the length histogram at report time
  rather than baking it into the build.

##### While the build is running

The step is read off the frame (`TokenizeStats.step`), never inferred from which
counters have arrived. The panel used to infer it, which worked while every step
reported something and read the longest step of the run — packing, which reported
nothing — as a tokenizer that had already finished. The LLM engine's half of that,
including why a vocabulary build no longer closes the phase, is in
[`../llm-engine/telemetry.md`](../llm-engine/telemetry.md).

- **The elapsed clock is the page's, anchored to the LLM engine's.** `BuildClock` is
  remounted by its `key` whenever a frame lands and counts up from the reading
  that frame carried, so a page attached halfway through reads the build's age
  rather than its own and the two cannot drift. It exists because the frame
  interval widens as a build runs and one step — learning the merges — cannot
  report at all: for minutes at a time it is the only thing on the page still
  moving, and a clock that has stopped is exactly what a hung build looks like.
  A finished build's stops where it stopped, because one still running would be
  claiming something still is.
- **The throughput curve is rebuilt from the frames, not sent.** The LLM engine
  reports `tokens_per_sec` and not a series, so `lib/job-telemetry.ts` accumulates one — and
  a page attaching to a build already in flight still gets the whole curve,
  because a run's stream replays from the start. Only the frames that *carry* a
  rate become points: a step boundary's frame has counts and no measurement of
  speed, and drawing that as zero would put a stall into every one of them.
- **A bar is drawn only for the step in flight.** A full bar on a finished step
  says nothing and an empty one on a step that has not started says something
  untrue.
- **The source strip is a window, and says so.** The LLM engine centres at most
  `BUILD_SOURCES` rows on the source being read and sends `sources_total`; the
  caption carries both, so a curriculum longer than the window is never mistaken
  for one that fits. The stage label appears only on a run with more than one
  stage, where it is not the same word on every row.

##### The span ribbon

`components/training/span-ribbon.tsx` is the picture behind the straddle stat,
rendered directly under it: a sample of the run's actual training windows, one
bar each, split into the documents inside it and coloured by dataset. The number
cannot tell a corpus of books (a window sitting mid-chapter) from a corpus of
tweets (forty unrelated things per window), and those want opposite decisions.
The LLM engine's half — the sampling strategy, the frame bound, the cost — is in
[`../llm-engine/telemetry.md`](../llm-engine/telemetry.md).

- **SVG, not canvas2d, and the element count is not why.** The LLM engine caps the
  frame at `RIBBON_WINDOWS × RIBBON_SEGMENTS` rects, which either would draw
  without noticing. What decides it is that the fills are the theme's own values
  applied as attributes (canvas would need `getComputedStyle` and a redraw on
  every theme switch), a per-window tooltip is a `<title>` child rather than a
  hit-test, and the file needs no `'use client'`, no ref and no effect — a canvas
  is a client component by construction (R14). Lifting the cap is the signal to
  revisit this, and nothing else is.
- **A bar's own edges are not document boundaries.** The hairlines are drawn only
  where a document actually ends inside the window (`closes`), never at the
  window's edge. Drawing the two alike is what turns one book into nine unrelated
  documents, which is precisely the reading the figure exists to correct.
- **The elided tail is drawn thin and muted, never at bar height.** A window past
  the LLM engine's per-window cap reports `overflow` and stops short; filling that
  space at full height would claim one long document where there are dozens of
  short ones.
- **A dataset is a category, so it takes a hue and not a ramp** — the same wheel
  `lossStageColor` uses (`datasetColor` in `chart-theme.ts`), because a dataset
  and the curriculum stage it was packed into are one subject seen twice. As
  everywhere else here: do not add a palette to a component.
