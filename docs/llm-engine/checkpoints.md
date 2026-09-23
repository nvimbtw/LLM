# Checkpoints — the layering that keeps them cheap, and resuming from one

What a checkpoint *is* and how often a run takes one are in
[`README.md`](README.md#checkpoints); why it is a whole model is
[`../decisions/checkpoints-are-models.md`](../decisions/checkpoints-are-models.md).
This file is the mechanism: which layer owns which half of it, and what
`resume_from` does with what it finds.

## Who owns which half

**The layering did not move for this.** `llm/` has still never heard of the
database:

- `llm/train/loop.py` decides **when** — `_Checkpoints` owns the interval, the
  epoch end, the run's last step, and the deduplication between them. The loss
  it reports is passed as a callable, not a number, because reading one drains
  a device-side buffer: a run taking no checkpoints must not pay a host sync for
  the ones it is not taking.
- `llm/orchestrate/train.py::CheckpointWriter` decides **what a checkpoint
  directory is**, and `orchestrate/sft.py` uses the same one: the two
  orchestrators differ only in what they know to record about themselves, which
  is what they hand its constructor.
- A `Checkpoint` frame rides the existing `ProgressCallback` — the seam a run
  already has for things it must say while still running — and
  `api/handlers.py::record_checkpoint` is the only place that becomes a row.
  The frame carries the **shape** as well, measured off the model whose weights
  were just written rather than read off the run's config: a resumed run states
  no architecture at all, so a row sourced from the config advertised pydantic's
  defaults as the checkpoint's shape and served a `context` nobody had chosen.
  It lands on the row as `result.architecture` and as the `context` column.
  It writes **immediately, bypassing `PROGRESS_WRITE_SECS`**: that throttle
  exists so a progress bar does not cost a row per step, and a checkpoint frame
  it swallowed would be weights on disk that nothing can name. It also never
  raises — losing a still-training run over an unregisterable checkpoint would
  cost the weights being computed.

## Resuming

`TrainConfig.resume_from` names a model id — a finished model or one of its
checkpoints, and the code does not branch on which: they are the same directory
shape, the difference is only whether a `train_state.pt` is in it, and a plain
weights file read as a state is a continuation with cold moments that says so.
Fine-tuning has its own field for the same idea (`SftConfig.base_model`); only a
pretrain resumes.

Two honest behaviours, and **the run records which it took** (`result.resume`),
because after the fact the two are indistinguishable from the weights and
produce very different curves:

- **Resumed** — the state was taken over this corpus (`corpus_key`) on this
  schedule (`total_steps`). Optimizer, generators and data order are restored
  and the run re-enters where it stopped, arriving where the uninterrupted run
  would have. The fast-forward is arithmetic, never a walk: whole epochs behind
  the saved step are subtracted rather than iterated, and `EpochStream` is
  offset-addressable so the skipped head of the re-entered epoch is never
  gathered.
- **Continued** — anything else. Weights and moments carry over; the step is
  reset and the RNG dropped, because a step index into a different schedule is
  not a position and a generator restored mid-way through data this run will
  never see is not an order.

**Architecture comes from the state, never from the request.** Only fields the
request actually set are compared (pydantic's `model_fields_set` — a resume
naming nothing but `resume_from` is asking for the checkpoint's shape, not for
the defaults filled in behind it), and a disagreement, vocabulary size included,
is refused rather than silently overridden: the weights would win regardless,
and the run would report an architecture it did not train.

Two things follow, and both were missing:

- **The shape is applied to the config, not only to the model**
  (`train.py::resumed_config`, before phase 1 reads a number off it).
  `ctx_window` is what the loop cuts windows at and what phase 1 measures the
  corpus against, and every record a run leaves — `meta.json`, each
  checkpoint's config, the model row's `context` — is that config dumped. A run
  that resolved the shape at `Transformer(...)` alone trained a 1024-token
  checkpoint on 256-token windows and wrote the request's numbers down as
  though they were what it had done. What it *did* train at is recorded under
  `result.architecture`, the same key SFT has always used.
- **Statedness has to survive the queue.** A run is queued now and started
  later, so `POST /train` stores `model_dump(exclude_unset=True)` — the request,
  not the request plus every default. Dumping the defaults in alongside made
  every field "set" by the time `handlers.train` rebuilt the config, and the
  comparison above then refused **every** resume from a checkpoint that was not
  the composer's default shape, for restating an architecture nobody had typed.
  The stored config being the request is also what lets the job recap say "the
  checkpoint's" instead of reporting a default as a decision.

**The tokenizer follows the same rule** (`train.py::resumed_tokenizer`), and it
has to: a corpus packed under a different vocabulary than the checkpoint's would
either index its embedding table out of bounds or, if the two vocabularies
happen to be the same size, mean different tokens by the same ids without
either failing at all. Before this, a resumed run with `tokenizer_id` left
unset re-derived a *fresh* vocabulary from its own dataset selection the way a
from-scratch pretrain does — plausible right up until it packed a corpus the
checkpoint was never trained to read. `resumed_tokenizer` recovers the
checkpoint's own id from `state.extra` (set by `CheckpointWriter` on every
checkpoint this pipeline writes) and falls back to the model's own `meta.json`
for a *finished* model, whose `train_state.pt` — a plain weights file — carries
no `extra` at all; that fallback is the same provenance lookup
`orchestrate/sft.py` uses to recover a base model's tokenizer
(`train.run_record`, née `sft._base_run`, generalized so both orchestrators
share it). A request that states a *disagreeing* `tokenizer_id` on a resume is
refused exactly like a disagreeing architecture field; one that states none
gets the checkpoint's silently, which is the point.

`max_vocab` is the loose end of that rule: `tokenizer_for` returns the
checkpoint's vocabulary before ever reading it, so a resumed run's stated
ceiling is neither honoured nor refused — it is ignored, and then stored on the
run as though it had applied. Nothing here needs it to fail; the composer does
not send it (`RESUMED_TOKENIZER_FIELDS` in the frontend), which is why this is
recorded rather than guarded against a second time.

The resume position is a **batch** ordinal. On a masked corpus the loop drops
windows with nothing trainable in them, so steps and batches differ by however
many were dropped and such a resume would re-enter within a few windows of where
it stopped rather than exactly on it. Nothing can reach that today — SFT does
not resume — and it is the same approximation `total_steps` already makes.

## Where the state lives

`storage.train_state_path` puts `train_state.pt` beside the weights, inside the
model directory. It replaced `storage.checkpoints_dir` — a `checkpoints/`
subtree under the *run's own* model holding one `epochNNN.pt` of weights alone,
which nothing could resume from, chat with or benchmark, and which no row ever
named. `llm/storage.py`'s docstring is the authoritative map of the tree;
nothing else states a path.

The file is a superset of a weights file: same `version` and `arch` and
`state_dict`, plus the optimizer, the position and the generators. That overlap
is deliberate — `load` reads a state file and `load_state` reads a plain weights
file — and it is why `CHECKPOINT_VERSION` did **not** move for the added keys:
they change nothing about what an existing file means, and a version bump would
have orphaned every model already on disk. `torch.load` stays
`weights_only=True`; numpy's generator state is converted to a tensor rather
than widening the unpickler for the `ndarray` its own `get_state` hands back.
