---
id: ISSUE-0107
type: issue
title: "Built SFT dataset is a fixed singleton — a second run with different options silently trains on the first run's corpus"
scope: data-pipeline
priority: P1
status: done
rule: R10
location: engine/llm/storage.py:64-73 engine/llm/corpus/sft.py:109-111,180-205 engine/llm/orchestrate/sft.py:66-79
blocked-by: [TODO-0106]
blocks: [TODO-0108]
---

## Description

`sft.build()` always writes to three fixed paths — `datasets/sft_corpus.bin`,
`sft_mask.bin`, `sft_segments.json` — and `is_built()` asks only whether two of
those files exist. Nothing records *what* was built into them, so the reuse
check in `orchestrate/sft.py:67` (`if cfg.rebuild_dataset or not
sft_corpus.is_built()`) cannot tell a matching corpus from an unrelated one.

**This is already wrong today**, before any of TODO-0106's selection lands.
`strip_thinking` picks `Reasoning.STRIP` vs `Reasoning.KEEP`, which changes
every rendered example and its mask:

1. Run SFT with `strip_thinking: false` → builds a KEEP corpus.
2. Run SFT with `strip_thinking: true`, `rebuild_dataset: false` (the default)
   → `is_built()` is true, the build is skipped, and the run fine-tunes on the
   **KEEP** corpus while reporting a strip-thinking config.

The result is a model trained on the opposite of what was asked for, with a
`meta.json` that says otherwise and a `"dataset": "reused"` field that hides it.
Adding per-set selection turns this from one wrong axis into two.

**Shape of the fix.** Key the built dataset by what produced it — the selected
sets (names + caps, order-normalized) and the reasoning mode — so distinct
requests land in distinct locations and can coexist:

    datasets/sft/<key>/corpus.bin | mask.bin | segments.json

`is_built()` and `load()` become key-scoped rather than global, and the key is
derived in one place so the builder and the reuse check cannot disagree about
it. A stable hash of the normalized selection is enough; it does not need to be
human-readable, but writing the selection itself into `segments.json` next to
the documents makes an on-disk dataset self-describing and lets a mismatch be
detected rather than assumed.

Validating at this boundary is the point (R10): a reuse check that verifies
nothing about the artifact it is reusing is not a check.

**Migration.** Any existing `datasets/sft_corpus.bin` pair is unkeyed and its
provenance is unknown — it cannot be adopted into a key safely. Deleting it and
rebuilding on the next run is correct and cheap; do not attempt to guess which
key it belonged to.

## Definition of done

- Two SFT builds differing only in reasoning mode, or only in selected sets,
  produce two datasets that coexist on disk.
- `rebuild_dataset: false` reuses a dataset only when its key matches the
  current request, and rebuilds otherwise.
- The reused path no longer reports a bare `"reused"` with no indication of what
  was reused.

### Builder · 2026-08-10
Built SFT datasets now live at `datasets/sft/<key>/{corpus,mask}.bin + segments.json`, keyed by a readable slug plus a sha1 of the normalized `(sets, reasoning)` request. `is_built` compares against the selection recorded *inside* segments.json rather than trusting the directory name or mere file existence.

`resolve()` expands an empty selection to the sets actually on disk before the key is derived, so "train on everything" re-keys once a new set is fetched instead of reusing a build that predates it. Order is preserved rather than sorted — an extra build is the cheap mistake, reusing the wrong corpus is the expensive one. The legacy unkeyed singleton is deleted on the next build.

Verified: three builds differing only in reasoning mode or selection coexist; `is_built` accepts its own request and rejects a different reasoning mode, a different selection, and an unknown key; strip-thinking produces a measurably shorter corpus than keep.
