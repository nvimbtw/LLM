# explore — behaviour probes, not benchmarks

A set of small, CPU-only experiments that peel the engine's *behaviour* off its
checkpoints and data. Nothing here edits engine code — every probe imports the
engine as a library and only writes JSON + an HTML report.

Run from the repo root with the engine's venv:

    engine/.venv/bin/python engine/explore/<theme>/e<num>_*.py

## Theme / branch map

One git branch per theme stems from `dev`, each carrying its own `report.html`
(what was done, key numbers, why it's a good addition, workflow impact).

| # | Theme            | Branch               | Script                 | Report                 | One-line finding |
|---|------------------|----------------------|------------------------|------------------------|------------------|
| 1 | RoPE geometry    | `explore/rope`       | `rope/e1_rope.py`      | `rope/report.html`     | default base 10k aliases ~half the channels by 128–512 tokens |
| 2 | Attention        | `explore/attention`   | `attention/e2_attention.py` | `attention/report.html` | training widens heads; no recency head ever forms |
| 3 | Embeddings       | `explore/embedding`   | `embedding/e3_geometry.py`  | `embedding/report.html` | anisotropy is 18% of norm, but table is not low-rank |
| 4 | Knowledge graph  | `explore/knowledge-graph` | `kg/e4_graph.py`     | `kg/report.html`       | embeddings see only hop-1; hop-2 = noise |
| 5 | Tokenizer        | `explore/tokenizer`   | `tokenizer/e5_tokenizer.py` | `tokenizer/report.html` | entities cost 2–12 pieces; 89.8% of vocab alive |
| 6 | Tiny real run    | `explore/training`    | `training/e6_tiny_run.py`   | `training/report.html` | canary harness; 12k tok/s CPU, anisotropy 0→0.19 |
| 7 | Entity prelexing | `explore/entity`      | `entity/e7_entity_vocab.py` | `entity/report.html`  | a 512-slot entity budget cuts corpus tokens ~9% |
| 8 | Attention temp   | `explore/attention`   | `attention/e8_attention_temperature.py` | `attention/report_e8.html` | τ=4 turns every diffuse head sharp — per-head temp has room |
| 9 | Layer ablation   | `explore/layers`      | `layers/e9_layer_ablation.py` | `layers/report.html`  | layer-0 FFN is ~10× more load-bearing than its attention |

All numbers are in each folder's `<experiment>.json` (a saved capture of the
probe's own stdout). Reports embed the actual values measured.

## How to read this

Each `report.html` has four sections: **What was done** (and the repro command),
**Results** (data), **Why it would be a good addition** (the rationale for the
engine), and **How it changes the workflow** (interaction with train/eval/CI).
A final table in `training/report.html` maps every other experiment's proposal to
a concrete A/B check in the tiny harness.