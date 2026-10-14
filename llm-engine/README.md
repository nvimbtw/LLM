# LLM engine

Training and inference. Everything that touches a model happens here; the Rust
backend orchestrates and never loads weights itself.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

CUDA wheels for torch are resolved by pip for the local platform, so the pinned
`nvidia-*` packages are deliberately absent from `requirements.txt`.

## Running

The backend spawns and supervises this process itself — see `crates/llm-engine`. To
run it standalone (and point the backend at it with `LLM_ENGINE_AUTOSTART=false`):

    DATA_DIR=./data .venv/bin/uvicorn main:app --port 8001 --reload

`./test_routes.sh` exercises the routes end to end.

Project/memory prompt contracts, reserved-token adoption and SFT source messages
are documented in [project-memory.md](../docs/llm-engine/project-memory.md).

## Data

`data/` is owned by this module and is not versioned. Corpora arrive through the
fetch scripts in `../scripts/`; the layout is described in `llm/storage.py`.

## Sizing a run

`presets.json` holds ready-made configs. To derive another one, the rule is
Chinchilla's: the compute-optimal point is about **20 training tokens per
non-embedding parameter**, trained for one epoch so no token is seen twice.

Non-embedding is the part that matters. The embedding table is tied and the
vocabulary is 8k, so it is a lookup on the way in and a matmul only on the way
out; counting it would size the corpus for arithmetic the trunk never does. A
512d/4L model is 16.80M parameters but 12.60M of trunk, and it wants 252M
tokens, not 336M.

`dimensions` and `ctx_window` must be **powers of two** — `jobs.rs` rejects
anything else and the frontend edits them as exponents. `layers` is unconstrained,
so depth is the knob for hitting a time budget.

Measured on the RTX 3060 in this machine, at 512d/8h, ctx 512, batch 16, one
epoch, `bf16 + compile`:

| architecture | params | non-emb | ms/step | tokens (20x) | steps | hours |
| --- | --- | --- | --- | --- | --- | --- |
| 512d/8h/4L | 16.80M | 12.60M | ~102 *(interpolated)* | 252M | 30.8k | ~0.87 |
| 512d/8h/6L | 23.10M | 18.90M | 125 | 378M | 46.1k | 1.61 |
| 512d/8h/8L | 29.40M | 25.20M | 148 | 391M (all there is) | 47.7k | 1.96 |

The 6L and 8L rows are measured. The 4L row is **interpolated** — step cost is
linear in depth across those two points at 11.5 ms per layer — so treat its hour
as +/-10% until a real run confirms it.

Three things that are easy to get wrong here:

**Measure through `loop.run`, not around it.** A microbenchmark of the step in
isolation put 512d/8h/6L at 87 ms; the real loop and a real training run both
say 119-125. The measured rows above are the marginal rate between report frames
of an actual run, which is what `bench.py full` reports. Repeats spread about 4%,
so none of this resolves differences smaller than ~5%.

**Depth is the expensive axis, width is not.** Between 6L and 8L every layer
costs 11.5 ms/step. Widening is comparatively cheap because 512 tiles this card's
28 SMs well — which is also why param-matched SwiGLU measured *slower* than the
GELU FFN it would replace, and why fusing Q/K/V paid.

**Changing `datasets` means a repack.** The packed corpus is 391M tokens; asking
for fewer sets `rebuild_dataset` and re-runs the tokenizer and the pack before
the first step. FineWeb is 99% of the corpus at ~70.5k tokens per file, so a
count of N files is roughly `N x 70.5k` tokens. If the budget is close to 391M
anyway, take the whole corpus and skip the rebuild.
