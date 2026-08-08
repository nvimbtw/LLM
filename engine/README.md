# engine

Training and inference. Everything that touches a model happens here; the Rust
backend orchestrates and never loads weights itself.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

CUDA wheels for torch are resolved by pip for the local platform, so the pinned
`nvidia-*` packages are deliberately absent from `requirements.txt`.

## Running

The backend spawns and supervises this process itself — see `crates/engine`. To
run it standalone (and point the backend at it with `ENGINE_AUTOSTART=false`):

    DATA_DIR=./data .venv/bin/uvicorn main:app --port 8001 --reload

`./test_routes.sh` exercises the routes end to end.

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
448d/6L model is 18.15M parameters but 14.48M of trunk, and it wants 290M
tokens, not 363M.

Measured throughput on the RTX 3060 in this machine, at ctx 512 / batch 16, one
epoch, `bf16 + compile`:

| architecture | params | non-emb | ms/step | tokens (20x) | steps | hours |
| --- | --- | --- | --- | --- | --- | --- |
| 320d/4h/6L | 10.01M | 7.39M | 67 | 148M | 18.1k | 0.34 |
| 384d/6h/6L | 13.78M | 10.64M | 80 | 213M | 26.0k | 0.57 |
| 448d/7h/6L | 18.15M | 14.48M | 105 | 290M | 35.4k | 1.03 |
| 512d/8h/6L | 23.10M | 18.90M | 125 | 378M | 46.1k | 1.61 |
| 512d/8h/8L | 29.40M | 25.20M | 148 | 391M (all there is) | 47.7k | 1.96 |

Three things that are easy to get wrong here:

**Measure through `loop.run`, not around it.** A microbenchmark of the step in
isolation put 512d/8h/6L at 87 ms; the real loop and a real training run both
say 119-125. Every number above is the marginal rate between report frames of an
actual run, which is what `bench.py full` reports. Repeats spread about 4%, so
none of this resolves differences smaller than ~5%.

**Wider beats deeper, per hour.** 512d/6L is 33% more parameters than 384d/8L
and costs about the same per step, because 512 tiles this card's 28 SMs and 384
does not. Depth is the more expensive axis here.

**Changing `datasets` means a repack.** The packed corpus is 391M tokens; asking
for fewer sets `rebuild_dataset` and re-runs the tokenizer and the pack before
the first step. FineWeb is 99% of the corpus at ~70.5k tokens per file, so a
count of N files is roughly `N x 70.5k` tokens. If the token budget is close to
391M anyway, take the whole corpus and skip the rebuild.
