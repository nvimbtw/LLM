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
