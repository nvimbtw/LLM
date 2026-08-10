---
id: ISSUE-0120
type: issue
title: "engine integration tests point DATA_DIR at a path outside the repo, so all ten fail instead of running"
scope: backend/llm
priority: P2
status: todo
rule: R10
location: backend/crates/engine/tests/client.rs:23
blocked-by: []
blocks: []
---

## Description

`config()` builds the child engine's environment with:

    data_dir: PathBuf::from("../../data"),

That is relative, and it is resolved against the *engine's* working directory
rather than the crate's. The engine runs from `dir` (`../../../engine`), so:

    engine cwd                 /home/x_x/Programming/LLM/engine
    DATA_DIR "../../data"  ->  /home/x_x/Programming/data

which is one level above the repository and has never existed. Every test that
gets as far as talking to the engine fails on it:

    unexpected terminal frame: Error("ValueError: No .txt files found under
    /home/x_x/Programming/LLM/backend/crates/engine/../../../engine/../../data/input")

All ten tests in the file fail this way — the seven that predate the run-socket
work and the three added with it. It is not a regression from either; the line
has been wrong since the file was written.

**Why it went unnoticed.** The crate's manifest says these tests "need the
engine's venv present and are skipped (not failed) without it", and `config()`
does return `None` when the venv is missing. On a machine with the venv they run
and fail instead. The workspace also did not compile its test targets at all
until ISSUE-0120's sibling cleanup, so `cargo test --workspace` never got far
enough to report these.

**The fix is not simply correcting the path.** Pointing `data_dir` at the real
`engine/data` would make `cargo test` train against the live 391M-token corpus
and write models into `artifacts/models/` beside the user's real ones —
`tiny()` asks for one epoch over whatever is selected, and with no `datasets`
that is everything. A test suite must not do either.

What it needs is a fixture: a temp DATA_DIR seeded with a few kilobytes of text
under `input/<something>/`, so the tiny architecture in `tiny()` trains in
seconds against data the test owns. `synth`'s e2e tests already do exactly this
(`std::env::temp_dir().join(format!("synth-e2e-{uuid}"))`) and are the model to
copy.

Until then the guard in `config()` should also check that the configured data
dir has input text, so these skip cleanly rather than failing — restoring the
behaviour the manifest already claims.

## Definition of done

- `cargo test --workspace` is green on a machine with the engine venv present.
- The tests do not read or write the real `engine/data`.
- A skipped run says why it skipped.
