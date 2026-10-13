# Chat tool rounds and tool providers

The models here are too small to stop at a tool call: a large model emits one and
halts, ours emits one and carries on. So a turn is not generate-pause-resume, it
is **let the reply finish, run what it asked for while it does, then ask again
with the answers in the conversation** — see
[`../decisions/tool-calls.md`](../decisions/tool-calls.md) for what that forces.

`api/src/turn.rs` drives it, with request policy in `turn/request.rs`,
owned concurrent tool work in `turn/tools.rs`, and commit-before-acknowledgement
streaming in `turn/relay.rs`. Per round: relay the LLM engine's frames to the browser,
spawn a tool the moment its `tool_call` frame arrives, keep relaying, join at
`done`, then POST the next round with a `tool_result`. Bounded at
`MAX_TOOL_ROUNDS` (six) — without it a model that searches in every reply never
terminates. Spawning on the call rather than at the end is what makes a search
overlap the tokens after it instead of stalling them.

The bound was two, on the assumption that a turn is "the reply that calls and
the reply that answers". A model doing real work does not oblige — it reads the
output, finds the path was wrong and calls again — and at two the loop ran out
mid-investigation, leaving a reply that ended in a call nothing answered. Note
what that costs when it *is* hit: the last round's call is still parsed and
still spawned, and its result is then discarded with the loop.

The LLM engine detects, this side executes. Detection needs token *ids*, which only
the generation loop has; the API key, the outbound HTTP and the round loop are
all properties of the machine, which is this side. Each `done` is stamped
`final` so the client can tell a round's end from the turn's.

### Interjecting mid-turn

A round is generated between a tool call and its result, and that gap is a real
window: the client can send a message during it and have the backend thread it
into the *running* turn instead of waiting for the turn to finish.
`routes/stream/chat.rs` spawns a task that keeps reading the browser's socket for
the rest of the turn's life — `{ "type": "interject", "text": "..." }` — and
forwards each one to `turn::run` over an unbounded channel. A `SocketReader`
owner aborts this task on completion or cancellation of its session. The queue
remains unbounded; this refactor does not introduce a new admission policy.
No `attachment_ids` on that frame: an interjection is text steering, not a new
question. Giving it its own upload path would be a second attachment contract to keep in step
with the first for a case that does not need one.

`turn::run` drains that channel once per round, right after `answer()` has
built the next round's `tool_result` payload — `drain_interjection` collects
everything that arrived since the last drain (joined with a blank line if more
than one landed), and `fold_interjection` writes it onto the payload as
`interject`, which `conversation::prepare` stores as a second message immediately
after the tool result, in one backend transaction. The instant it folds one in, the
client is told with `{ "type": "interjected", "text": "..." }`, sent before the
next round's request goes out — the frontend's cue to move that message out of
its queued state and into the turn's trace, rather than guessing from timing.

A message that arrives with no round left is never drained. The client may
resend queued text as an ordinary new turn after close, but this is not a
lossless or exactly-once protocol: acknowledgements contain combined text, lack
individual message IDs, and precede persistence in the next preparation call.
Failures and replay can therefore lose or duplicate steering. See
[audit F06](../audits/2026-10-13-codebase.md#f06--p2--multiple-interjections-are-acknowledged-incorrectly).
The refactor preserves this wire behavior; an identified, committed steering
protocol requires a coordinated frontend/backend change.

Adding a tool is an `impl Tool` and a line in `Registry::with_config`, plus a row
in the LLM engine's table (`llm-engine/llm/chat/tools.py`). **No reserved slots**: every
tool is called through one `<|tool-call|>` marker naming it in JSON, so the tool
table is a property of the build rather than of the vocabulary — see
[`../decisions/generic-tool-calls.md`](../decisions/generic-tool-calls.md). A
tool whose configuration is missing is **not registered** rather than registered
and failing — chat has to work on a checkout with no API keys and no sandbox,
and `Registry::answer` turns the resulting `Unknown` into a `<|warn|>` the model
reads.

`Tool::run` takes the argument *and* a `session` key, which is the chat id.
Most tools ignore it — a search is the same search whoever asked — but the
sandboxed shell keeps a filesystem under it, so a reply that writes a file in
one round finds it there in the next. It is a key and not a chat: `tools` never
learns what it stands for.

A tool's answer is text a stranger wrote going into a prompt, so every tool runs
it through `tools::markup::neutralize` — the same transformation
`scripts/corpus_shards.py` applies to scraped corpus text. Without it a page
containing `<|/assistant|>` ends the turn.

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_URL` | unset | A self-hosted SearXNG, e.g. `http://localhost:8888` |
| `SEARCH_API_KEY` | unset | Brave Search, used only when no `SEARCH_URL` is set |
| `SEARCH_RESULTS` | `3` | Ceiling on how many results reach the model |
| `SEARCH_BUDGET_CHARS` | `320` | Ceiling on the whole rendered result. **Raise it with the model's context window** |
| `BASH_URL` | unset | The just-bash sandbox — `http://localhost:8899` natively, `http://just-bash:8899` under compose, which sets it. Unset, no `bash` tool |
| `BASH_BUDGET_CHARS` | `320` | Ceiling on the rendered command output. **Raise it with the window** |
| `TOOL_TIMEOUT_SECS` | `10` | Per call; the reply is already generated by then |
| `MAX_TOOL_ROUNDS` | `6` | How many times a reply may be handed tool output and asked to continue. Each round is a fresh generation the person waits on |
| `SEARCH_CACHE_MAX_AGE_DAYS` | `7` | How old a cached result may be before `web-search` re-fetches it live — see "The search cache" below |

Neither search setting means `web-search` is not registered. Both set is
SearXNG: naming an instance is a deliberate act, where a key can be left over in
an environment from something else. The startup line says which — `tools ready:
web-search (SearXNG at http://localhost:8888), bash (just-bash at
http://localhost:8899)`.

### Web search

`crates/tools/src/search/` is one tool over two providers. Everything the model
sees — query validation, the result cap, the rendering into markers — lives in
`mod.rs`, so swapping providers cannot change the shape of what a reply is
handed; a `Provider` only turns a query into `Hit`s.

**SearXNG (default, self-hosted).** No key, no quota, no third party:

```sh
cd deploy/searxng
./run.sh                                          # or: docker compose up -d
curl 'http://localhost:8888/search?q=test&format=json'   # must not be 403
```

`run.sh` is the same container as one `docker run`, for a machine without the
compose plugin — it is a separate package on most distributions. It waits for
the JSON API to answer and says what to check if it does not. Prepend `sudo` if
your user is not in the `docker` group; being in that group is root-equivalent,
so which of the two you pick is a decision about the machine.

Then `SEARCH_URL=http://localhost:8888` in `backend/.env`.

Two settings in `deploy/searxng/config/settings.yml` are load-bearing and both
fail the same way — a bare `403` that says nothing about its cause.
`search.formats` must include `json`, which is **not** a default; and
`server.limiter` must be off, because it works by profiling requests as
browser-like and an API client is exactly what it then blocks.
`searxng.rs::refusal` names both in the message, so a 403 in a chat reply tells
you which file to open.

Two details the image dictates. The **whole config directory** is mounted
writable, because the container runs as its own user and takes ownership of it
on start (`FORCE_OWNERSHIP`) — mounting `settings.yml` alone and read-only is
the obvious thing to try and fails on exactly that. And the port binds to
`127.0.0.1` only: an instance with its limiter off will be scraped by anyone who
can reach it.

**Brave (hosted).** `SEARCH_API_KEY`. $5/month of free credits at $5 per 1,000
requests, and signing up requires a card even for those.

#### Sizing the result

`SEARCH_BUDGET_CHARS` is the one worth thinking about, because the models here
are small enough that a search result competes with the conversation for room.
The LLM engine's default `ctx_window` is **256 tokens**, half of which
`orchestrate/chat.py` reserves for the reply — so the whole prompt, system block
and history included, is about 380 characters. Five untrimmed results measured
~1,200. Nothing errors on that; `build_prompt` simply evicts the question to fit
the answer to it.

So the default is deliberately mean: 320 characters, which is two results. Raise
it with the window — a model at 1024 tokens can afford four or five times this
and will answer better for it.

To see what your own instance actually returns, and what it costs:

```sh
cargo run -p tools --example search_once -- "how tall is elbrus"
```

It reads the same environment the server does and prints the rendered result
with its size in tokens.

### The search cache

`web-search` checks a cache before paying for a live fetch, and writes what it
finds back — see
[`../decisions/search-result-cache.md`](../decisions/search-result-cache.md).
The cache lives in the **data engine**, as a dataset (`web-search-cache`,
`origin = search`, `owner_id = NULL`) rather than a table of its own: a search
result is data on the same terms anything else pulled from the Hub is, and
storing it there is what makes it selectable for training later without work
in the LLM engine.

`tools::SearchCache` is the trait `WebSearch` calls through — `get`/`put`
keyed by a normalized query — and it stays a trait rather than a concrete type
because `tools` is the leaf documented above: it must not learn the shape of
the data engine. The implementation, `api::search_cache::DataEngineCache`, is
wired into `Registry::from_env` by `src/startup.rs::Services::application`, which holds
both the registry and `data_engine::Proxy`. Every call it makes is
`auth::Caller::System` — a search result is a fact about the world, not about
whoever happened to ask first, so nothing here is scoped to a person.

Freshness is decided here, not by the data engine: a lookup returns the
newest match and how long ago it was written, and `WebSearch::run` compares
that against `SEARCH_CACHE_MAX_AGE_DAYS` itself. A live fetch that then fails
falls back to a stale entry rather than erroring — the same principle
`search::mod::fit` already applies to a budget too small for a whole result:
something beats nothing.

### The sandboxed shell

`backend/crates/tools/src/bash.rs` is one tool over a shell that cannot touch this
machine. It POSTs the command to `deploy/just-bash/`, a small Node service
wrapping [just-bash](https://github.com/vercel-labs/just-bash) — a bash
*simulation* over a virtual filesystem, with `cat`, `grep`, `sed`, `jq`, `awk`,
`sqlite3` and a hundred others implemented in TypeScript and no `execve`
underneath any of them.

That is the whole reason the tool is allowed to exist. Handing a language model
a real shell on the box holding the database is not a thing to do; handing it a
simulated one costs nothing when the model writes `rm -rf /`, which it will.

```sh
cd deploy/just-bash
./run.sh                                  # installs on first run, then starts
curl -s localhost:8899/health             # {"ok":true,"sessions":0}
```

Then `BASH_URL=http://localhost:8899` in `backend/.env`. It needs Node ≥ 20.18.1
and binds loopback only — an unauthenticated command endpoint has no business on
a public interface even when its filesystem is imaginary.

**Under `docker compose up` it is a service** (`deploy/just-bash/
docker-compose.yml`, included by the root compose file) and compose points the
backend at `http://just-bash:8899` itself. Nothing above is needed then, and the
`backend/.env` line stays right for `cargo run` — dotenvy does not overwrite a
variable compose already set.

That it is containerised there is not a retreat from "the library *is* the
sandbox". It still is: bash simulated over a virtual filesystem, no `execve`
under any of it, nothing a container would be isolating. What the container
buys is an **address**. A backend inside a container resolves `localhost` to
itself, so a sandbox on the host's loopback is unreachable from it no matter
what is listening — the tool registers, and every call fails on connect.

**One filesystem per chat.** The service keeps a sandbox per `session` key, the
chat id the backend sends, so `echo x > f.txt` in one round is readable in the
next. Sessions expire an hour after their last command and the oldest is
evicted past 256 of them (`BASH_SESSIONS`, `BASH_SESSION_TTL_SECS`); eviction is
silent, and the next command simply gets a fresh filesystem the way a rebooted
machine would.

**Everything optional is off.** No network, no Python, no JavaScript runtime,
in-memory filesystem only. Each is a capability, and a capability this project
has not decided to hand a language model is one it does not have — turning any
of them on is an edit in `server.mjs`, deliberately.

What the model is handed back mirrors the search tool's discipline: the output
in a `<|codeblock|>`, a non-zero exit named in a `<|warn|>` before it, silence
said rather than shown as an empty block, and the whole thing held to
`BASH_BUDGET_CHARS`. A command that overruns is cut at a line boundary and told
so, because a model that needs the end can say `| tail` — which is the habit
worth teaching anyway.

**That rendering is exported, and it is the reason the gym is driven from
here.** `tools::observation` is the same function, called with the fields the
gyms engine reports for a command it ran; a second renderer in Python would
mean a model practising against a shell it is never actually served. See
[the gym socket](gyms.md).
