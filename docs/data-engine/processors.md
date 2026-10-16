# Processor authoring, reloads and provenance

`processors/helpers.py` is the public authoring surface. Drop one UTF-8 file
declaring one decorated function into `processors/custom/`:

```python
from processors.helpers import Ctx, Param, processor

@processor(name="example", title="Example", kinds=["pretrain", "sft"],
           params=[Param(name="minimum", label="Minimum", type="int", default=0)])
def run(text: str, p: dict, ctx: Ctx) -> str | None:
    """Uppercase text that meets the requested minimum length."""
    return text.upper() if len(text) >= p["minimum"] else None
```

The decorator returns the existing `Processor` entry. Legacy `Apply(doc,
params,state)` still works. Builtin text/filter metadata, parameter coercion and
clamping are unchanged.

- `mode="text"` rewrites prose or each turn. One `None` drops the whole exchange.
  Roles, metadata and other fields survive.
- `mode="turns"` receives/returns the full message list, or `None` to drop it.
- `mode="document"` receives/returns a document. Length and dedup filters use it
  to preserve whole-document policy and newline-joined conversation lengths.
- `get_text`, `set_text`, `map_messages` and `drop` expose shape operations.
  Empty prose stays empty rather than falling back to turns.
- `ctx.state` is per-run shared scratch; processors should namespace keys.
  Each preview starts fresh.

`strip_urls.py` removes HTTP/HTTPS/www URLs, drops whitespace-only results and
applies the optional `min_len` threshold.

## Loading

Startup merges custom files with curated builtins; duplicate names abort
startup. Malformed/prohibited files are logged and logically quarantined without
moving bytes. A bad edit retains its last working entry. A runtime collision
retains the complete previous registry. Requests/workers resolve immutable
registry generations.

The lifespan owns a polling thread and joins it at shutdown. It polls every
200 ms and reloads after 200 ms of stable state. PUT explicitly reloads too.
Development Uvicorn commands exclude `processors/custom/*` from process reloads
and install pinned `watchfiles` to honor that exclusion. The catalogue watcher
owns custom edits, preserving the active callback after an invalid save. Other
service code changes still restart Uvicorn. See [Uvicorn reload settings](https://www.uvicorn.org/settings/).

After updating the pinned development watcher dependency, rebuild the service:

```sh
docker compose up -d --build --no-deps --force-recreate data-engine
```

There are at most 128 custom files, each at most 1 MiB; source writes are bounded
by UTF-8 bytes. Symlinks are rejected. An edited entry must keep its name.

AST checks and an import guard permit `re`, `html`, `unicodedata`, `json`,
`urllib.parse`, `dataclasses`, `math`, `collections`, `itertools` and
`processors.helpers`. Storage, relative/private imports and private attribute
access are rejected. This catches accidental boundary violations; it is **not
an untrusted-code sandbox**. Files and edits execute in process under operator
trust.

## Source API

Browsers use Rust's `/data/processors/{name}/source`: GET requires `AuthUser`,
PUT `AdminUser`. The `agents` development account is an administrator. Rust
replaces browser credentials with a service token; Python authenticates the hop
and owns file identity/writes. Builtins are viewable and read-only.

Administrators can create a processor with `POST /processors {source}` through
Rust's `POST /data/processors`. The first definition must load successfully;
syntax/import/metadata errors return 422 without creating a file. Its declared
name becomes the stable identifier and filename. Existing identities/files
return 409 and are never overwritten. `new-processor` is reserved for the browser
creation route. A successful create returns the same source representation as GET.

Callbacks must accept value, parameters and context. Custom metadata must have
a nonblank title, unique named parameters, supported scalar defaults and numeric
ranges; non-JSON values are rejected during loading. Source replacement preserves
file permissions and, when running as root, the operator's existing ownership.

GET returns `{name,title,file,source,editable,version_hash,active_version_hash,
reload_error}`. `version_hash` hashes saved bytes; `active_version_hash` hashes
loaded code. Unknown names return 404, builtin writes 403, oversized source 422.

`contracts/processor-source.json` is shared by the Python HTTP tests and frontend
decoders, covering saved/active divergence and catalogue metadata.

PUT accepts `{source}` for a known custom entry. It atomically replaces and
flushes bytes before reloading, without rejecting Python syntax before saving.
It returns `{name,saved:true,version_hash,active_version_hash,reload_error}`.
Invalid code remains saved and preview uses the prior callback; repair and save
again to activate it. Catalogue/preview always report active hashes.

## Version records

Submission stores name, resolved parameters and `version_hash`. Workers capture
current callbacks at startup, log drift and persist `executed_version_hash` and
`version_mismatch` alongside the submission hash on `datasets.steps` and in run
results. Preview includes the same `steps` metadata and aggregate before/after;
no per-step document trace is collected. A pipeline has at most 32 steps.

Hashes cover file bytes without retaining historical source or hashing helper
dependencies. Queued jobs execute current code after a mismatch. Historical
unversioned rows remain readable; legacy queued jobs record their executing
version. Restarting with invalid source quarantines the entry: last working
callbacks only survive in memory.
