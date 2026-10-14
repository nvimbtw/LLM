# Private projects and source memories

Rust owns private projects and each chat's optional project membership. Every
project read and write checks the authenticated owner; admin access grants no
access to another user's project or conversations. The database enforces matching
owners through a composite foreign key. Projects have a name (1–120 characters,
nonblank), description (at most 2000 characters) and notes/instructions (at most
8000 characters), plus creation and update timestamps. Lengths count Unicode
characters. Project names need not be unique.

## HTTP routes

| Route | Behavior |
| --- | --- |
| `GET /projects?limit=50&offset=0` | Returns `{projects,total,limit,offset}` for the authenticated owner. |
| `POST /projects` | Accepts `{name,description?,notes?}` and returns committed metadata with 201. |
| `GET /projects/{id}?limit=50&offset=0` | Returns project metadata plus `threads`, `threads_total`, `threads_limit` and `threads_offset`. |
| `PATCH /projects/{id}` | Applies supplied name/description/notes and returns metadata. |
| `DELETE /projects/{id}` | Returns 204 after clearing membership; all chats, messages and files remain stored. |
| `POST /chats` | Accepts optional `project_id`; checks ownership and creates membership atomically with the chat. |
| `PUT /chats/{id}/project` | Requires `{project_id: UUID|null}`, moves or detaches an owned chat, and returns chat metadata. |

Project metadata contains `project_id`, `name`, `description`, `notes`,
`thread_count`, `created_at` and `updated_at`. Chat metadata now includes nullable
`project_id`. Existing `GET /chats` still lists every owned chat, including project
chats. Project and thread pages accept 1–200 rows and offsets from 0 through
2147483647, newest first with UUID tie-breaking. Unknown and foreign project/chat
IDs share a 404. Empty descriptions and notes clear those fields on PATCH;
omitted or null fields leave them unchanged.

Project detail metadata, `thread_count`/`threads_total`, and the returned thread
page share one repeatable-read database snapshot. Concurrent additions, moves or
deletions cannot combine a count from one state with a page from another.

Moves take the same per-chat process-local turn mutex as generation and
context-floor requests. A project deletion can occur while a turn runs: context
already accepted by that generation may finish, while later rounds reload
membership and obtain no project context after detachment. Multi-process turn
serialization retains the limitation described in [conversations](conversations.md).

## Retrieval and the engine boundary

The shared conversation preparation path handles both HTTP and websocket turns.
It strips browser-supplied `project`, `project_id` and `memories`, loads authorized
visible history, and derives context from stored membership. Retrieval chooses
the newest submitted interjection or user text; a tool continuation without new
user text uses the latest user message remaining after the authorized history
floor. Client-supplied turns never provide a retrieval source. Project metadata
and source memories are read in one consistent database snapshot.

Only user and assistant messages in **other chats in the same owned project**
qualify. A generated `simple`-dictionary word vector over each message's first
8192 characters and partial GIN index supply
word matching; it uses exact lowercase words, rather than embeddings, summaries
or stemming. The query takes at most 24 distinct 2–64-character alphanumeric
words from at most 8192 source characters, omits common question words, and joins
the terms with OR. One shared word can match a source even when the natural
question contains additional terms. Rank orders by word overlap, then recency
and source UUID. Queries have a two-second PostgreSQL statement deadline; a
storage failure prevents accepting the new input with partial context.

The engine receives at most eight source snippets, each containing a PostgreSQL
word-matched excerpt of at most 2000 Unicode characters, with original thread and
message UUIDs, `user`/`assistant` role and nonnegative `age_seconds`. Excerpt
selection examines only the first 8192 source characters; later words do not
qualify, and this ceiling also bounds migration and indexing work for long legacy
messages. Before indexing and excerpt selection, an immutable role-aware database
function removes every complete assistant `<|thinking|>` block and any unfinished
opener-to-end suffix. User messages remain literal, so quoted markers do not hide
their content. This prevents excerpt selection from removing a thinking opener
and recalling its hidden body as ordinary text. Short visible messages are
forwarded exactly; longer messages use a matching excerpt. Project metadata
is delivered even when no snippets match. Stored notes and recalled messages
are untrusted text for prompt rendering; Python neutralizes markers and owns
token fitting and overflow policy. There is no memory write-back or summary
store.

Trusted context is attached before `/chat/prepare`, so tokenizer compatibility
errors occur before new input persistence, and carried to `/chat` or
`/chat/stream`. `/chat/context-floor` receives the same project metadata and
retrieval policy over the current authorized history. Standalone chats send no
project or memory fields. The shared [project-memory fixture](../../contracts/project-memory.json)
checks the Rust/Python wire fields.

## Verification

`db/tests/projects.rs` runs real PostgreSQL migrations, checks owner consistency,
atomic membership, cross-project and role exclusions, source identity, deletion
preservation, Unicode snippet limits and age bounds. `api/tests/projects.rs`
checks private CRUD including admin isolation, pagination and field validation,
real engine HTTP forwarding, tool continuation/interjection retrieval,
context-floor parity, malicious context stripping and a real browser websocket
generation. Run these through the backend module's documented Cargo verification
commands with a disposable PostgreSQL instance.
