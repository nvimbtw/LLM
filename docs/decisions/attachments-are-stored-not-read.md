# Attachments are stored, not read

**2026-08-20**

A file put in the composer is uploaded, kept, and shown back in the history. No
part of it reaches a prompt. It lives in the engine's database, beside the
messages it hangs off.

## What forced it

The composer had an attach menu, a drag target and a strip of thumbnails, and
`chat-composer.tsx` discarded every one of them on send. The files existed for
as long as the tab did. Making them real was worth doing on its own — a
conversation you can look back through is worth more with the picture still in
it — and it opens the design work the strip was built for.

Feeding them to the model was never on the table. The models served here have a
context window measured in hundreds of tokens and no vision at all; a PNG is not
something they can be handed in any form.

## Where they live

**The engine**, which is not where instinct puts user uploads — the module is
named for training and inference — but is where the boundary puts them.

An attachment's lifetime is exactly a message's lifetime, and the messages are
rows in `slate_engine`. Everything that makes the feature cheap is free only in
that database: deleting a chat takes its files with it through the existing
cascade, `DELETE /owners/{id}` already sweeps them, the history request that
renders a conversation already selects the rows they hang off, and the scope
check is the one `chats` and `messages` already pass.

**The data engine** was weighed and rejected. Its identity is text on its way to
becoming training data: documents with an ordinal, a provenance and a processor,
pulled and exported in bulk. An attachment is none of those. And it would put
the file in a different database from the message, so `message_id` becomes a
reference with no foreign key behind it — orphaned blobs on every chat delete,
and an account deletion that can half-succeed across two services.

**The Rust backend** was the tempting one: it owns the user, it is the least
code (no multipart forwarding, no new Python dependency, it could serve the
bytes itself), and an upload is arguably a property of a person. Rejected for
the same cross-database reason, and because "the backend owns users and presets
and nothing else" survived the migration that moved conversations *out* of it.
It is not a line to spend on saving a hop.

## What keeps them away from the model

Placement was never going to enforce this, so it is enforced where it can be:

- **Nothing under `llm/` imports the attachments module or names the table.**
  The bytes are touched by `store/queries.py` and `api/routes/attachments.py`,
  and by nothing else.
- **`attachment_ids` on a chat request binds rows and returns.** It is not read
  into `ChatConfig`, not counted into `tokens`, not rendered into the
  conversation `build_prompt` sees. `test_attachments.py` asserts the stored
  question is the text the person typed and nothing more.
- **A tool result may carry none.** The backend's `carry_settings` drops the
  field on a follow-up round because its allowlist never named it; the engine
  refuses one outright, which turns that from a coincidence into a contract.

If a model here ever *can* read a file, this is the record to come back and
overturn — deliberately, not by wiring a field that was left lying around.

## What the bytes cost

A `bytea` column, on the same terms
[`dataset-content-in-postgres.md`](dataset-content-in-postgres.md) took: it only
works with caps, so there are two, in `engine/api/routes/attachments.py` and
mirrored at the edge in the backend's route. 50 MB a file, 5 files a turn — the
product is the number that matters, since a turn's files pass whole through
three processes, and the count came down when the per-file ceiling went up. An
upload is read in chunks and refused the moment it passes the ceiling rather
than after the whole of it is in memory, and `Attachment.content` is a deferred
column so the history read that lists ten files does not load ten files.

Files under `engine/data/` were the alternative. That tree is documented as
derived artifacts and nothing else — a person's upload is derived from nothing —
and the repository already has a record of moving *away* from directories for
the indexing cost they carry.

Real object storage (MinIO beside `deploy/searxng/`) stays available and would
change one function, because every read goes through `GET /attachments/{id}`.
It is not worth a container yet.

## What it cost the seam

The one thing this side of the system had never done: forward something that is
not JSON. `service::Proxy` gained `upload` and `fetch` — a multipart request out,
a buffered response with the service's own content type back. Buffered rather
than streamed because the cap above bounds it, and because a streaming body
cannot carry an upstream failure through a response this side has already begun
to answer.
