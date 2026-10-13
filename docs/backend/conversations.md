# Conversations and files

Rust owns chats, messages, uploads, feedback and usage. CRUD/history/file routes
query the backend database and remain usable when inference is unavailable.
Every query scopes by authenticated owner; admin status does not open another
person's conversations. `users → chats → messages → attachments` deletion
cascades locally. Unsent attachments belong directly to their uploader.

Messages have a role and a per-chat position. Insertions lock the chat row;
a tool result and buffered interjection commit in one transaction. A process-wide
per-chat turn mutex spans generation and tool rounds, so sockets cannot interleave
history in the single backend process. Multi-process backend deployment requires
a cross-process turn lease before it is supported.

`model_id` is a logical remote UUID; `model_name` is a snapshot at generation.
Renaming/deleting a model leaves old attribution intact. Token counts come from
Python, never a Rust approximation. History and usage need no live model lookup.

For a new turn, Rust checks ownership, calls `/chat/prepare` for canonical tool
rendering and counts, atomically stores its new messages and binds uploads, then
calls stateless `/chat` or `/chat/stream` with authorized history. Templates,
context fitting, tokenizer work and generation stay in Python. `/chats/{id}/context-floor` passes
history to `/chat/context-floor` and translates the returned index into message
IDs; the client's stored floor and prefetch/cache invalidation behavior are
unchanged.

A completed assistant reply commits before HTTP success or a streamed `done`.
A disconnect before completion leaves the accepted user/tool messages but no
partial assistant message. If commit succeeded but delivery failed, history is
the authority. Clients send `request_id`; the `(chat, request, role)` uniqueness
constraint refuses duplicate submission with 409 rather than adding another
turn. A deliberately new request gets a fresh ID. Tool follow-ups each get a new
request ID. No claim of exactly-once external tool execution is made.

Uploads are at most 50 MiB, at most five per user turn. Binding requires all IDs
to be unsent and owned; failure rolls back the entire message batch. Only unsent
uploads can be discarded separately. Download disposition remains inline with
UTF-8 filenames. Files are stored as bytes and never enter model prompts.
Metadata queries exclude content bytes.

Usage is derived from stored messages: user tokens asked, assistant tokens
produced, replies and model-name snapshots. Source cutover and recovery are in
[ownership-cutover.md](ownership-cutover.md).
