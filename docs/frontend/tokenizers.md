# Tokenizers

`/tokenizers` is the board-width vocabulary catalogue. All, Built and From runs
filter the open ledger. Vocabulary names link to `/tokenizers/{id}`; Build
vocabulary opens `/tokenizers/new-tokenizer`. There is no creation dialog.
The new and detail pages also use board width and page-native fields. Their
dataset selection and facts ledger share the catalogue's available workspace.

Creation queues a durable vocabulary build, preserving its existing dataset and
learning-ceiling contracts. Empty dataset selection means all available corpora
at execution time. `/tokenizers/builds/{execution uuid}` owns progress,
cancellation and errors, then links to the resulting vocabulary. Its execution
identity is distinct from the vocabulary UUID; see [routing](routing.md).

The detail page reads recorded vocabulary/merge/file/storage figures, origin,
build time, stable identity, fingerprint, datasets and model references. Missing
artifacts and unavailable measurements remain explicit. Recorded model references
include base/resume inheritance when the engine has that provenance.

Administrators edit name and description directly in the header. Save changes
persists both fields; Cancel restores stored metadata. Blank/oversized names,
duplicate names and failed saves preserve the draft and display an actionable
error. Rust authorizes `PATCH /tokenizers/{id}`; the engine owns persistence and
validation. Renaming never changes IDs, artifact paths, fingerprints, tokens or
historical model configurations. Non-admin readers see text, with no edit or
delete controls. Confirmed deletion preserves trained models' own vocabulary
copies and removes the packed corpora belonging to the deleted vocabulary.

The detail editor and pending requests belong to the actual browser session and
resource identity. Switching resource/account discards its draft; late reads or
writes cannot change or navigate the replacement editor. Creation has the same
session-bound handoff to its durable build. Vocabulary requests remain on Rust's
API paths; the browser never reaches the engine directly.
