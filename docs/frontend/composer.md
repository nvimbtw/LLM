# Chat composer

## Layout

An established conversation uses a compact editor for empty and one-line drafts.
The model picker, attachments, microphone, Send/Stop and context footer stay
available. Wrapping, additional paragraphs and attachments add only their natural
height; typing the first character or focusing does not enlarge the editor.
Sending or clearing reduces it again. The shell measures its content and animates
height changes over 200ms; reduced motion removes that transition. The new-chat
composer keeps a larger editor with a slightly reduced empty footprint; its
heading artwork retains its full size. Loading placeholders reserve the
corresponding editor height.

In an active chat, the transcript extends behind the frosted composer so its
translucency and blur remain visible while scrolling. The composer's measured,
animated height reserves matching bottom padding in the transcript. This moves
the latest content and status throbber above its edge as it grows. A reader at the
bottom stays pinned; someone reading earlier messages keeps their position.
Question navigation centers within the clear reading area above the composer.

The composer shell keeps its resting border when focused. Its text caret indicates
editor focus; individual buttons still show their keyboard focus outlines.
Deleting a fully selected draft collapses the selection to a caret, so the empty
placeholder carries no leftover selection highlight. Undo restores the draft.

## Attachments

A file is uploaded **when it is picked**, not when the turn is sent
(`lib/attachments.ts`). That is the whole design decision on this side, and it
buys one thing: the composer on the home screen has no conversation to attach
to, so any scheme that holds files until send makes the first message of a new
chat a different code path from every other one. An upload belongs to its owner
until a turn claims it, and `attachment_ids` on the socket's opening frame is
what claims it.

The upload lifecycle belongs to `lib/composer-attachments.ts`, subscribed through
`hooks/use-composer-attachments.ts`:

- **Emptying the strip comes in two kinds.** After a send the rows belong to the
  turn, so only the local object URLs are released. Abandoning a draft
  (`/clear`) has to delete the rows too, or every discarded draft leaves stored
  files nothing will ever claim.
- **Nothing sends while an upload is in flight.** A turn stored without the file
  the person is watching arrive cannot be given one afterwards — there is no
  route that edits a stored message.
- **A failed upload stays visible** rather than being dropped from the strip, and
  is not sent. A file past the per-turn cap becomes one of these without being
  uploaded at all, so an oversized drop says which files are not going.

Rendering one costs a fetch: the bytes are behind the same bearer token
everything else carries and an `<img src>` cannot send a header, so
`useAttachmentUrl` fetches the blob and hands back an object URL it revokes on
unmount. Thumbnails sit in a fixed box, so a thread of mixed aspect ratios does
not reflow as each one lands.

**The strip has one height, not one shape.** A picture and a document are
different objects and get different tiles, but both are `h-28`, because a mixed
row whose tops do not line up reads as two things that happen to be adjacent.
The file tile lifts the extension out of the name and into the meta line
(`TXT · 5.0 MB`): a card that wide truncates the end of a long filename, and the
end is exactly the part that says what the file is.

## Dictation

The microphone next to Send is the browser's own `SpeechRecognition` — free,
keyless, nothing in the bundle, and absent on Firefox, where the button does not
render at all. What it costs in privacy and why it beat a local Whisper is in
[`decisions/voice-input-uses-the-browser.md`](../decisions/voice-input-uses-the-browser.md).

Two hooks, because they are two browser APIs with two lifetimes:

- **`use-speech-recognition.ts`** owns the recogniser. It hands back `final` and
  `interim` as separate strings — the second is a guess the recogniser keeps
  rewriting — plus `onDevice`, true when Chrome's local pack is already
  installed. It restarts its own session on `onend`, because a session ends by
  itself after a stretch of silence even with `continuous` set; a run of
  sessions too short to have heard anything is how that restart chain ends
  rather than spinning against a browser that has said no.
- **`use-audio-level.ts`** opens a *second* capture of the same microphone,
  because `SpeechRecognition` exposes no audio to draw. It calls back once per
  frame with three bands and never sets state; `VoiceWaveform` writes the
  transforms to the DOM itself.

In `hooks/use-composer-dictation.ts` the spoken text lives in a **range** — where this
dictation is writing — that is replaced whole on every event rather than appended
to, since the interim tail is revised and not added to. The range is ProseMirror
positions, so the editor's `onTransaction` maps it through anything typed
alongside it, skipping the composer's own inserts by a meta flag. The interim
tail carries the `interimTranscript` mark (faded, `inclusive: false`), which is
stripped when dictation ends and which `docToMarkdown` would ignore anyway.

Sending or `/clear` stops dictation. The footer band holds the three bars in its
middle column while it runs — it is a `1fr auto 1fr` grid rather than a two-ended
row precisely so the bars are centred on the band itself, and so the model choices and
the context meter never move to make room for them. The bars are also the only
thing that animates: the mic button just fills blue and swaps to a stop glyph,
the same swap the send button makes mid-turn. `Ctrl`/`Cmd`+`Shift`+`Space`
toggles it; the mic button takes both a click (dictate until clicked again) and
a hold (dictate until released), told apart by how long the press lasted.

## The context meter

The right end of the footer band: a bar and `used/window`, tinted at 70% and
again at 90% (`meterTone`). It counts two things that are known to different
standards, and keeping them apart is the whole of the design.

- **The conversation behind the draft is an estimate.** The last persisted total
  from the chat's history response, plus whatever a turn in flight has tallied
  client-side as it streamed — see `chat-thread.tsx`. A completed turn replaces
  the live estimate and stored total together. Persisted totals sum the token
  counts recorded with messages; they do not recount the fitted prompt under the
  currently selected model's tokenizer.
- **The draft in the composer is a real count.** `hooks/use-draft-tokens.ts`
  posts it to `/tokenize/count` 200ms after typing settles, and the LLM engine
  encodes it under that model's own vocabulary. A guess from character count
  would be wrong by an amount that changes with the model — these vocabularies
  are trained per model — against windows that start at 256 tokens, where a
  handful either way is a percentage.

So the bar is drawn as two segments, the draft's at 45% opacity, and the tooltip
says which number is which. The alternative — one figure labelled "estimated" —
threw away the half that is exact.

Three things the hook does that are not optional:

- **It counts the markdown, not the text on screen.** `docToMarkdown` is what
  `submit` sends, and a list's `- ` is in the string the model is handed.
- **It holds the last count while a request is in flight**, and on failure keeps
  it rather than falling to zero. A meter that emptied itself mid-sentence would
  be a worse lie than one a keystroke stale.
- **It returns to zero without a round trip** when the composer empties, adjusted
  during render rather than in an Effect — otherwise the count of a turn already
  sent survives a frame into the next one.

A recognised standalone command (`/clear`, `/model …`) reads as zero: it runs
locally and never reaches the model, so it costs the window nothing.

Counting per keystroke is only affordable because the LLM engine caches the parsed
vocabulary — see [`../llm-engine/README.md`](../llm-engine/README.md). Attachments
contribute nothing here, and that is a property of the system rather than an
omission: nothing reads them
([`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md)).

## `/prune`

Cuts a chat's usable range back to what it would be at `context_multiplier: 1`,
while leaving the multiplier itself wherever it's set — real headroom past the
trained window again, instead of already sitting near a raised cap. Nothing is
deleted: every message stays exactly where it was, visible in the thread and
counted in `/usage`. What changes is what a *future* turn is built from.

The mechanism is deliberately stateless on the LLM engine's side. `GET
/chats/{id}/context-floor?model_name=...` is a pure read — it answers "where
would the cut land under *this* model's own `ctx_window`," using the identical
walk `build_prompt` already runs, and writes nothing down. `lib/context-floor.ts`
is what remembers the answer: a message id **and the model it was cut for**,
kept in `localStorage`, per chat, at the same durability tier as every other
chat setting in this file — lost on a cleared browser or a different device,
same as your temperature would reset.

**A floor is only ever resent to the model it was computed for.** `ctx_window`
is a property of the model, not the chat, so a cut sized to one model doesn't
mean anything for another — `sendMessage` in `use-chat.ts` compares the stored
`modelId` against the model the turn is actually going to and sends
`history_floor` only on a match; switching models makes a stale floor inert
rather than wrong, without discarding it, so switching back makes it live
again with no extra round trip. The LLM engine's own fallback (a stale or
unrecognised floor is treated as no floor rather than erroring, `_floored` in
`api/routes/chats.py`) is what makes "inert" safe to fall back to: a client's
memory of a prune is a convenience, not something a turn should fail over.

If a prune ever needs to survive a lost `localStorage` or a second device, this
is the point to reconsider: persist the floor against the chat server-side
instead of trusting the client to keep resending it.

The composer footer subtracts the floor's own reported token count from
`chatSummary.tokens` rather than asking the LLM engine for a second, smaller total
— `chatSummary.tokens` is also what `/usage` bills from, and has to keep
counting every token this chat ever cost regardless of what a later prune
excludes from the model's view of it.

**The meter's badges are how a prune is actually seen.** Past the trained
window under a raised multiplier, one reads `+190% ×4` — how far over the
trained size you are, and what's letting you get away with it. Once `/prune`
succeeds, a second badge appears beside it: `62% unseen`, how much of this chat
the model can no longer see. `use-chat.ts` reports this as 0 — same as never
having pruned — the moment the stored floor stops matching the selected model,
so switching models doesn't leave a stale "unseen" figure claiming a cut that
isn't actually being applied to what's being asked now. They are independent
facts shown independently,
not two views of one state — pruning brings usage back under the trained
window the instant it runs, but the conversation can grow past that window
again afterwards, at which point both are true at once (some of the chat is
unseen, *and* what's left has gone past the trained size again). Both badges
read off the same `tone` (`composer-footer.tsx`'s `meterTone`) as the bar and
the count beside them, so either turning red means the same thing.

## Settings

Settings and account controls share a modal with section navigation on the left.
The Profile button opens its Profile section. See [Settings and profile](settings.md)
for entry points, persistence, and the standalone `/profile` and `/usage` routes.

## The composer's own choice

`model` and `budget` — what to send with, not how to behave — are not in
`Settings` and never render in the ⌘K modal; they stay `ComposerChoiceProvider`
state (`components/chat/composer-choice-provider.tsx`), read and set through
`useComposerChoice()`, exactly as before. What changed is that the choice now
persists too, through its own resource, `GET`/`PUT /user/me/composer-choice` —
a separate blob from `Settings` rather than two more fields on it, because a
model or budget pick writes immediately (it's one discrete action, not a
dragged slider) while a settings change debounces; sharing a column would mean
whichever write lands second clobbering the other's field. Same
account-not-browser reasoning, same derive-on-logout shape as `Settings` above.

A persisted model id can go stale between sessions — the model got deleted or
unpublished — so `ComposerPicker` corrects an id that isn't in the loaded list
back to the first available one, the same effect that already handled "no
choice yet"; that's the one place with the list, so it's the one place that
needs to know.

**The stored id may also name a checkpoint**, which is in no model list by
design, and so is indistinguishable from a stale one until asked about.
`hooks/use-checkpoint-parent.ts` is what asks — one request, cached against the
id it answered for — and the correction above waits on it, or every checkpoint
choice would be discarded the moment it was made. It also supplies the model a
checkpoint belongs to, which is what the trigger names and what the list marks
as active. Nothing on the chat path ever learned what a checkpoint is: it is a
model id and a directory, exactly like the finished weights
([`../decisions/checkpoints-are-models.md`](../decisions/checkpoints-are-models.md)).

Model selection sits at the left of the footer, followed immediately by a
separate thinking dropdown; context stays right. The compact searchable picker
opens on published models. Administrators can switch to All models. Checkpoint
options are absent, but existing persisted checkpoint ids keep their identity
through parent resolution. The footer no longer repeats the conversation title
or rename control; sidebar and archive renaming remain available.

## Slash commands & fuzzy finder

The composer also accepts an optional `plan` snapshot. Its presentation and
the interactive `/markup#composer-plan` fixture are documented in
[Composer plan](composer-plan.md). `/plan add`, `/plan edit`, and `/plan save`
manage browser-local plans in live chat, with draft editing and drag or keyboard
reordering. Plans are not yet synchronized with the backend or model.

Typing a leading `/` in the composer opens its fuzzy finder drawer
(`components/chat/slash-command-menu.tsx`, driven by `lib/commands.ts`).

Commands and plans share a faint grey textured region inside the existing
composer outline, above the message editor. Slash discovery temporarily replaces
the plan and restores it on dismissal. Suggestions retain their bottom-up match
and keyboard order, with bounded scrolling and reduced-motion support. No
separate drawer, outline or shadow surrounds this upper region.

- **Stage 1 (Command Discovery)**: Fuzzy-matches command names, aliases, and
  descriptions (e.g. `/model`, `/thinking`, `/show-thinking`, `/temperature`,
  `/top-k`, `/top-p`, `/context-multiplier`, `/prune`, `/system-prompt`,
  `/notify`, `/theme`, `/settings`, `/help`, `/clear`, `/new`).
- **Stage 2 (Argument Selection & Range Guides)**:
  - `/model <name>`: Fuzzy-filters and selects available models.
  - `/thinking <budget>`: Sets the thinking budget (`auto | low | med | high`).
  - `/show-thinking` / `/notify`: Toggles reasoning visibility and notifications.
  - `/temperature`, `/top-k`, `/top-p`, `/context-multiplier`: Displays allowed
    ranges, current values, and clickable preset buttons.
  - `/prune`: No argument. A no-op toast at `context_multiplier: 1`; otherwise
    asks the LLM engine for this chat's floor and toasts what it cut — see the
    note on `/prune` above.
  - `/system-prompt <text>`: Previews current prompt and offers templates or `clear`.
  - `/theme <light | dark | system>`: Switches visual theme.
  - `/new`: No argument. Navigates to `/`, the empty composer — the same
    destination the sidebar's own "New" link goes to, not a route this command
    invented. Available from any composer mount, including the new-chat screen
    itself, where it's a no-op navigation to where you already are.
- **Keyboard navigation**: Seamless `↑` / `↓` navigation, `Tab` autocomplete,
  `Enter` execution / selection, and `Esc` dismissal. Standalone commands execute
  locally with toast feedback and are never dispatched over the chat socket.
