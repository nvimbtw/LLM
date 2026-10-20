# Chat message presentation

## Message position rail

Conversations show a quiet index of user questions centered vertically in the chat
pane's left gutter when that pane is at least 768px wide. Each question has a
neutral stroke; the current exchange has a slightly longer, darker hairline. The current position
follows the question whose center has passed the viewport's center; replies retain
their preceding question's position. At either end, the first or last question is
current.
Trace expansion, streaming growth and viewport changes update the position too.
Long indices scroll within the rail and keep the current stroke in view.

Hovering or focusing a stroke previews its question. Buttons expose the question
number and current position to assistive technology. Selecting a stroke centers
that question with a short eased scroll (240–420ms) and pauses automatic
bottom-following. A new selection replaces the jump; a wheel, touch, pointer or
keyboard gesture in the transcript cancels it. Reduced motion makes jumps instant;
scrolling back to the bottom resumes following. Narrow screens retain the full
reading width without the rail. Queued interjections get no position until sent.

## Opening one

`AttachmentViewer` is a panel beside the thread, not a dialog over it — the
point of opening a file mid-conversation is to read it *against* what was said
about it, and a modal would take the conversation away to show its attachment.
`ChatThread` is a flex row: the chat column is centred in whatever width is
left, so it drifts left as the panel arrives and back when it closes, with no
second layout to maintain. Below `lg` there is no room for both and the panel
takes the width; a half-conversation beside a half-document is neither.

Which state lives where matters: the open file is `ChatThread`'s, not a
message's, because opening a second one has to *replace* the first — a turn that
owned its own viewer could not do that.

Four previews, and each is its own component because each needs a different
fetch and a hook cannot be called conditionally:

| Kind | What it is | Shown as |
| --- | --- | --- |
| `image` | `image/*` | The picture, contained |
| `text` | `text/*`, JSON/XML/YAML, or a known extension | The decoded contents |
| `embed` | `application/pdf` | An `<iframe>` — the browser's own viewer |
| `none` | anything else | A note, and the download in the header |

Extensions are trusted over the media type for the text case, because the
browser guesses the type from the extension anyway and guesses badly: a `.bak`
arrives as `application/x-trash`, which is a verdict rather than a type.

**A text preview is capped** at `PREVIEW_CHARS` (200k). People attach novels —
one of the files this was built against is a five-megabyte `.txt` — and laying
that out as one text node stalls the tab. The rest is stored and downloadable,
and the panel says so rather than letting the reader discover the file stops
mid-sentence.

Nothing here is sent to the model, and nothing is counted against the context
window — see
[`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md).

## Acting on a message you sent

A user bubble carries three actions (`UserActions` in
`components/chat/chat-message.tsx`): copy, "edit and ask again", "ask again".

**Both re-ask; neither rewrites.** A new turn is appended with the same (or
amended) question, and the original stays where it is. That is a property of the
data, not a UI choice — there is no route that removes a stored message
(`routes/chats.rs` has create, rename, and delete-the-whole-chat, nothing
finer), and the thread *is* the prompt, so editing in place would either lie
about what was sent or silently drop everything after it. Giving these branching
semantics means adding a truncate to the backend and the LLM engine first.

Edit goes through the composer rather than sending directly, via its `draft`
prop — so the question can actually be changed, and so sending still happens on
the one path that owns sending. The prop is keyed by object identity, not by the
string: editing the same message twice has to load it twice.

## Maths

A reply's maths has **no special token**. The model writes it the way markdown
does — `$$…$$` displayed, `$…$` inline — inside whatever `<|par|>` or list item
it was already writing, so it reaches the frontend as ordinary text.
The parser also accepts `\[...\]` for display maths, `\(...\)` inline, and
bare `\boxed{...}` answers. Boxed arguments support nested and escaped braces;
incomplete expressions remain literal text until their closing delimiter arrives.

`lib/math-parser.ts::splitMath` runs as the last step of *both* parsers, over
the finished tree rather than inside either scanner, and replaces the TeX inside
text nodes with `math block` / `math inline` nodes. One pass for both because
`parseMessage` (model markup) and `parseUserMarkdown` (what a person types)
build the same node vocabulary from different syntax, and the dollar rules are
the same on both sides. It does not descend into `code block`, `code inline` or
`tool-call`: a `$` in a shell snippet or in a call's JSON is a dollar sign.

The inline rule is the one that has to be careful, because prose is full of
prices. An opener may not follow a word character or precede whitespace, a
closer may not follow whitespace or precede one, and a body may not contain a
bare `$` or cross a blank line — which is what leaves "$5 and $10" alone and
stops an unclosed `$` from swallowing the rest of a reply while it streams.

`components/chat/math.tsx` typesets with KaTeX, synchronously, so an equation is
in the HTML that first paints and never reflows the paragraph around it after
the fact — which matters most on a surface where the text is still arriving.
Results are cached by TeX source, since a streaming reply re-renders on every
token and its finished equations do not change. TeX that KaTeX rejects falls
back to its own source in the mono face rather than to KaTeX's red error text,
which reads as the app failing rather than the equation being malformed.

Three of KaTeX's own rules are overridden in `globals.css`, and they hang off
the `.rich-math` wrapper rather than being written flat against `.katex` —
both stylesheets are unlayered, so a flat rule would be settled by which
`<link>` the bundler emitted last. The reasoning for each is written on it
there. `/markup` carries both forms in the Mersenne answer.

## The trace

A turn's working — every round before the one that answered, plus what each call
came back with — renders as one panel above the answer
(`components/chat/trace.tsx`), not as the rows it is stored as.
`lib/chat-messages.ts::groupTurns` does the folding: `assistant → tool → assistant`
is the right shape for the *model*, whose next prompt has to contain all three,
and the wrong shape for a reader who asked one question.

**However many rounds it took.** A turn is not capped at one call and one answer
— the backend allows `MAX_TOOL_ROUNDS` of them, and a model that reads a failed
command and tries something else is the normal case, not the exception. Both the
folding and the live stream accumulate: `groupTurns` appends each round's markup
to `working` rather than assigning it once, and the streaming path folds the
answer-so-far back into the working when a later round calls again. Getting
either wrong shows up immediately — as a second reasoning block and a bare call
hanging under the trace, or, in the stored view, as a middle round vanishing.

The panel's rail is its timeline. Reasoning is prose against it, a tool call is a
tick off it (`components/chat/trace-step.tsx`), and the whole thing is
achromatic except the point the pen has reached — a step in flight, whose tick
warms to `--throbber-spark` and reaches out to 1.55x with a light travelling its
length, or, when nothing is running, the rule's own end, which graduates into
that same ink over its last 3.2em and turns right into a foot. That corner is
the nib. The foot yields the moment a call starts, because that call's tick is
then the warm mark and two would be two answers to one question. A turn that
called nothing has no trace and renders exactly as it always did, reasoning
inline.

The shell — rule, header, fold, stagger — is `components/chat/margin-note.tsx`,
and `Trace` and `ReasoningPanel` are its two callers. They were the same panel
written twice, down to a duplicated `LONG_TRACE_WORDS = 70` and a duplicated
paragraph splitter, which is a thing that only gets worse: every movement below
would have had to be authored, and then maintained, in both.

**What moves is in `globals.css` under "The trace, in motion"**, with one curve
throughout — the `word-ink-in` curve — and everything expressed as opacity,
transform or a registered custom property, because a long trace is twenty of
these on screen while the model is still writing. The disclosure is
`grid-template-rows` between `0fr` and `1fr`, which animates a height nobody
measured; it was a bare
`{expanded && …}` before, so opening a note, closing it, and the auto-collapse
when the answer arrives were all instant swaps — and that last one is the moment
the panel is judged on, since it is where the working gets out of the way of the
conclusion. Steps arrive 34ms apart, capped at six so a twenty-step trace still
opens inside a quarter second. The rule's warming is a registered `@property`,
because an unregistered custom property is a string to the animation system and
a string does not interpolate.

The header is one line and not a label with a readout after it: the word, a
leader (the hairline off a table of contents), then the figures, all in the
`tag` face. There is no chevron — a lucide icon is a 2px stroke, which made it
the heaviest mark in a panel whose whole argument is hairlines, and the steps
underneath already say whether the note is open. The word is tensed: `Working…`
while it runs, `Worked` after. It said `Worked…` mid-turn until the label became
two words instead of one.

Results are set two ways, because they are two kinds of text. A search returns
**findings**, which take a hanging index in the `tag` face — a footnote
apparatus. A shell returns **output**, set in the mono face with no index, since
`01` over the lines of `wc -l` would be a claim about what those lines are. They
shared one class until this, so terminal output rendered in the body face.

Reasoning still arriving is held to five lines with a top mask and scrolled so
the newest sits on the bottom edge (`.trace-live-prose`). A turn that thinks for
four hundred words otherwise pushes its own answer off the screen while it
writes, which is the one thing a panel *above* the answer must never do. The cap
lifts when the turn closes. The working also inks in word by word now
(`components/chat/inked.tsx`, shared with `RichContent`) — only the answer had
that, so the model's thinking arrived in hard chunks while its conclusion was
treated, which is the wrong way round.

**`showReasoning` off redacts the working; it does not delete it.** Each prose
step becomes the single word `Thought` — no count, no rule, because a
measurement standing in for withheld text is measuring something the reader was
not shown — and the calls are untouched, since a query and what it returned are
not reasoning. The shape of the work is still the honest account of what the
reply was built from: thought, searched, thought, ran. `Trace` did not consult
the setting at all before this, so with it off the panel printed the whole of
the thinking above an answer while `rich-message.tsx` was hiding the thinking
*inside* one; and an inline `<|thinking|>` rendered as `null`, so on a turn that
called no tool a reader with the setting off lost every sign that the model had
reasoned. Both now redact. The setting is off by default, so this is the path
most readers are on.

Because `stepsOf` merges a `<|thinking|>` block and the `<|par|>` after it into
one prose step — inside the trace the distinction has stopped mattering — a
redacted step covers both. There is no "reasoning" left to separate from
"narration" by that point, and reintroducing the seam only to hide half of it
would put a distinction back into the panel that the panel exists to remove.

`/markup` carries two fixture traces, including the failure path, and its
thinking blocks follow the setting — toggle it there to see both states.

While a call is in flight the thread's status row names it —
`Search: tallest mountain in europe`, `Bash: ls -l /etc` — off the same reading
the live tick uses (`Trace::callInFlight`, the first call with no answer yet).
How a tool is said in English is `lib/tools.ts` and nowhere else: a noun for
the status row, a present and a past verb for the trace, and a fallback to the
tool's own wire name so a new tool reads acceptably before anyone has written
words for it. That table is the frontend's only knowledge of which tools exist
— the vocabulary stopped carrying one marker per tool, so nothing else here can
tell `bash` from `web-search`.

The row reads mark, clock, count, label, ruled between: `◯ 1m 04s | ↓ 4.9k
tokens | Thinking…` — time, then size, then what is being done. The clock
(`components/chat/turn-clock.tsx`) is mounted only while a turn is in flight and
starts from its own mount rather than being handed a start time, which is what
keeps the second hand out of `ChatThread`: a clock owned up there would
re-render every message in the conversation once a second, for the length of the
reply, to move two characters. It measures the wait rather than the model — a
turn queued behind three jobs on the GPU counts from when the question was
asked, because that is the number the person watching is keeping.

A finished turn keeps the clock and drops everything else: the mark goes to
`idle` and the row reads `Worked for 2 minutes 5 seconds`, spelled out, because
it is now a sentence about something finished rather than a readout of something
happening, and the two states sit in the same row a second apart. It coarsens as
it grows — the seconds inside an eleven minute turn are noise. The figure is
`ChatMessage.elapsedSecs`, summed across the turn's rounds and never persisted,
so a conversation reopened from history shows the resting mark alone.

There is
one figure and it is input *and* output — everything the conversation has cost,
the context so far plus what this turn has written into it, which is what
`contextTokensUsed` already is. It only ever climbs. Showing whichever half
matched the phase was tried, and the changeover dropped the number to zero
mid-turn and started again: two honest figures, and the sequence between them a
lie, since nothing about that moment resets anything.

So the arrow carries the phase instead (`components/chat/token-flow.tsx`).
Queuing, a tool call, and the beat before the first token are the model being
*given* something and point in; from the first token on it is writing, and they
point out. Neither the context estimate nor the streamed tally is
server-verified against the tokenizer — this is the composer footer's figure,
not an accounting one. Generating carries no word beside its count — the tokens
arriving *are* the message, already on screen — so the rule is drawn only when
something stands on both sides of it.

Both halves of the row move, and neither is decoration: how fast the count
turns is how fast the tokens are coming. The figure is
`components/ui/rolling-number.tsx`. Past a thousand it abbreviates — `11.3k`,
`2.4M`, `1.1B`, truncated to the tenth rather than rounded, because a counter
reading `12.0k` while the twelve thousandth token is still on its way is
claiming tokens that have not arrived. The exact figure stays in the row's
`title` and in what a screen reader is given. Abbreviating costs the movement
something: the figure then turns on the hundred rather than on the unit, so the
wheels go still between steps — which is the trade past the point where the last
digits of a token count are noise. It is a column of wheels that only ever roll
*upward*, nine to zero included — each column is two stacked runs of 0–9 and
tracks a row rather than a digit, so it climbs into the second run and then
drops back a whole ten with the transition off, between rows carrying the same
glyph. The word is `ThinkingLabel`, and it types itself in a letter at a time
and then walks into the next word in place: frame `i` is
`next.slice(0, i) + previous.slice(i)`, so the two words meet at a boundary that
travels left to right and the row is never blank — `Thinking… → Whinking. →
Woinking.. → … → Working..`. Typing in is that same operation against an empty
previous word, and the dots keep one count across all of it rather than
restarting per word.

A run starts slow and accelerates to its last letter, interpolated
geometrically (these are speeds) and eased at both ends, as is the wheels' own
roll: a movement that leaves at full tilt has arrived before you look at it, and
only its settling is visible. The dots keep their own pace and it is a slow one
— they are not part of the word's movement, which is a thing with a beginning
and an end, where the dots only say that time is passing. Running them at the
speed a word types at was tried, and it made the row twitch.

The wheels have two speeds, because a count changes for two different reasons. A
count of tokens being written climbs a few at a time and the turning *is* the
stream, so it rolls at 220ms. A count that jumps because something was handed
over whole — a message sent, a tool's answer coming back — has already happened
by the time it is drawn, so it lands in 90ms (`swift`); every incoming figure is
one of those. And a wheel mounts showing zero and rolls to its digit on the next
frame rather than arriving already on it, which is what makes the count spin up
when a message is sent, where the row and the number appear at the same instant.
The figure behind an incoming arrow is the last *persisted* context — the
message just typed has not been tokenized by anything yet — so it corrects
itself when the turn is saved.

Reduced motion keeps the rotation and the counting and drops only the
per-character and per-wheel movement — the row still has to say the model is
working.

## Words arriving

A streamed word is **set**: it arrives about 0.055em loose and tightens to its
own tracking over 340ms, under an opacity rise from 0.3
(`word-ink` in `globals.css`, applied by `components/chat/inked.tsx` to both the
answer and the working). It was a 2px blur before, and the blur was the defect —
at 16px it makes a word unreadable for the first fifth of a second, and the word
it does that to is the newest one, which is the one the reader is reaching for.

**Tracking is not a stylistic preference, it is the only property available.**
CSS transforms do not apply to non-replaced inline elements, so every slide,
scale or rise needs `display: inline-block` on each word — and an inline-block
word cannot break, so a long URL or identifier overflows the column instead of
wrapping. That rules out most of the obvious treatments. The trade taken in
exchange is layout: tracking relays the line out on every frame where `filter`
did not, which is affordable only because the paragraph is already being relaid
on every token as text is appended to it.

The keyframes name a `from` and no `to` for the tracking, deliberately.
`letter-spacing` is inherited and the type scale sets it per step
(`--text-base--letter-spacing` is -0.003em), so animating to a literal `0` would
pin every streamed word at zero tracking and then re-track the whole message the
instant streaming stopped. With no 100% entry the browser interpolates to the
element's own underlying value instead.

**The word is an approximation, and knowingly one.** The LLM engine yields
byte-level BPE fragments — `llm-engine/llm/orchestrate/chat.py` says "a byte-level
token can be half a UTF-8 character" — and one socket frame is one token, so a
word is never the unit of arrival.

The unit the animation *is* hung on is a word **together with the whitespace in
front of it** (`\s*\S+`), because that is the shape a token arrives in: a
leading space rides with its word rather than arriving as anything of its own.
Splitting on `/(\s+)/` instead, which is what this did first, took that space
back out and rendered it as a bare text node between two animated spans — twice
the nodes, and a space left sitting un-animated until its word turned up. Pairing
them does *not* narrow the leading edge, and nobody should re-do it expecting
that: measured against a settled line the tracking pushes the edge about 21px
wide at 26px type either way, and carrying the space inside the span tracks the
space too. The delta is the knob, not the split. The animation therefore fires on a word's
first fragment and the rest of the word grows inside a span that is already
moving. Waiting for the word to complete is worse, because it animates type the
reader can already read; animating the true unit would mean threading token
offsets through the parser, since `RichContent` is handed the whole accumulated
string and re-parses it on every token. The treatment is quiet enough that the
approximation does not show, and that quietness is the requirement it is tuned
against — if the line-break flicker at the leading edge ever reads as noise, the
0.055em delta is the knob before the duration is.

`budget` is the composer's thinking-budget choice, one section of
`components/chat/composer-picker.tsx` (the type and its options live in
`lib/think-budget.ts`, shared with `/thinking`'s argument list) — `"low" |
"med" | "high"`, matching the labels the SFT corpus is tagged with, since the
string sent becomes the marker the model was trained under. **"Auto" is the absence of a budget, not a fourth
value**: it is omitted from the frame entirely, so the LLM engine renders no marker
and the model is prompted exactly as it always was. It rides through
`PendingMessage` when a new chat navigates, same as `model` — see
"The composer's own choice" below for both.

`temperature` / `top_k` / `top_p` are the decoding knobs, and unlike `budget`
they **are** settings (`lib/settings.ts`, rendered as sliders in the ⌘K modal) —
how random you want decoding to be is a standing preference, not a property of
one question. Each is omitted individually when unset or non-finite, and the
LLM engine then falls back to its own defaults.

`context_multiplier` is a setting the same way: how much headroom past the
trained context window you want is a standing preference, not a property of one
question. 1 (the default) is the model exactly as trained; the LLM engine builds an
NTK-scaled RoPE table fresh per turn for anything above that, computed from the
model's own `ctx_window` rather than a token count this side would have to keep
in sync with it — see `docs/llm-engine/README.md`.
