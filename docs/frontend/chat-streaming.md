# Chat streaming

## Stream protocol (chat)

One socket per chat, opened at `/ws/chats/{id}`:

```
client → { token }                            (auth frame)
server → { type: "ready" }
client → { model_name, text, attachment_ids?, system?, budget?,
           temperature?, top_k?, top_p?, context_multiplier? }
server → { type: "token", token }             (repeated)
server → { type: "tool_call", tool, argument } (the reply asked for something)
server → { type: "done", message_id, content, tokens, usage, final }
server → { type: "tool_result", tool, argument, body }
client → { type: "interject", text }          (any time after the opening ask)
server → { type: "interjected", text }        (spliced into the next round)
server → { type: "error", error }
```

The chat exists before the socket opens — its id is in the URL — so there is no
"which conversation is this" frame and no id to wait for coming back. The reply
is written when the last token lands, so closing the tab loses it rather than
storing half of one.

**A `token` frame is a delta, not a word — concatenate it, never join it.** The
LLM engine yields `after[len(before):]`, the difference between decoding the reply
so far and decoding it one id longer
(`llm-engine/llm/orchestrate/chat.py`), so the pieces already carry whatever
whitespace belongs to them and their concatenation is *byte-identical* to the
`content` on `done`, which is one `decode` over the whole sequence. `use-chat.ts`
joined them with `" "` instead, on the belief that a token was a whitespace-split
word. It is not: the tokenizer is byte-level BPE and a piece can be half a UTF-8
character, let alone half a word. The symptom was a reply that streamed as
`b ust y b oot s` and `I ' ve` and then silently became `busty boots` and `I've`
the moment the stored message replaced it — the two stages disagreeing about the
same reply. Nothing downstream can repair that, because by then the extra spaces
are indistinguishable from real ones.

`joinWorking` is a different boundary and keeps its space: it joins one *round's*
markup to the next, across a closed block, and both the streaming path and
`groupTurns` on reload use it — so the live trace and the stored one agree.

**A turn that calls a tool runs more than one round**, so `token` and `done` can
each arrive twice — the reply that called, then the reply that answers with what
came back. `final` on a `done` is the one that ends the turn; the client closes
the socket on that and not on the first, which would cut the turn off before the
answer it went to fetch. Absent is treated as final.

The two tool frames bracket the wait: `tool_call` the moment the marker closes
(so a search can be shown as started while the rest of the reply is still
streaming), `tool_result` once the backend has run it. `body` is in the marker
vocabulary — a `<|list|>` of findings, or a `<|warn|>` if the tool failed.

A fast model streams straight through at whatever rate it generates —
`ChatSession` applies each decoded token event to the visible turn, no artificial pacing. What has
to keep up with that is the thread's own scroll, next.

**The composer stays live while a turn streams, and `interject` is what typing
into it then does.** `lib/chat-session.ts::ChatSession.send` checks whether a turn is
already in flight; if so, the message goes into a small local `queued` list
(rendered by `components/chat/queued-message.tsx`, deliberately not a bubble —
muted text and a hairline rule rather than a fill, so it reads as *not sent
yet*) and out over the open socket as `interject`, rather than opening a second
one. `interject` has no `attachment_ids`: see the "Interjecting mid-turn"
section of `docs/backend/README.md` for why the backend keeps that contract
text-only.

The backend can only splice an interjection in between a round's tool result
and the next round's request — see the same section — so it is not
instant. The `interjected` confirmation is the frontend's cue that it landed:
`onInterjected` moves the message out of `queued` and into the turn's own
trace, following the same folding contract as `groupTurns` —
a `user` row arriving while the previous assistant turn is still mid-flight
(`toolAnswers` set, `content` still empty) can only be an interjection, the
same test the `tool` row branch already uses to know a round hasn't finished.
**A message still `queued` when the turn ends is sent as an ordinary new turn**
rather than lost — the turn closing with no further rounds means the backend
never got the chance to fold it in, not that it should be discarded. All three
endings drain it: `done`, an error, and Stop, which settles the socket by hand
and so has to put the turn's state back itself. The drain belongs to `ChatSession`,
which settles its active-turn identity before
starting the next queued message. It does not wait for React to rerender, so a
just-ended turn cannot accidentally re-queue the message being drained.

## Following the bottom

`chat-thread.tsx` keeps the thread pinned to the bottom while a reply streams
in, without fighting a reader who scrolls up mid-turn to check something
earlier. Two refs carry the state a re-render shouldn't own:

- **`stickRef`** — whether to follow at all. Set by the user's own wheel/touch
  gestures (`trackStickIntent`), not by where the scroll happens to sit: it
  reads live `scrollTop` only on a genuine gesture, so a programmatic scroll
  this component just made never reads as the user asking to unstick.
  Re-engages the moment a gesture lands within `STICK_TO_BOTTOM_THRESHOLD_PX`
  of the bottom — scrolling back down by hand resumes following, no separate
  "jump to latest" affordance needed.
- **`lastHeightRef`** — the last measured `scrollHeight`, so the follow effect
  only fires on genuine growth (a new line) and not on every token mutating
  text within the line that's already on screen.

**The scroll itself is always instant (`behavior: "auto"`), never
`"smooth"`.** It used to glide during streaming, and a fast model breaks
exactly that: a line wrapping is a new `scrollTo` call, a fast enough model
wraps a line every render, and each call retargets a `smooth` animation that
hadn't finished chasing the *previous* target. The visible position ends up
permanently trailing the true bottom rather than catching up — which is what
"the text disappears behind the composer" actually was: not a rendering
slowdown, a scroll animation that can't keep pace with how often it's being
retargeted. Snapping on every growth event keeps the view exactly pinned
regardless of arrival rate; the growth events are themselves small (one
wrapped line at a time), which is what keeps this reading as a glide rather
than a jump even though nothing is animating.
