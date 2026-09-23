# Voice input uses the browser's recogniser

**2026-08-21**

The microphone in the composer is `SpeechRecognition`, the browser's own Web
Speech API. No model, no service, no key, nothing added to the bundle. It is
absent on Firefox, and on Chrome and Edge it sends audio to Google unless the
on-device pack is already installed.

## What was wanted

Text appearing as it is spoken, in the composer, revising itself as the
recogniser hears more — not a recording that is transcribed after the fact. That
requirement decided this on its own.

## What was weighed

**`SpeechRecognition`**, chosen. It streams: `interimResults` gives a tail that
is rewritten on nearly every event and a `isFinal` flag on the parts that have
settled. The interim/final split is the feature — the composer renders the tail
faded and it firms up as you keep talking, which is the whole reason it reads as
live rather than as a form that fills in late. Free in both senses that matter
here: no bill, and no bytes.

**Whisper in the browser** (`transformers.js`, WASM or WebGPU) was the real
alternative and is the one that fits this repository's premise better — a model
you own, running on your machine, audio that never leaves it. Rejected on the
requirement above: it transcribes in chunks, so the text arrives in blocks after
the fact, and the first word costs a 40–150 MB model download. A dictation
feature that opens with a progress bar is not the feature that was asked for.

**Our own LLM engine** was never a candidate. It is a text model; there is no audio
path into it and building one is a different project.

## The privacy cost, and what is done about it

Chrome and Edge stream the audio to a Google service. This is the one place in
the system where something a person produces leaves the machine to a third party
we did not choose, and it is worth saying plainly rather than burying.

Three things hold the line:

- **It is not on unless it is asked for.** Nothing opens the microphone until
  the button is pressed, and the browser's own recording indicator is lit for
  exactly as long as it is open.
- **On-device is used wherever it exists.** Chrome 138+ can run the recogniser
  locally; `useSpeechRecognition` probes `SpeechRecognition.available({
  processLocally: true })` on mount and sets `processLocally` when the language
  pack is already installed. It never *installs* one — that is a several-hundred
  megabyte download, and starting it because somebody tapped a microphone is not
  a thing to do behind their back.
- **The microphone says which it is.** Its tooltip reads either "recognised on
  this device" or "audio goes to the browser's speech service", and the
  waveform carries the same sentence as screen-reader-only text. Where the audio
  goes is not something a person can see, so it is written down — on the control
  that starts it, which is where someone would think to ask.

If `transformers.js` Whisper ever streams well enough, this is the record to come
back and overturn.

## Why the waveform is a second microphone

`SpeechRecognition` exposes no audio at all — not a level, not a buffer. Drawing
what someone is saying means opening the microphone again, with `getUserMedia`
and an `AnalyserNode`, alongside the capture the recogniser holds. Two
simultaneous captures of one device, which the browsers handle, and which is the
only way to have both the text and the picture of it.

`useAudioLevel` calls back from a `requestAnimationFrame` loop and
`VoiceWaveform` writes the three transforms to the DOM itself. Sixty React
renders a second to move three bars was never worth it.

The bars are the only thing that moves while dictation runs, and that is
deliberate: they move because a voice is moving them. The microphone button
holds still — filled blue with a stop glyph, the same swap the send button makes
while a turn is in flight — because the colour already says the state once, and
a pulse would have said it twice.

## Why the transcript is replaced, not appended

The interim tail is a guess that gets revised, not a chunk that gets added.
Appending each event's text is how you get a paragraph of stuttered half-words.

So the composer holds a range — where the current dictation is writing — and
rewrites the whole of it on every event. The range is made of ProseMirror
positions, which move when the document does, so the editor's `onTransaction`
maps it through anything the person types alongside it; the composer's own
inserts carry a meta flag and are skipped, having already set the range
themselves. That is what makes it safe to keep typing while dictating.
