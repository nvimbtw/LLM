# Decision records

A log of significant architecture, design, and implementation decisions:
what was weighed, chosen, and rejected, dated once.

## Index

| Date | Title | Summary |
| --- | --- | --- |
| 2026-08-18 | [Slash Commands & Fuzzy Finder](slash-commands.md) | Centralised interactive command bar directly over chat input with two-stage fuzzy completion |
| 2026-08-19 | [Tool Calls](tool-calls.md) | The engine detecting and the backend executing, and a bounded second round — because a model this size cannot stop at the call. Its marker-per-tool decision is superseded below |
| 2026-08-19 | [Dataset content in Postgres](dataset-content-in-postgres.md) | The data engine stores documents as rows, not files — what that cost, and the caps that pay for it |
| 2026-08-19 | [A crate per service](a-crate-per-service.md) | Extracting a generic `service` crate, and why each service needs its own `Proxy` newtype |
| 2026-08-19 | [The engine stopped owning training data](datasets-out-of-the-engine.md) | Corpora moved to the data engine and a run reads them by page through the backend — the one hop that runs toward it |
| 2026-08-20 | [Attachments are stored, not read](attachments-are-stored-not-read.md) | Uploads live beside the messages they hang off, in the engine's database — and reach no prompt, enforced rather than assumed |
| 2026-08-21 | [Voice input uses the browser's recogniser](voice-input-uses-the-browser.md) | Dictation is `SpeechRecognition`, not a model of ours — what streaming bought, what it costs in privacy, and what is done about that |
| 2026-08-22 | [One marker for every tool](generic-tool-calls.md) | `<|tool-call|>` carrying JSON replaces a marker pair per tool, so the tool table stops being part of the vocabulary — and what the grammar gives up for it |
