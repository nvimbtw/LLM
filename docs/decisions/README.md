# Decision records

A log of significant architecture, design, and implementation decisions:
what was weighed, chosen, and rejected, dated once.

## Index

| Date | Title | Summary |
| --- | --- | --- |
| 2026-08-18 | [Slash Commands & Fuzzy Finder](slash-commands.md) | Centralised interactive command bar directly over chat input with two-stage fuzzy completion |
| 2026-08-19 | [Tool Calls](tool-calls.md) | The LLM engine detecting and the backend executing, and a bounded second round — because a model this size cannot stop at the call. Its marker-per-tool decision is superseded below |
| 2026-08-19 | [Dataset content in Postgres](dataset-content-in-postgres.md) | The data engine stores documents as rows, not files — what that cost, and the caps that pay for it |
| 2026-08-19 | [A crate per service](a-crate-per-service.md) | Extracting a generic `service` crate, and why each service needs its own `Proxy` newtype |
| 2026-08-19 | [The LLM engine stopped owning training data](datasets-out-of-the-llm-engine.md) | Corpora moved to the data engine and a run reads them by page through the backend — the one hop that runs toward it |
| 2026-08-20 | [Attachments are stored, not read](attachments-are-stored-not-read.md) | Uploads live beside the messages they hang off, in the LLM engine's database — and reach no prompt, enforced rather than assumed |
| 2026-08-21 | [Voice input uses the browser's recogniser](voice-input-uses-the-browser.md) | Dictation is `SpeechRecognition`, not a model of ours — what streaming bought, what it costs in privacy, and what is done about that |
| 2026-08-22 | [One marker for every tool](generic-tool-calls.md) | `<|tool-call|>` carrying JSON replaces a marker pair per tool, so the tool table stops being part of the vocabulary — and what the grammar gives up for it |
| 2026-08-24 | [One command to start everything](one-command-development.md) | `docker compose up` replaces four manual starts — why several containers rather than one, and the three implicit facts it forced into the open |
| 2026-08-25 | [The GPU's memory is handed back](gpu-memory-is-handed-back.md) | The pool is released at run boundaries and diagnostics record off the training path — why a run's ceiling was set by the step that measures it |
| 2026-08-29 | [A checkpoint is a model](checkpoints-are-models.md) | A run's intermediates get a uuid, a directory and a row like any other model, so chat, benchmarking and SFT need no second path — and what the disk pays for it |
| 2026-08-29 | [A gym episode is driven by the backend](gyms-are-driven-by-the-backend.md) | The rollout loop lives beside the tool renderer rather than beside the sampler, so a gym shows the model exactly what serving does — and what that costs the RL driver that comes later |
| 2026-08-30 | [Reinforcement learning runs in the LLM engine](reinforcement-learning-in-the-llm-engine.md) | The trainer drives its own rollouts and the backend serves each move already rendered — GRPO on group-relative returns, a shaping clip that cannot outrank an answer, and the measured pass rate that blocks the whole thing until the model learns to stop and answer |
| 2026-09-08 | [Activations are recomputed](activations-are-recomputed.md) | Three quarters of a step's VRAM was intermediates held for the backward pass; recomputing each block trades 1.3x the time for 3x the memory at identical gradients — and moves the card's ceiling from ~50M parameters to ~300M, where the optimizer state takes over |
| 2026-09-08 | [A run plans its own memory](a-run-plans-its-own-memory.md) | The recompute flag shipped unreachable from the UI, and this card answers an oversized run out of host RAM instead of refusing it — so a run neither failed nor progressed. Depth becomes a count planned against free VRAM, the allocator is capped so overflow is loud, the first step always reports, and a diagnosed step stops holding every layer's attention matrix |
