# Decision records

A log of significant architecture, design, and implementation decisions:
what was weighed, chosen, and rejected, dated once.

## Index

| Date | Title | Summary |
| --- | --- | --- |
| 2026-08-18 | [Slash Commands & Fuzzy Finder](slash-commands.md) | Centralised interactive command bar directly over chat input with two-stage fuzzy completion |
| 2026-08-19 | [Tool Calls](tool-calls.md) | A marker per tool, the engine detecting and the backend executing, and a bounded second round — because a model this size cannot stop at the call |
| 2026-08-19 | [Dataset content in Postgres](dataset-content-in-postgres.md) | The data engine stores documents as rows, not files — what that cost, and the caps that pay for it |
| 2026-08-19 | [A crate per service](a-crate-per-service.md) | Extracting a generic `service` crate, and why each service needs its own `Proxy` newtype |
| 2026-08-19 | [The engine stopped owning training data](datasets-out-of-the-engine.md) | Corpora moved to the data engine and a run reads them by page through the backend — the one hop that runs toward it |
