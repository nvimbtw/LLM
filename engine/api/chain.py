"""A run that follows another, against the model that one produces.

The model does not exist when the chain is authored, so a follow-up cannot name
it. It names the run instead, and this fills in the answer when that run lands —
which is why a follow-up waits rather than queueing: a queue position would be a
lie about when it runs.
"""

import uuid

import store
from store import queries as q
from store.schema import RunKind, RunStatus

#: Where a produced model's id belongs in the follow-up's config, by kind.
CONSUMES = {
    RunKind.train: "base_model",
    RunKind.benchmark: "model_name",
}


def promote(run_id: uuid.UUID, produced: uuid.UUID | None) -> int:
    """Release everything waiting on a finished run.

    A run that produced a model hands it to its dependents; one that failed or
    was cancelled fails them, because the model they were written against is
    never going to exist.
    """
    with store.session() as db:
        waiting = q.waiting_on(db, run_id)
        for run in waiting:
            field = CONSUMES.get(run.kind)
            if produced is None or field is None:
                run.status = RunStatus.failed
                run.error = f"the run it follows produced no model ({run_id})"
                continue
            # SQLAlchemy tracks the attribute, not the dict, so a mutated
            # config is not a change it would write back.
            run.config = {**(run.config or {}), field: str(produced)}
            run.status = RunStatus.queued
        return len(waiting)
