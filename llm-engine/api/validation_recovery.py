"""Startup adoption of complete post-cleanup validation publications before orphan jobs are failed."""

from sqlalchemy import select

import store
from api import builtin_catalog
from api.validation_jobs import complete_validation
from llm.architecture.validation_artifacts import read_publication, remove_pending
from store import queries
from store.schema import Run, RunKind, RunStatus


def reconcile() -> None:
    """Reconcile at most 100 interrupted validation jobs after WorkerManager has verified old process cleanup.

    Only a complete parent publication can finish a formerly running job.
    Cancellation and missing/corrupt outputs remain non-successful and are handled
    by normal orphan recovery. Unpublished staging never authorizes evidence.
    """
    remove_pending()
    with store.session() as db:
        ids = list(db.scalars(select(Run.id).where(Run.kind == RunKind.model_validation,
            Run.status == RunStatus.running).order_by(Run.created_at).limit(100)))
    for run_id in ids:
        try:
            publication = read_publication(run_id)
        except FileNotFoundError:
            continue
        except (OSError,ValueError) as error:
            print(f"validation recovery: {run_id}: {type(error).__name__}")
            continue
        try:
            with store.session() as db:
                run = queries.lock_run(db,run_id)
                if run is None or run.status != RunStatus.running:
                    continue
                result = publication.output.report.model_dump(mode="json")
                queries.finish_run(db,run_id,RunStatus.done,result=result,progress=1.0)
                complete_validation(db,run,result)
                builtin_catalog.queue_ready_recipes(db)
        except ValueError as error:
            print(f"validation recovery: {run_id}: {type(error).__name__}")
