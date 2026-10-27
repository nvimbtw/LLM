"""Supervised validation queue handler; only the child imports/constructs administrator source."""

from api.worker import SupervisedRun
from llm.architecture.validation_artifacts import publish_after_cleanup
from llm.architecture.validation_records import VALIDATION_PHASES, ValidationOutput, ValidationSnapshot
from llm.architecture.wire import StageOutcome, ValidationEvent
from llm.runtime.protocol import Inspect, Progress, RuntimeDeadlines, RuntimePolicy, RuntimeResources, Validate
from llm.runtime.supervisor import RequestCancelled
from llm.train import memory


def validation_policy(snapshot: ValidationSnapshot) -> RuntimePolicy:
    """Bind finite parent-owned CPU/CUDA admission, compiler lifetime and artifact limits for this snapshot."""
    device = snapshot.request.profile.device if snapshot.request.profile is not None else "cpu"
    allocator = memory.budget_bytes(device) if device == "cuda" else 0
    if device == "cuda" and allocator <= 0:
        raise ValueError("validation CUDA admission is unavailable")
    return RuntimePolicy(deadlines=RuntimeDeadlines(validation_seconds=180.0),resources=RuntimeResources(
        process_address_bytes=1 << 40 if device == "cuda" else 16 << 30,
        tree_resident_bytes=8 << 30 if device == "cuda" else 3 << 30,
        tree_memory_enforcement="preflight_and_monitor",device=device,device_allocator_bytes=allocator,
        owned_processes=2,artifact_bytes=8 << 20,total_artifact_bytes=16 << 20))


class ValidationHandler:
    """Run exact immutable snapshots and publish only after guardian-verified process-tree cleanup."""

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Return a bounded report after worker stop/publication, leaving terminal/evidence commit to Runner."""
        snapshot = ValidationSnapshot.model_validate(context.config)
        if snapshot.run_id != context.job_id:
            raise ValueError("queued validation snapshot belongs to another job")
        inspection = snapshot.request.mode == "inspect"
        sequence = 0

        def report(event) -> None:
            """Translate bounded phase progress without exposing source bytes or claiming completed validation."""
            nonlocal sequence
            if isinstance(event,Progress) and event.unit == "phases" and event.completed < len(VALIDATION_PHASES):
                context.progress(ValidationEvent(schema_version=1,run_id=context.job_id,sequence=sequence,
                    type="validation_stage",stage=StageOutcome(stage=VALIDATION_PHASES[event.completed],outcome="running")))
                sequence += 1

        with context.workers.session(None if inspection else context.lease,validation_policy(snapshot),
                                     "llm.architecture.worker:ValidationBackend",metadata_only=inspection) as session:
            artifact = session.artifacts.write_bytes(snapshot.model_dump_json().encode(),
                "source_snapshot_v1" if inspection else "validation_snapshot_v1")
            command = Inspect(type="inspect",snapshot=artifact) if inspection else Validate(type="validate",snapshot=artifact)
            completed = session.run(command,request_id=context.job_id,on_event=report,cancel=context.cancel)
            if completed.outcome != "ok" or completed.result is None:
                raise RequestCancelled("validation worker did not complete")
            output = ValidationOutput.model_validate_json(session.artifacts.read_bytes(completed.result))
        if context.cancel.is_set():
            raise RequestCancelled("validation was cancelled before publication")
        context.progress(ValidationEvent(schema_version=1,run_id=context.job_id,sequence=sequence,
            type="validation_stage",stage=StageOutcome(stage="publication",outcome="running")))
        publish_after_cleanup(snapshot,output)
        return output.report.model_dump(mode="json")
