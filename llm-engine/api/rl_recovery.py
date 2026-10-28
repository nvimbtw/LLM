"""Exclusive parent startup/drained-job retirement of known gym resources, without worker write replay or job-history dependence."""

import shutil
import time
from uuid import UUID

from api import rl_admission, training_admission
from api.rl_effect_log import GymEffectLog
from api.rl_effects import GymResourceOwner
from llm import storage
from llm.architecture.package_integrity import sync_directory

STARTUP_SECONDS = 30.0


def discard_settled(log: GymEffectLog) -> None:
    """Remove only this drained job's fully settled evidence; uncertain worker writes or cleanup attempts remain durable and quota-bound."""
    if (log.pending is not None or len(log.retirements) != len(log.retired)
            or any(not terminal for terminal in log.runs.values())
            or any(not episode.terminal for episode in log.episodes.values())):
        return
    directory = storage.gym_effect_dir(log.authority.job_id)
    if log.path.parent != directory:
        raise ValueError("RL evidence retirement requires its authoritative job directory")
    shutil.rmtree(directory)
    sync_directory(directory.parent)


def reconcile() -> None:
    """After exclusive worker/callback cleanup, bound startup retirement and retain every uncertain write instead of replaying it."""
    root = storage.gym_effects_dir()
    try:
        retained = rl_admission.inventory(root)
    except (OSError, ValueError) as error:
        print(f"recovery: cannot inspect RL gym evidence: {error}")
        return
    deadline = time.monotonic() + STARTUP_SECONDS
    for directory in retained.jobs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("recovery: remaining RL gym evidence retained after startup cleanup allowance")
            return
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                sync_directory(root)
                continue
            path = directory / "ownership.jsonl"
            required = 16 * path.stat(follow_symlinks=False).st_size + 4 * rl_admission.RESPONSE_BYTES
            if training_admission.available_host_bytes() < required:
                raise ValueError("insufficient parent memory for bounded gym evidence reconstruction")
            log = GymEffectLog.open_existing(path, max_bytes=rl_admission.MAX_LOG_BYTES)
            if log.authority.job_id != UUID(directory.name):
                raise ValueError("gym evidence directory differs from its durable queued job identity")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("recovery: remaining RL gym evidence retained after startup cleanup allowance")
                return
            GymResourceOwner(log, response_bytes=rl_admission.RESPONSE_BYTES).retire_resources(seconds=remaining)
            discard_settled(log)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"recovery: cannot retire RL gym evidence {directory.name}: {error}")
