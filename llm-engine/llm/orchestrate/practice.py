"""Evaluate lessons without feeding validation or test tokens to the optimizer."""

from llm.orchestrate import train
from llm.train import policy, rollout
from llm.train.curriculum import Lessons

VALIDATION_SEED_STRIDE = 10_000


def prepare(cfg, client, saved=None, *, inherit_base=True):
    """Build mastery from selected public tasks and explicit saved state; legacy callers may read base metadata locally."""
    if cfg.curriculum is None:
        return None
    catalogue = next((gym for gym in client.catalogue() if gym["id"] == cfg.gym_id), None)
    if catalogue is None:
        raise ValueError(f"unknown gym {cfg.gym_id}")
    selected = {task.id for task in cfg.task_types}
    tasks = [{**task, "generator_version": catalogue.get("generator_version")}
             for task in catalogue["task_types"] if task["id"] in selected]
    if len(tasks) != len(selected):
        raise ValueError("curriculum contains unknown task types")
    if saved is None and inherit_base:
        saved = train.run_record(cfg.base_model).get("curriculum")
    return Lessons(tasks, cfg.curriculum, saved)


def measure(cfg, client, model, state, task_type, split, samples, seed, cancel):
    """Micro-batch generation as well as training: a 100-question exam must fit."""
    if cancel is not None and cancel.is_set():
        return None
    run_id, episodes = client.create_run(
        gym_id=cfg.gym_id, model_name=cfg.base_model, task_types=[task_type],
        count=samples, seed_base=seed, group_size=1, split=split,
    )
    correct = invalid = truncated = 0
    sampling = cfg.model_copy(update={"temperature": 0.0, "top_k": 0, "top_p": 1.0})
    try:
        for start in range(0, len(episodes), cfg.micro_batch):
            if cancel is not None and cancel.is_set():
                for episode in episodes[start:]:
                    client.abandon(episode.id)
                client.set_status(run_id, "abandoned")
                return None
            batch = rollout.run(model, state.tokenizer, client, episodes[start:start + cfg.micro_batch],
                                sampling, window=state.window, rope=state.rope, max_rounds=1)
            if len(batch) != len(episodes[start:start + cfg.micro_batch]) or any(not policy.counts(roll) for roll in batch):
                raise ValueError("validation transport failed; mastery was not updated")
            correct += sum(roll.score >= 1 for roll in batch)
            invalid += sum(not roll.answer for roll in batch)
            truncated += sum(roll.truncated for roll in batch)
        if len(episodes) != samples:
            raise ValueError("gym returned an incomplete exam")
        client.set_status(run_id, "done")
    except Exception:
        client.set_status(run_id, "failed")
        raise
    return {"task_type": task_type, "correct": correct, "samples": samples,
            "accuracy": correct / samples, "invalid_answers": invalid,
            "truncated": truncated, "run_id": run_id}


def validate(cfg, client, model, state, lessons, step, cancel, notify=None):
    measured = []
    # Decide the exam before observing any results, so promotion cannot change it.
    for key in lessons.checking():
        if notify:
            notify(key)
        result = measure(cfg, client, model, state, key, "validation",
                         cfg.curriculum.validation_samples,
                         lessons.cycle * VALIDATION_SEED_STRIDE, cancel)
        if result is None:
            return False
        measured.append(result)
    for result in measured:
        lessons.observe(result["task_type"], result["correct"], result["samples"], step,
                        result["invalid_answers"], result["truncated"])
    lessons.cycle += 1
    return True


def benchmark(cfg, client, model, state, lessons, cancel, notify=None):
    results = []
    for key in lessons.tasks:
        if notify:
            notify(key)
        result = measure(cfg, client, model, state, key, "test",
                         cfg.curriculum.test_samples, cfg.holdout_base, cancel)
        if result is None:
            return None
        results.append(result)
    return {"accuracy": sum(row["correct"] for row in results) / sum(row["samples"] for row in results),
            "tasks": results, "seed_base": cfg.holdout_base,
            "samples_per_task": cfg.curriculum.test_samples}
