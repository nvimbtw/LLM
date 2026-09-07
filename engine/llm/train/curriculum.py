"""Mastery gates and sampling, independent of generation and the gym's answers."""

from dataclasses import asdict, dataclass

VERSION = 1
REVIEW_WEIGHT = 0.3
PRACTICE_WEIGHT = 0.6
STRETCH_WEIGHT = 0.1
NEEDS_SFT_BELOW = 0.05


@dataclass
class Mastery:
    mastered: bool = False
    passes: int = 0
    failures: int = 0
    low_checks: int = 0
    accuracy: float | None = None
    samples: int = 0
    checks: int = 0
    invalid_answers: int = 0
    truncated: int = 0


class Lessons:
    def __init__(self, tasks: list[dict], schedule, saved: dict | None = None):
        self.tasks = {t["id"]: t for t in tasks}
        self.schedule = schedule
        for task in tasks:
            missing = set(task.get("prerequisites", [])) - self.tasks.keys()
            if missing:
                raise ValueError(f"{task['id']} needs selected prerequisites: {', '.join(sorted(missing))}")
        self.mastery = {key: Mastery() for key in self.tasks}
        self.cycle = 0
        self.training_groups = 0
        self.events: list[dict] = []
        if saved:
            if saved.get("version") != VERSION or saved.get("tasks") != tasks:
                raise ValueError("saved curriculum differs from the selected task catalogue")
            if saved.get("schedule") != schedule.model_dump():
                raise ValueError("saved curriculum uses different mastery settings")
            self.mastery = {key: Mastery(**saved["mastery"][key]) for key in self.tasks}
            self.cycle = saved["cycle"]
            self.training_groups = saved.get("training_groups", 0)
            self.events = saved.get("events", [])

    def eligible(self, key):
        return all(self.mastery[p].mastered for p in self.tasks[key].get("prerequisites", []))

    def status(self, key):
        value = self.mastery[key]
        if not self.eligible(key):
            return "locked"
        if value.mastered:
            return "mastered"
        if value.low_checks >= self.schedule.consecutive_checks:
            return "needs_sft"
        return "practice"

    def checking(self):
        return [key for key in self.tasks if self.eligible(key)]

    def observe(self, key, correct, samples, step, invalid_answers=0, truncated=0):
        if samples != self.schedule.validation_samples:
            raise ValueError("an incomplete validation must not change mastery")
        value = self.mastery[key]
        value.accuracy, value.samples = correct / samples, samples
        value.invalid_answers, value.truncated = invalid_answers, truncated
        value.checks += 1
        if not self.eligible(key):
            # A prerequisite may have regressed earlier in this same exam.
            # Keep the measurement, but require fresh checks after it unlocks.
            value.passes = value.failures = value.low_checks = 0
            return
        value.passes = value.passes + 1 if value.accuracy >= self.schedule.promote_at else 0
        value.failures = value.failures + 1 if value.accuracy < self.schedule.retain_at else 0
        value.low_checks = value.low_checks + 1 if value.accuracy < NEEDS_SFT_BELOW else 0
        before = value.mastered
        if value.passes >= self.schedule.consecutive_checks and self.eligible(key):
            value.mastered = True
        if value.failures >= self.schedule.consecutive_checks:
            value.mastered = False
        if before != value.mastered:
            self.events.append({"task_type": key, "step": step,
                                "action": "promoted" if value.mastered else "review",
                                "correct": correct, "samples": samples})
        # Losing a prerequisite also invalidates dependent mastery, transitively.
        for _ in range(len(self.tasks)):
            for task_id, state in self.mastery.items():
                if state.mastered and not self.eligible(task_id):
                    state.mastered, state.passes = False, 0

    def draw(self, rng, count):
        review = [key for key in self.tasks if self.status(key) == "mastered"]
        practice = [key for key in self.tasks if self.status(key) in ("practice", "needs_sft")]
        stretch = [key for key, task in self.tasks.items() if self.status(key) == "locked"
                   and any(p in practice for p in task.get("prerequisites", []))
                   and all(self.mastery[p].mastered or p in practice for p in task.get("prerequisites", []))]
        choices, weights = [], []
        for pool, weight in ((practice, PRACTICE_WEIGHT), (review, REVIEW_WEIGHT), (stretch, STRETCH_WEIGHT)):
            choices.extend(pool)
            weights.extend([weight / len(pool)] * len(pool) if pool else [])
        if not choices:
            raise ValueError("curriculum has no reachable tasks")
        return rng.choices(choices, weights=weights, k=count)

    def snapshot(self):
        return {"version": VERSION, "tasks": list(self.tasks.values()),
                "schedule": self.schedule.model_dump(), "cycle": self.cycle,
                "training_groups": self.training_groups,
                "mastery": {key: asdict(value) for key, value in self.mastery.items()},
                "events": list(self.events)}

    def report(self):
        return [{"task_type": key, "label": task["label"], "family": task["family"],
                 "level": task["level"], "status": self.status(key), **asdict(self.mastery[key])}
                for key, task in self.tasks.items()]
