"""Pure selection of profile-specific recomputation estimates by actual execution identities."""

from dataclasses import asdict, dataclass, replace

from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence
from llm.train.model_memory import ModelMemoryPlan
from llm.train.recompute_capture import ReplayStorage


@dataclass(frozen=True)
class RecomputeEstimate:
    """A bounded preflight estimate for one explicit occurrence policy and exact runtime profile digest."""

    profile_digest: str
    occurrences: tuple[ExecutionOccurrence, ...]
    estimated_bytes: int

    def __post_init__(self) -> None:
        """Refuse ambiguous duplicate identities and ineffective byte/profile bounds."""
        if (len(self.profile_digest) != 64 or any(value not in "0123456789abcdef" for value in self.profile_digest)
                or len(self.occurrences) > 128 or len(set(self.occurrences)) != len(self.occurrences)
                or type(self.estimated_bytes) is not int or self.estimated_bytes < 1):
            raise ValueError("invalid recomputation estimate")


def plan_recomputation(
    eligible: tuple[ExecutionOccurrence, ...], estimates: tuple[RecomputeEstimate, ...], *,
    profile_digest: str, target_bytes: int,
) -> RecomputeEstimate:
    """Choose the least replay work that fits from bounded exact-profile preflight candidates.

    Candidates must come from supported admitted estimation, never from a guessed
    width/layers discount. Missing/unknown/stale estimates fail instead of claiming
    that recomputation can make an unsupported model fit. This function has no
    model, storage, allocation or execution effects.
    """
    if type(target_bytes) is not int or target_bytes < 0 or not 1 <= len(estimates) <= 129:
        raise ValueError("invalid recomputation planning budget")
    known = set(eligible)
    policies = set()
    for estimate in estimates:
        policy = frozenset(estimate.occurrences)
        if estimate.profile_digest != profile_digest or policy in policies or policy - known:
            raise ArchitectureError("recompute_profile", "unknown, duplicate or stale recomputation profile estimate")
        policies.add(policy)
    available = [estimate for estimate in estimates if estimate.estimated_bytes <= target_bytes]
    if not available:
        raise ArchitectureError("memory_budget", "no supported recomputation estimate fits the planning target")
    return min(available,key=lambda estimate:(len(estimate.occurrences),estimate.estimated_bytes))


@dataclass(frozen=True)
class TrainingRecomputation:
    """Retained replay policy and conservative candidates for one exact captured training profile."""

    selected: RecomputeEstimate
    candidates: tuple[RecomputeEstimate, ...]
    estimator: str = "full-retention-replay-reserve-v1"
    storage: ReplayStorage | None = None

    def as_dict(self) -> dict:
        """Serialize bounded occurrence identities and estimator provenance for immutable package metadata."""
        return {"schema_version": 1, "estimator": self.estimator,
                "selected": asdict(self.selected), "candidates": [asdict(value) for value in self.candidates],
                **({"storage": asdict(self.storage)} if self.storage is not None else {})}


def requested_count(eligible: tuple[ExecutionOccurrence, ...], requested: bool | int | None) -> int | None:
    """Validate the bounded actual target set and distinguish automatic selection from an explicit count."""
    if len(eligible) > 128 or len(set(eligible)) != len(eligible):
        raise ArchitectureError("recompute_occurrence", "training supports at most 128 unique replay occurrences")
    if requested is None:
        return None
    if requested is False:
        return 0
    if requested is True:
        return len(eligible)
    if type(requested) is int and 0 <= requested <= len(eligible):
        return requested
    raise ArchitectureError("recompute_occurrence", "requested replay count exceeds actual execution occurrences")


def captured_recomputation(
    eligible: tuple[ExecutionOccurrence, ...], memory: ModelMemoryPlan, storage: ReplayStorage, *,
    profile_digest: str, requested: bool | int | None, retained_policy: dict | None = None,
) -> tuple[ModelMemoryPlan, TrainingRecomputation]:
    """Choose the fewest actual prefix occurrences whose captured storage lifetime bound fits admission.

    Candidates retain output backing storage, unselected work and one complete
    replay region. Unknown/stale partitions fail instead of applying discounts.
    An explicit request wins only when its own supported estimate fits. Exact
    recovery can retain a byte-identical previously selected candidate under a
    changed allocator ceiling; its definition/profile/estimate must still match.
    """
    count = requested_count(eligible, requested)
    if tuple(region.occurrence for region in storage.regions) != eligible or storage.captured_training_bytes != memory.activation_bytes:
        raise ArchitectureError("recompute_capture", "storage partition differs from the exact captured model estimate")
    plans = tuple(replace(memory, activation_bytes=storage.activation_bytes(eligible[:number]))
                  for number in range(len(eligible) + 1))
    candidates = tuple(RecomputeEstimate(profile_digest, eligible[:number], plan.estimated_bytes)
                       for number, plan in enumerate(plans))
    allowed = candidates if count is None else (candidates[count],)
    if retained_policy is not None:
        allowed = tuple(value for value in allowed if asdict(value) == retained_policy)
        if not allowed:
            raise ArchitectureError("recompute_recovery", "saved replay selection differs from captured profile/storage evidence")
    selected = plan_recomputation(eligible, allowed,
                                  profile_digest=profile_digest, target_bytes=memory.target_bytes)
    return plans[len(selected.occurrences)], TrainingRecomputation(selected, candidates,
        estimator="captured-occurrence-lifetimes-v1", storage=storage)


def conservative_recomputation(
    eligible: tuple[ExecutionOccurrence, ...], memory: ModelMemoryPlan, *,
    profile_digest: str, requested: bool | int | None,
) -> tuple[ModelMemoryPlan, TrainingRecomputation]:
    """Admit explicit replay without claiming a reduction unsupported by checkpoint capture.

    Nonempty policies reserve an additional complete captured activation/workspace
    bound for replay alongside the ordinary full-retention bound. Automatic mode
    therefore chooses zero replay. Both bounds must fit before real allocation;
    the worker separately verifies actual replay outputs, gradients and RNG.
    """
    count = requested_count(eligible, requested) or 0
    plain = RecomputeEstimate(profile_digest, (), memory.estimated_bytes)
    replay_memory = replace(memory, activation_bytes=2 * memory.activation_bytes)
    replay = RecomputeEstimate(profile_digest, eligible[:count], replay_memory.estimated_bytes) if count else None
    candidates = (plain, replay) if replay is not None else (plain,)
    selected = plan_recomputation(eligible, (replay,) if replay is not None else candidates,
                                  profile_digest=profile_digest, target_bytes=memory.target_bytes)
    return (replay_memory if replay is not None else memory), TrainingRecomputation(selected, candidates)
