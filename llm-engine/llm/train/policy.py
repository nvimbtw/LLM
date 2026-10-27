"""What a batch of rollouts is worth, and the gradient that follows from it.

GRPO, and the two choices that shape everything here are both about the
*baseline*. A critic would be a second network the size of the policy, on one
device, predicting a binary outcome — so the baseline is a group instead: G
rollouts of one seed give G returns, and the advantage is how much better a
rollout did than its siblings on the identical task. That the gym's own
`group_size` already hands us the groups is why this is a few dozen lines rather
than a second model.

The batch uses the current weights once, with no old-policy ratio, clipping or
stored old log-probabilities. Recomputed unit-temperature log-probabilities are
unfiltered, while sampling uses temperature, truncation and grammar masks. This
retains the existing surrogate objective rather than claiming an exact unbiased
gradient of the transformed sampling distribution. `inner_epochs > 1` is
the first thing to want later and the change it needs is written down in
`docs/decisions/reinforcement-learning-in-the-llm-engine.md`.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
from torch import nn

from llm.model.language import evaluating, logits
from llm.model.transformer import Transformer
from llm.train.rollout import Rollout, padded

#: How much the dense shaping may move an episode's return, in total.
#:
#: The invariant is `2 * CLIP < 1`: **no incorrect episode can outrank a correct
#: one in its own group**, whatever it did with the shell. That turns "the step
#: reward is kept deliberately small" from an argument about magnitudes into a
#: guarantee that does not depend on `gyms-engine/gyms/bash/rewards.py` continuing to be
#: modest. What the shaping can still do is break ties among rollouts that all
#: failed and among rollouts that all succeeded, which is precisely the two
#: places a binary reward is silent.
CLIP = 0.2
assert 2 * CLIP < 1.0, "shaping could outrank an answer"

#: Guards the division when a group's returns are near-identical. Not a
#: tolerance: a group whose spread is genuinely zero is dropped before this is
#: reached, and this only stops a group that differs in the tenth decimal from
#: producing an advantage of ten thousand.
EPSILON = 1e-4

#: Where the k3 log-ratio is cut before it is exponentiated. A safety rail and
#: nothing more: the policy starts *as* the reference, so the ratio is exactly 0
#: at step 0 and only grows as the run moves — but `exp` of an unbounded ratio is
#: one degenerate rollout away from an infinite penalty and a NaN model.
MAX_LOG_RATIO = 10.0


@dataclass
class Batch:
    """One optimizer step's worth of rollouts, and what they came to."""

    advantages: dict[str, float] = field(default_factory=dict)
    #: Groups that contributed no gradient. All-failed and all-passed alike —
    #: this is the number that says whether a run is learning or merely
    #: spending, which is why it is a frame and not a log line.
    groups: int = 0
    skipped: int = 0
    pass_rate: float = 0.0
    mean_return: float = 0.0

    @property
    def signal(self) -> float:
        return 0.0 if not self.groups else 1.0 - self.skipped / self.groups


def episode_return(roll: Rollout) -> float:
    """`score + clip(sum of step rewards, -CLIP, +CLIP)`."""
    return roll.score + max(-CLIP, min(CLIP, roll.shaping))


def counts(roll: Rollout) -> bool:
    """Whether a rollout is evidence about the model.

    An episode this side abandoned is not: it measures a transport fault or a
    wedged sandbox. It is absent from a batch the same way a closed episode is
    absent from a run's mean, rather than being a zero that blames the model for
    it. An episode that merely ran out of rounds or window *is* evidence — it was
    graded on what it last said, and that zero is earned.
    """
    return roll.stopped not in ("unreachable", "ungraded")


def score(rollouts: list[Rollout]) -> Batch:
    """Group-relative advantages, keyed by episode id.

    Grouped on `Rollout.group` rather than on the index, and keyed by the id
    rather than the index, because a batch may pool several gym runs — one per
    gym, when a run spans them — and every one of them numbers its episodes from
    zero.
    """
    grouped: dict[str, list[Rollout]] = {}
    for roll in rollouts:
        if counts(roll):
            grouped.setdefault(roll.group, []).append(roll)

    batch = Batch(groups=len(grouped))
    graded = [roll for group in grouped.values() for roll in group]
    if graded:
        batch.pass_rate = sum(roll.score >= 1.0 for roll in graded) / len(graded)
        batch.mean_return = sum(episode_return(roll) for roll in graded) / len(graded)

    for group in grouped.values():
        returns = torch.tensor([episode_return(roll) for roll in group], dtype=torch.float32)
        # A group of one has no mean to be better than, and a group whose
        # returns are identical has nothing to prefer. Both are exactly zero
        # gradient, so they are dropped rather than multiplied by zero — the
        # difference is that a dropped group is *counted*.
        if len(group) < 2 or float(returns.std(unbiased=False)) <= EPSILON:
            batch.skipped += 1
            continue
        centred = (returns - returns.mean()) / (returns.std(unbiased=False) + EPSILON)
        for roll, advantage in zip(group, centred.tolist()):
            batch.advantages[roll.episode.id] = advantage

    return batch


@dataclass
class Update:
    """What one optimizer step's backward pass measured."""

    loss: float = 0.0
    kl: float = 0.0
    tokens: int = 0
    #: Absent, not zero, on a run that did one inner epoch — there is no ratio
    #: to clip. Kept as a name so the frame's shape is decided once.
    entropy: float = 0.0


def _logprobs(model: nn.Module, ids, positions, live, targets, rope):
    """Log-probability of each row's next token, [batch, length - 1].

    Through `cross_entropy` rather than an explicit `log_softmax`: the latter
    materializes a `[batch, length, vocab]` fp32 tensor and autograd keeps it for
    a backward pass that only ever needed one number per position. At a rollout
    batch's shapes that tensor, not the model, is what sets the peak.
    """
    if rope is not None:
        if not isinstance(model, Transformer):
            raise ValueError("legacy positional tensors cannot configure a declared model")
        values = model(ids, positions=positions, key_mask=live, rope=rope)[0]
    else:
        values = logits(model, ids, positions=positions, key_mask=live)
    flat = values[:, :-1].reshape(-1, values.shape[-1])
    return -F.cross_entropy(flat, targets.reshape(-1), reduction="none").view(targets.shape)


def accumulate(
    model,
    reference,
    rollouts: list[Rollout],
    batch: Batch,
    *,
    beta: float,
    micro_batch: int,
    rope=None,
) -> Update:
    """Backward the policy-gradient loss over `rollouts`, in micro-batches.

    Gradients are left on the parameters; clipping and the optimizer step are the
    caller's. Micro-batched because the logits of a whole rollout batch are the
    largest tensor in the process by a wide margin, and a run that fits only when
    every group has signal is a run that fails at random.

    Evaluation mode matches sampling while autograd remains enabled. Each
    module's original mode restores on success/error. Only the tokens drawn by
    the sampler carry loss. The prompt, the system block
    and every observation are context. A `<|tool-result|>` block is not a training
    target under `template.training_segments` and must not become one here:
    training it teaches a model to invent the shell's output, which is the same
    failure as teaching it to invent search results.
    """
    if type(micro_batch) is not int or micro_batch < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError("policy update requires a positive micro-batch and finite nonnegative KL weight")
    scored = [roll for roll in rollouts if roll.episode.id in batch.advantages]
    if any(len(roll.ids) != len(roll.policy) or len(roll.ids) < 2 or roll.policy[0]
           or any(type(flag) is not bool for flag in roll.policy) for roll in scored):
        raise ValueError("policy action provenance must align with its causal token sequence")
    update = Update()
    if not scored:
        return update

    weight = float(sum(sum(roll.policy) for roll in scored))
    if weight == 0:
        return update
    with evaluating(model), evaluating(reference):
        for start in range(0, len(scored), micro_batch):
            measured = _accumulate_chunk(model, reference, scored[start:start + micro_batch], batch, beta, weight, rope)
            update.loss += measured.loss
            update.kl += measured.kl
            update.tokens += measured.tokens

    update.loss /= weight
    update.kl /= weight
    return update


def _accumulate_chunk(model: nn.Module, reference: nn.Module | None, chunk: list[Rollout],
                      batch: Batch, beta: float, weight: float, rope) -> Update:
    """Backward one bounded ragged chunk, selecting drawn next-token targets and excluding left padding/observations.

    A policy flag at index i belongs to logits at i-1. Reduction uses the full
    batch's drawn-token denominator, so micro-batches preserve one update.
    """
    device = next(model.parameters()).device
    ids, positions, live = padded([roll.ids for roll in chunk], device)
    targets = ids[:, 1:]
    drawn = padded([[int(flag) for flag in roll.policy] for roll in chunk], device)[0]
    mask = (drawn[:, 1:] > 0) & live[:, 1:]
    logp = _logprobs(model, ids, positions, live, targets, rope)
    advantage = torch.tensor([batch.advantages[roll.episode.id] for roll in chunk],
                             dtype=torch.float32, device=device)[:, None]
    objective = -(logp * advantage * mask).sum()
    penalty = torch.zeros((), device=device)
    if beta and reference is not None:
        penalty = beta * _kl(reference, ids, positions, live, targets, mask, logp, rope)
    ((objective + penalty) / weight).backward()
    return Update(loss=float(objective.detach()), kl=float(penalty.detach()), tokens=int(mask.sum()))


def _kl(reference, ids, positions, live, targets, mask, logp, rope) -> torch.Tensor:
    """`KL(policy || reference)` over the drawn tokens, k3 estimator.

    Not a diagnostic. Without it a binary reward and a forgiving grader converge
    on a model that has learned to end every reply with a plausible-looking token
    and has forgotten how to write a sentence.

    k3 — `exp(r) - r - 1` for `r = logp_ref - logp` — because it is non-negative
    for every sample, where the naive `-r` is not: a single rollout can otherwise
    contribute a *negative* divergence and pay the policy for moving away from the
    reference, which is the opposite of what the term is for.

    The reference is frozen and its forward is under `no_grad`; the gradient
    reaches the policy through `logp`, which the caller already computed.
    """
    with torch.no_grad():
        reference_logp = _logprobs(reference, ids, positions, live, targets, rope)
    ratio = (reference_logp - logp).clamp(max=MAX_LOG_RATIO)
    return ((ratio.exp() - ratio - 1.0) * mask).sum()
