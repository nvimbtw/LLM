"""A batch of episodes rolled out, and the mask that comes back with it.

The mask is what this file is really about. Everything else a rollout produces
could be recovered afterwards from the gym's own record; which tokens the model
*drew* could not, and getting it wrong trains the model on the shell's output —
teaching it to invent observations, which is the same failure as teaching it to
invent search results.
"""

import pytest

from llm.chat import tools
from llm.chat.markup import Rules
from llm.chat.template import ASSISTANT_OPEN, TOOL_RESULT_OPEN
from llm.config import RlConfig
from llm.corpus import bpe
from llm.gym.client import Episode, Graded, Observed
from llm.model import checkpoint
from llm.train import rollout
from llm import storage
from tests.fixtures.tiny_env import TINY_MODEL

OBSERVATION = "<|codeblock|>notes.txt report.pdf<|/codeblock|>"
ROUNDS = 3


class FakeGym:
    """The gyms service, minus the service. Records what it was asked and answers
    with a *rendered* observation, because that is what the real seam returns —
    the engine never sees a `{stdout, stderr, exit_code}` triple."""

    def __init__(self, *, unreachable: bool = False):
        self.stepped: list[tuple[str, str, str]] = []
        self.graded: list[str] = []
        self.closed: list[str] = []
        self.unreachable = unreachable

    def step(self, episode_id, tool, argument, reply):
        if self.unreachable:
            from llm.backend import RemoteError

            raise RemoteError("nothing is listening")
        self.stepped.append((episode_id, tool, argument))
        return Observed(observation=OBSERVATION, reward=0.05, step_num=len(self.stepped))

    def finish(self, episode_id, reply):
        self.graded.append(episode_id)
        return Graded(score=1.0 if "report" in reply else 0.0, answer=reply[-20:])

    def abandon(self, episode_id):
        self.closed.append(episode_id)


@pytest.fixture
def loaded():
    model, _ = checkpoint.load(storage.weights_path(TINY_MODEL), "cpu")
    model.eval()
    return model, bpe.load(storage.model_dir(TINY_MODEL))


@pytest.fixture
def cfg():
    return RlConfig(
        model_name="x", base_model=TINY_MODEL, task_types=["bash_ls_easy"], max_tokens=12
    )


def _episodes(count: int) -> list[Episode]:
    return [
        Episode(
            id=f"ep-{i}",
            index=i,
            gym_id="bash",
            task_type="bash_ls_easy",
            seed=i,
            prompt=f"Which file is the odd one out? {i}",
            system="Use the bash tool to look around.",
        )
        for i in range(count)
    ]


def _rolled(model, tokenizer, gym, cfg, count=3, rounds=ROUNDS):
    return rollout.run(
        model, tokenizer, gym, _episodes(count), cfg,
        window=model.ctx_window, max_rounds=rounds,
    )


def test_every_episode_ends_graded_or_abandoned(loaded, cfg):
    model, tokenizer = loaded
    gym = FakeGym()
    batch = _rolled(model, tokenizer, gym, cfg)

    assert len(batch) == 3
    assert not any(roll.live for roll in batch)
    assert sorted(gym.graded) == [roll.episode.id for roll in batch]


def test_the_mask_marks_only_what_the_model_drew(loaded, cfg):
    """The prompt is context, every observation is context, and the assistant's
    own tokens are the policy. Nothing else may carry loss."""
    model, tokenizer = loaded
    started = rollout.start(_episodes(3), tokenizer, model.ctx_window)
    prompts = [len(roll.ids) for roll in started]
    batch = _rolled(model, tokenizer, FakeGym(), cfg)

    for roll, prompt in zip(batch, prompts):
        assert len(roll.ids) == len(roll.policy)
        # The prompt is the head of the sequence and none of it was drawn.
        assert not any(roll.policy[:prompt])


def test_an_observation_is_appended_verbatim_and_never_trained_on(loaded, cfg):
    """The engine renders no observation: what the backend sent is appended as
    tokens, and every flag over those tokens is False.

    Driven directly rather than through a rollout, because whether *this* model
    happens to call the shell is not what is being claimed — and a test that
    skips itself when it does not is a test that pins nothing.
    """
    model, tokenizer = loaded
    roll = rollout.start(_episodes(1), tokenizer, model.ctx_window)[0]
    roll.calls = [{"tool": "bash", "argument": "ls -la"}]
    roll.took(bpe.encode(tokenizer, "let me look"), drawn=True)
    before = len(roll.ids)

    scaffold = rollout.Scaffold.of(Rules.build(tokenizer), tokenizer)
    rollout._continue(roll, scaffold, tokenizer, OBSERVATION)

    assert len(roll.ids) > before
    assert not any(roll.policy[before:]), "the shell's own output carried loss"
    appended = bpe.decode(tokenizer, roll.ids[before:])
    assert OBSERVATION in appended
    assert appended.endswith(ASSISTANT_OPEN), "the next round has nowhere to generate into"


def test_appending_a_round_reproduces_what_one_pass_would_encode(loaded, cfg):
    """`build_prompt` encodes a whole rendered conversation in one pass to avoid
    cross-turn BPE merges, and this rollout appends instead. The two agree only
    because every seam is a marker, and a marker is an added token the BPE never
    merges across — so that property is pinned here rather than assumed."""
    model, tokenizer = loaded
    answered = tools.render_result("bash", "ls -la", OBSERVATION)
    seam = f"{ASSISTANT_OPEN}the odd one{TOOL_RESULT_OPEN}"

    whole = bpe.encode(tokenizer, seam)
    joined = (
        bpe.encode(tokenizer, ASSISTANT_OPEN)
        + bpe.encode(tokenizer, "the odd one")
        + bpe.encode(tokenizer, TOOL_RESULT_OPEN)
    )
    assert whole == joined
    assert bpe.decode(tokenizer, bpe.encode(tokenizer, answered)) == answered


def test_a_sandbox_that_cannot_be_reached_abandons_rather_than_invents(loaded, cfg):
    """The engine may not render an observation, so it cannot hand the model a
    caveat and carry on the way the browser's driver does. The episode leaves the
    batch instead — absent, not a zero."""
    model, tokenizer = loaded
    gym = FakeGym(unreachable=True)
    batch = _rolled(model, tokenizer, gym, cfg)

    abandoned = [roll for roll in batch if roll.stopped == "unreachable"]
    for roll in abandoned:
        assert roll.episode.id in gym.closed
        assert roll.score == 0.0


def test_the_terminal_round_cannot_call_the_shell(loaded, cfg):
    """A one-round rollout is all terminal, so no step may be taken at all: the
    tool-call opener is masked out of the grammar from the first token."""
    model, tokenizer = loaded
    gym = FakeGym()
    batch = _rolled(model, tokenizer, gym, cfg, rounds=1)

    assert gym.stepped == []
    assert all(roll.rounds == 1 for roll in batch)
    assert all(roll.stopped == "terminal" for roll in batch)


def test_generation_stops_at_the_window_rather_than_running_off_it(loaded):
    """A position past the RoPE tables *raises*, it does not degrade — so one
    long episode would take the whole batch down with it. The reply budget is
    the smaller of `max_tokens` and what is left of the window, and this asks for
    a reply far longer than the window to prove which one binds."""
    model, tokenizer = loaded
    greedy = RlConfig(
        model_name="x",
        base_model=TINY_MODEL,
        task_types=["bash_ls_easy"],
        max_tokens=model.ctx_window * 4,
    )
    batch = rollout.run(
        model, tokenizer, FakeGym(), _episodes(2), greedy,
        window=model.ctx_window, max_rounds=4,
    )

    for roll in batch:
        assert len(roll.ids) <= model.ctx_window
        assert not roll.live


def test_an_episode_with_no_window_left_is_graded_not_truncated(loaded, cfg):
    """A model answering a question it was shown part of reads afterwards as a
    model that scored zero."""
    model, tokenizer = loaded
    gym = FakeGym()
    batch = rollout.run(
        model, tokenizer, gym, _episodes(2), cfg,
        window=rollout.MIN_REPLY_TOKENS * 3, max_rounds=8,
    )
    assert all(not roll.live for roll in batch)
    assert all(roll.rounds <= 8 for roll in batch)


def test_a_window_exhaustion_does_not_abort_the_other_rollouts(loaded, cfg):
    """A context-full episode is graded without poisoning the batch's next pass."""
    model, tokenizer = loaded
    gym = FakeGym()
    batch = rollout.run(
        model, tokenizer, gym, _episodes(3), cfg,
        window=rollout.MIN_REPLY_TOKENS * 3, max_rounds=8,
    )

    assert len(batch) == 3
    assert all(not roll.live for roll in batch)
    assert all(len(roll.ids) <= rollout.MIN_REPLY_TOKENS * 3 for roll in batch)
