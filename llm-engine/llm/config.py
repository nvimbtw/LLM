"""Run configuration. These double as the FastAPI request bodies, so the API
layer parses and validates a run without knowing anything about training.

One object rather than a positional argument list: the Rust `train()` needed
`#[allow(clippy::too_many_arguments)]` at 13 parameters, and every one of them
had to be threaded through each layer by hand.
"""

from typing import Annotated

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `model_name` collides with pydantic v2's protected `model_` namespace, which
# would warn on every instantiation. The field name is worth keeping — it is
# what the Rust config, the job rows, and the on-disk layout all call it.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class DatasetSelection(BaseModel):
    """A dataset to train on, optionally capped to its first `count` documents.

    The dataset is one the **data engine** owns, named by its id: the LLM engine
    stores no corpora of its own, and a name is something a person can change
    without changing a single token.

    The cap is what keeps a run's size under control when one dataset dwarfs the
    rest — FineWeb holds 99% of the text in a typical selection here, so
    `{"id": ..., "count": 50000}` is the difference between a run that finishes
    tonight and one that does not.

    Documents are ordered by `ordinal` and the first `count` taken, so a capped
    run is reproducible and a re-pack selects the same text. `None` or `0` means
    the whole dataset. It used to mean "the first N *files*", which was
    reproducible only by accident of how a directory listing sorted.

    A bare string parses as an uncapped selection, so `["<id>"]` and
    `[{"id": "<id>"}]` are the same request.
    """

    id: str = Field(description="the data engine's id for the dataset")
    name: str = Field(
        default="",
        description="what to call it, for readability; the data engine's own name wins",
    )
    count: int | None = Field(
        default=None,
        ge=0,
        description="take only the first N documents; 0 or null means all of them",
    )


def _as_selections(value):
    """Accept `["<id>"]`, `[{"id": ..., "count": ...}]`, or a mix of both.

    A selection naming only `name` is a request written before datasets moved out
    of the LLM engine. It is refused rather than looked up: there is no directory
    left to look in, and two datasets may share a name.
    """
    if not isinstance(value, list):
        return value

    selections = []
    for item in value:
        if isinstance(item, str):
            selections.append({"id": item})
        elif isinstance(item, dict) and "id" not in item and "name" in item:
            raise ValueError(
                f"dataset {item['name']!r} is selected by name. Datasets live in the "
                f"data engine now and are selected by id — re-queue this run from the "
                f"dataset list"
            )
        else:
            selections.append(item)
    return selections


class CurriculumStage(BaseModel):
    """One stage: a set of datasets trained for its own epoch count before the
    run moves on to the next.

    Order is the curriculum — TinyStories before Wikipedia before FineWeb — and
    it is the whole point: a model that meets easy text first has somewhere to
    stand when the hard text arrives.
    """

    name: str | None = None
    datasets: list[DatasetSelection] = Field(default_factory=list)
    epochs: int = 1

    _normalize = field_validator("datasets", mode="before")(_as_selections)


class TrainConfig(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    model_name: str = Field(description="names the run's directory under artifacts/models/")

    # ── Architecture ────────────────────────────────────────────────────────
    dimensions: int = 256
    heads: int = 8
    layers: int = 4
    ff_multiplier: int = 4
    ctx_window: int = 256
    rope_base: float = Field(
        default=10000.0,
        gt=1.0,
        description="RoPE frequency base. Higher keeps distant positions distinguishable "
        "further out, at the cost of fine-grained local resolution; the usable range is "
        "roughly proportional to it. Travels with the checkpoint, so inference reuses "
        "whatever the run trained with",
    )
    learned_temperature: bool = Field(
        default=False,
        description="give every attention head its own learned softmax temperature, so a "
        "head can choose how sharp or diffuse it is instead of inheriting one shared "
        "1/sqrt(head_dim). Adds `heads` parameters per layer and starts at exactly the "
        "untempered behaviour; off keeps the state_dict byte-identical to older runs",
    )

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer_id: str | None = Field(
        default=None,
        description="reuse a vocabulary built by a `tokenize` run; null learns one from "
        "this run's own corpora, which is what every run used to do",
    )
    #: Ignored when `tokenizer_id` names a build — a vocabulary's size is a
    #: property of the build, not of the run that consumes it.
    max_vocab: int = 8192

    # ── Optimization ────────────────────────────────────────────────────────
    epochs: int = 1
    lr: float = 3e-4
    batch_size: int = 8
    warmup_steps: int = 300
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # ── Continuation ────────────────────────────────────────────────────────
    recover_exact: bool = False
    resume_from: str | None = Field(
        default=None,
        description="a model id to carry on from — a finished model or one of its "
        "checkpoints, either works. When this run's corpus and schedule match the "
        "state's, the optimizer, the RNG and the data order are restored and training "
        "picks up at the saved step; otherwise the weights and the moments carry over "
        "and the schedule is this run's. Architecture comes from the state, and a "
        "request that restates it differently is refused",
    )
    checkpoint_target: int | None = Field(
        default=None,
        ge=0,
        description="roughly how many recovery saves to make; null is "
        "`progress.TARGET_CHECKPOINTS` and 0 disables saves. Retention keeps the "
        "latest automatic generation and any explicitly preserved generations",
    )

    # ── Data ────────────────────────────────────────────────────────────────
    datasets: list[DatasetSelection] = Field(
        default_factory=list,
        description="datasets to train on, by data-engine id. Each entry is an id or "
        "{id, count} to cap it to its first N documents. A run must name at least one",
    )
    curriculum: list[CurriculumStage] = Field(
        default_factory=list,
        description="ordered stages; empty collapses to one stage from `datasets` + `epochs`",
    )
    shuffle: bool = True
    seed: int = 0
    rebuild_dataset: bool = Field(
        default=False,
        description="re-learn the vocabulary and re-pack the corpus even though a build "
        "matching this request exists. An escape hatch for a corpus edited or corrupted "
        "on disk — correctness is the key's job, not this flag's",
    )

    # ── Performance ─────────────────────────────────────────────────────────
    # These change speed, and `precision` changes numerics. Defaults are the fast
    # path; drop to "fp32" and compile=False to reproduce a run bit-for-bit
    # against the pre-Ampere-features behaviour.
    precision: str = Field(
        default="bf16",
        pattern="^(bf16|fp16|fp32)$",
        description="bf16 unlocks the flash-attention kernel, which fp32 cannot use at all; "
        "master weights stay fp32 either way",
    )
    compile_model: bool = Field(
        default=True,
        description="torch.compile the forward/backward; worth ~1.4x on top of bf16 at small "
        "model sizes, where kernel launch overhead dominates. Falls back to eager if it fails",
    )
    allow_tf32: bool = Field(
        default=True,
        description="TF32 for fp32 matmuls on Ampere+; torch defaults this off",
    )
    # The one setting here that buys memory rather than speed, and the only lever
    # that decides how large a model this card can hold at all: activations are
    # ~75% of a step's peak at the sizes trained here, and they are what grows
    # with depth.
    #
    # `None` — the default — is not "off". It asks `llm.train.memory.plan` to
    # read the card and recompute the *fewest* blocks that make the run fit,
    # which is almost never all of them and is very often none. The two ends are
    # still sayable and still win when said: a run that has been measured should
    # not have its setting second-guessed by an estimate.
    recompute_blocks: bool | int | None = Field(
        default=None,
        description="how many transformer blocks recompute their activations in the "
        "backward pass instead of storing them, rather than whether any do. Each one "
        "is worth ~300 MiB and costs ~8 ms a step at 512d/18L/ctx2048/bs8, and the "
        "arithmetic is identical either way, so this changes what fits on the card "
        "rather than what the model learns. null plans it against the card's free "
        "memory — which is what you want; true means every block, false none, and an "
        "integer names the count",
    )

    _normalize_datasets = field_validator("datasets", mode="before")(_as_selections)

    @model_validator(mode="after")
    def _check_architecture(self):
        if self.dimensions % self.heads != 0:
            raise ValueError(
                f"dimensions {self.dimensions} must be divisible by heads {self.heads}"
            )
        head_dim = self.dimensions // self.heads
        # RoPE rotates adjacent pairs (2i, 2i+1) of each head's dimensions, so an
        # odd head_dim leaves the last one unrotated — a silently half-positioned
        # model rather than a failure. Cheap to catch here.
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE: dimensions {self.dimensions} "
                f"/ heads {self.heads} = {head_dim}"
            )
        return self

    @property
    def head_dim(self) -> int:
        return self.dimensions // self.heads

    def stages(self) -> list[CurriculumStage]:
        """The curriculum, or the single implicit stage a flat config describes.

        One place so nothing downstream has to handle both shapes.
        """
        if self.curriculum:
            return self.curriculum
        return [CurriculumStage(name=None, datasets=self.datasets, epochs=self.epochs)]


class SftConfig(BaseModel):
    """Supervised fine-tuning of an already-pretrained model.

    Architecture is deliberately absent: it comes from the base model's
    checkpoint. Letting a request restate it would let the two disagree, and the
    weights would win — silently, after the run.
    """

    model_config = _ALLOW_MODEL_PREFIX

    model_name: str = Field(description="names the fine-tuned run's directory")
    base_model: str = Field(description="the pretrained model to fine-tune from")
    resume_from: str | None = None
    checkpoint_target: int = Field(default=10, ge=0)

    epochs: int = 2
    # An order of magnitude below pretrain. SFT starts from weights that already
    # work, and the usual pretrain rate walks straight off them.
    lr: float = 3e-5
    batch_size: int = 4
    warmup_steps: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    # The one performance knob SFT restates, because it is not one: a base model
    # large enough to need it in pretrain cannot be fine-tuned without it either,
    # and the rest of that section — precision, compile, tf32 — only decides how
    # fast a run that already fits goes. See `TrainConfig.recompute_blocks`.
    recompute_blocks: bool | int | None = Field(
        default=None,
        description="how many transformer blocks recompute their activations in the "
        "backward pass instead of storing them, rather than whether any do. Same "
        "arithmetic either way. null plans it against the card's free memory — which "
        "is what you want; true means every block, false none, an integer the count",
    )

    # ── Data ────────────────────────────────────────────────────────────────
    # Which SFT sets to fine-tune on, exactly as `TrainConfig.datasets` selects
    # pretrain corpora.
    #
    # Deliberately not a curriculum: the point is two *separate* runs from one
    # base model — a non-thinking fine-tune and a thinking one — producing two
    # models that can be loaded and compared against each other. Ordered stages
    # would collapse them into a single set of weights.
    datasets: list[DatasetSelection] = Field(
        default_factory=list,
        description="chat datasets to fine-tune on, by data-engine id. Each entry is an "
        "id or {id, count} to cap it to its first N conversations",
    )

    # Drop the <|thinking|> blocks from the chat data this run trains on,
    # teaching the model to answer directly — for a model too small to hold a
    # reasoning block together.
    #
    # Independent of `datasets` on purpose. Pointing this at a thinking set is
    # usually a mistake, but it is also the only way to build a strip-thinking
    # control against the same conversations, so the two levers stay separate.
    strip_thinking: bool = False

    shuffle: bool = True
    seed: int = 0
    rebuild_dataset: bool = False

    # Report KL(base || tuned) on a held-out chat batch, on diagnosed steps.
    #
    # Off by default because it is the one diagnostic that costs *memory* rather
    # than time: it keeps the unmodified base model resident beside the one being
    # trained, so a run that can just fit its weights cannot afford it. Everything
    # else in `Diagnostics` is a reading off a pass that was happening anyway.
    measure_base_kl: bool = False

    _normalize_datasets = field_validator("datasets", mode="before")(_as_selections)


class TaskWeight(BaseModel):
    """One task type and how often a batch should draw it.

    A bare string parses as weight 1, so `["bash_ls_easy"]` and
    `[{"id": "bash_ls_easy"}]` are the same request — the shape
    `DatasetSelection` already uses, for the same reason: the common case should
    not have to be spelled out.

    In a general run these weights apply within each source gym. gym_weights
    controls the source shares, independently of how many types each registers.
    """

    id: str = Field(description="the gym's id for the task type")
    weight: float = Field(default=1.0, gt=0.0, le=1_000_000, allow_inf_nan=False)


def _as_task_weights(value):
    if not isinstance(value, list):
        return value
    return [{"id": item} if isinstance(item, str) else item for item in value]


class PracticeSchedule(BaseModel):
    """Validation controls the lessons; the fixed test never controls them."""

    check_every: int = Field(default=10, ge=1)
    validation_samples: int = Field(default=100, ge=10, le=100)
    test_samples: int = Field(default=16, ge=1, le=100)
    promote_at: float = Field(default=0.9, gt=0.0, le=1.0)
    retain_at: float = Field(default=0.75, ge=0.0, lt=1.0)
    consecutive_checks: int = Field(default=2, ge=2, le=10)

    @model_validator(mode="after")
    def _thresholds(self):
        if self.retain_at >= self.promote_at:
            raise ValueError("retain_at must be lower than promote_at")
        return self


class RlConfig(BaseModel):
    """Reinforcement learning against a gym, from a model that already scores.

    Architecture is absent for the same reason `SftConfig` omits it: it comes
    from the base model's checkpoint, and a request that could restate it is a
    request that could disagree with the weights.

    **A pass rate strictly between 0 and 1 is a precondition, not a result.**
    These tasks are pass or fail, and a group of rollouts that all fail has a
    zero gradient — exactly, not approximately. `gyms-engine/scripts/gate.py` is what
    asks whether a model is in that band before a run is worth starting.
    """

    model_config = _ALLOW_MODEL_PREFIX

    model_name: str = Field(description="names the run's directory under artifacts/models/")
    base_model: str = Field(description="the model to start from, and the frozen reference")
    resume_from: str | None = None

    # ── The batch ───────────────────────────────────────────────────────────
    gym_id: str = Field(
        default="bash",
        description="which gym the task types belong to",
    )
    gym_weights: dict[str, Annotated[float, Field(gt=0, le=1_000_000, allow_inf_nan=False)]] = Field(
        default_factory=dict, description="relative source gym weights for the general gym",
    )
    task_types: list[TaskWeight] = Field(
        default_factory=list,
        min_length=1,
        description="which task types to draw from; an id, or {id, weight}",
    )
    #: Rollouts per task, and therefore the size of the baseline an advantage is
    #: measured against. A group of one contributes exactly zero gradient.
    group_size: int = Field(default=8, ge=2, le=256)
    #: Distinct tasks per iteration. `tasks_per_batch * group_size` is the whole
    #: batch, and it is bounded by the gyms engine's own ceiling on episodes per
    #: run — which exists because just-bash holds a finite number of filesystems.
    tasks_per_batch: int = Field(default=4, ge=1, le=256)
    iterations: int = Field(default=50, ge=1)
    max_rounds: int = Field(default=8, ge=1, le=32)

    # ── The update ──────────────────────────────────────────────────────────
    lr: float = 1e-6
    #: How hard the run is held to the model it started from. Not a diagnostic:
    #: without it a binary reward and a forgiving grader converge on a model that
    #: ends every reply with a plausible-looking token and has forgotten how to
    #: write a sentence.
    beta: float = 0.02
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    #: How many rollouts are backwarded at once. The logits of a whole batch are
    #: the largest tensor in the process, so this is a memory dial and not a
    #: statement about the gradient — every micro-batch lands in the same step.
    micro_batch: int = Field(default=4, ge=1)

    # ── Sampling ────────────────────────────────────────────────────────────
    #: **Temperature above zero or the group is one rollout counted G times.**
    #: There is no sampler seed here for the same reason: pinning the draw would
    #: make G siblings G copies, and a group with no variance in its returns has
    #: no gradient in it.
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = 40
    top_p: float = 0.95
    max_tokens: int = 256
    context_multiplier: float = Field(default=1.0, ge=1.0)

    # ── Seeds ───────────────────────────────────────────────────────────────
    seed_base: int = Field(
        default=0,
        description="where the training batches start drawing tasks from",
    )
    #: Reserved for measuring this run, and never drawn during it. Held-out seeds
    #: are the whole of the methodology — a seed that trained is a seed the model
    #: may have memorised the answer to — and keeping them apart is a
    #: subtraction rather than a feature.
    holdout_base: int = Field(default=1_000_000)

    checkpoint_target: int | None = Field(default=None, ge=0)
    curriculum: PracticeSchedule | None = None

    _normalize_task_types = field_validator("task_types", mode="before")(_as_task_weights)

    def lanes(self) -> list[tuple[str, list[TaskWeight]]]:
        """One gym run per iteration, including a general run's mixed sources.

        The gyms engine selects general tasks before forming rollout groups.
        Advantages remain standardized within each group of one question;
        source gym identity never merges groups.
        """
        return [(self.gym_id, self.task_types)]

    @model_validator(mode="after")
    def _holdout_is_disjoint(self) -> "RlConfig":
        """A run cannot be measured on seeds it trained on, and the arithmetic
        that would overlap them is not obvious from the two numbers."""
        if not self.task_types:
            raise ValueError("select at least one task type")
        if self.gym_weights and self.gym_id != "general":
            raise ValueError("gym_weights is only supported by the general gym")
        if self.gym_id == "general" and len({t.id for t in self.task_types}) != len(self.task_types):
            raise ValueError("duplicate task types; use weights to change their frequency")
        drawn = self.seed_base + self.iterations * self.tasks_per_batch
        if self.curriculum is not None and self.gym_id != "math":
            raise ValueError("automatic curriculum is currently available for the math gym")
        if self.seed_base <= self.holdout_base < drawn:
            raise ValueError(
                f"holdout_base {self.holdout_base} is inside the seeds this run will "
                f"train on ({self.seed_base}..{drawn})"
            )
        return self


class TokenizerConfig(BaseModel):
    """Learning one reusable vocabulary from a selection of corpora.

    Its own kind of run rather than a step inside training: a vocabulary is the
    slowest part of phase 1, it is shared by every run that names it, and what it
    is a function of — which corpora, at what size — has nothing to do with an
    architecture or a learning rate.
    """

    model_config = _ALLOW_MODEL_PREFIX

    tokenizer_id: str = Field(description="names the directory under artifacts/tokenizers/")
    max_vocab: int = 8192
    datasets: list[DatasetSelection] = Field(
        default_factory=list,
        description="datasets to learn merges from, by data-engine id. Each entry is an "
        "id or {id, count} to cap it to its first N documents",
    )
    rebuild: bool = Field(
        default=False,
        description="re-learn even though a build matching this request exists",
    )

    _normalize_datasets = field_validator("datasets", mode="before")(_as_selections)


class ChatTurn(BaseModel):
    """One turn of a conversation a caller is holding.

    A `tool` turn can be given two ways, and the difference is who renders it.
    **Naming `tool` and `argument` hands the rendering to the LLM engine**
    (`llm.chat.tools.render_result` wraps `text` in the call it answers), which
    is what a caller holding its own conversation should do: that layout is the
    one SFT teaches, and a caller spelling `<|tool-call|>…<|/tool-call|>` itself
    would be a second copy of it, free to drift into a shape the model was never
    trained on. Leaving them out means `text` is already the rendered inner
    content — which is how a turn read back from a stored chat arrives, because
    that is the form it was written in.
    """

    role: str
    text: str
    #: Given together or not at all, and only on a `tool` turn: which call
    #: `text` answers.
    tool: str | None = None
    argument: str | None = None

    @model_validator(mode="after")
    def _tool_turns_name_their_call(self) -> "ChatTurn":
        if (self.tool is None) != (self.argument is None):
            raise ValueError("a tool and its argument are given together or not at all")
        if self.role != "tool" and self.tool is not None:
            raise ValueError("only a tool turn carries a tool and argument")
        return self


class ChatConfig(BaseModel):
    """One assistant turn, generated from a whole conversation."""

    model_config = _ALLOW_MODEL_PREFIX

    model_name: str
    turns: list[ChatTurn]
    system: str | None = None
    # "low" | "med" | "high", or absent for no budget marker at all — the absence
    # is a real state, not a fourth level.
    budget: str | None = None

    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int | None = None

    max_tool_calls: int | None = Field(
        default=None,
        ge=0,
        description="how many tool calls this turn may open; absent means the "
        "LLM engine's own per-turn budget. 0 masks the tool-call marker out of the "
        "grammar entirely, so the turn can only answer in prose — which is what "
        "a gym episode's last round asks for",
    )

    context_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        description="stretch the context window to this multiple of the model's trained "
        "ctx_window, via NTK-aware RoPE scaling computed fresh for the turn. 1 (default) "
        "runs the model exactly as trained; above 1 trades some quality at the tokens "
        "past ctx_window for headroom the model was never trained at",
    )


class CountRequest(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    model_name: str
    text: str


class EvalConfig(BaseModel):
    """Benchmark a trained model against held-out reference prose."""

    model_config = _ALLOW_MODEL_PREFIX

    model_name: str
    # Which suite to run. "language" scores held-out prose and generation;
    # "markup" scores marker use; "assistant" scores held-out replies. They are
    # separate because a base model has never seen the chat scaffold, and folding
    # those stages into one number would mark every pretrain down for failing a
    # task it was not trained on.
    suite: Literal["language"] = "language"

    max_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    # Fixed so a benchmark is reproducible: two runs of one model must not differ
    # because the sampler drew differently.
    seed: int = 1234
