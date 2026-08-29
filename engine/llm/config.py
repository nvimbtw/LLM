"""Run configuration. These double as the FastAPI request bodies, so the API
layer parses and validates a run without knowing anything about training.

One object rather than a positional argument list: the Rust `train()` needed
`#[allow(clippy::too_many_arguments)]` at 13 parameters, and every one of them
had to be threaded through each layer by hand.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `model_name` collides with pydantic v2's protected `model_` namespace, which
# would warn on every instantiation. The field name is worth keeping — it is
# what the Rust config, the job rows, and the on-disk layout all call it.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class DatasetSelection(BaseModel):
    """A dataset to train on, optionally capped to its first `count` documents.

    The dataset is one the **data engine** owns, named by its id: the engine
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
    of the engine. It is refused rather than looked up: there is no directory
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
        description="roughly how many intermediate checkpoints to leave behind; null is "
        "`progress.TARGET_CHECKPOINTS` and 0 is none at all. Each one is a whole model "
        "plus the optimizer moments, so this is a disk budget as much as a cadence",
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

    epochs: int = 2
    # An order of magnitude below pretrain. SFT starts from weights that already
    # work, and the usual pretrain rate walks straight off them.
    lr: float = 3e-5
    batch_size: int = 4
    warmup_steps: int = 50
    grad_clip: float = 1.0
    weight_decay: float = 0.1

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
    role: str
    text: str


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
    suite: str = "language"

    max_tokens: int = 96
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    # Fixed so a benchmark is reproducible: two runs of one model must not differ
    # because the sampler drew differently.
    seed: int = 1234
