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
    """A dataset to train on, optionally capped to its first `count` files.

    The cap is what keeps a run's size under control when one folder dwarfs the
    rest: FineWeb averages ~70k tokens per file and holds 99% of the corpus here,
    so `{"name": "FineWeb", "count": 500}` is the difference between a 56M-token
    run and a 390M-token one.

    Files are ordered by name and the first `count` taken, so a capped run is
    reproducible and a re-pack selects the same text. `None` or `0` means all of
    them.

    A bare string parses as an uncapped selection, so `["FineWeb"]` and
    `[{"name": "FineWeb"}]` are the same request.
    """

    name: str
    count: int | None = Field(
        default=None, ge=0, description="take only the first N files; 0 or null means all"
    )


def _as_selections(value):
    """Accept `["Name"]`, `[{"name": ..., "count": ...}]`, or a mix of both."""
    if not isinstance(value, list):
        return value
    return [{"name": item} if isinstance(item, str) else item for item in value]


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

    # ── Tokenizer ───────────────────────────────────────────────────────────
    max_vocab: int = 8192

    # ── Optimization ────────────────────────────────────────────────────────
    epochs: int = 1
    lr: float = 3e-4
    batch_size: int = 8
    warmup_steps: int = 300
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # ── Data ────────────────────────────────────────────────────────────────
    datasets: list[DatasetSelection] = Field(
        default_factory=list,
        description="dataset folders to train on; empty means all of them. Each entry is "
        "a name or {name, count} to cap it to its first N files",
    )
    curriculum: list[CurriculumStage] = Field(
        default_factory=list,
        description="ordered stages; empty collapses to one stage from `datasets` + `epochs`",
    )
    shuffle: bool = True
    seed: int = 0
    rebuild_dataset: bool = Field(
        default=False,
        description="re-tokenize even when a corpus is already on disk",
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

    # Drop the <|thinking|> blocks from the chat data this run trains on,
    # teaching the model to answer directly — for a model too small to hold a
    # reasoning block together.
    strip_thinking: bool = False

    shuffle: bool = True
    seed: int = 0
    rebuild_dataset: bool = False


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


class MeasureRequest(BaseModel):
    """A conversation to price against SFT's window."""

    model_config = _ALLOW_MODEL_PREFIX
    model_name: str
    turns: list[ChatTurn]
    system: str | None = None


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


class InferConfig(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX

    model_name: str
    prompt: str
    max_tokens: int = 128

    # Each filter is disabled by its own neutral value — temperature 0.0
    # collapses to argmax, a top_k past the vocabulary keeps everything, and a
    # top_p of 1.0 cuts nothing — so wanting one filter need not mean knowing
    # about the other two.
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int | None = None
