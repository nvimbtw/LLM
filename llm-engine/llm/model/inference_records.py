"""Typed private serving requests and bounded results; no arbitrary executable operations cross IPC."""

import math
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator
from pydantic import TypeAdapter

from llm.architecture.spec import Contract, NonnegativeInt
from llm.architecture.wire import AttentionSelection
from llm.config import ChatConfig, EvalConfig

MAX_MEASUREMENT_TOKENS = 65536
MAX_REQUEST_CHARACTERS = 1 << 20


def sampling_bounds(cfg: ChatConfig | EvalConfig) -> None:
    """Reject nonfinite or excessive generation work before custom model execution; retain the existing neutral sampling values."""
    if (type(cfg.max_tokens) is not int or not 1 <= cfg.max_tokens <= 65536
            or type(cfg.top_k) is not int or not 0 <= cfg.top_k <= 65536
            or not math.isfinite(cfg.temperature) or not 0 <= cfg.temperature <= 10
            or not math.isfinite(cfg.top_p) or not 0 <= cfg.top_p <= 1
            or cfg.seed is not None and (type(cfg.seed) is not int or not 0 <= cfg.seed < 2**53)):
        raise ValueError("inference sampling options exceed finite supported bounds")


class GenerationInput(Contract):
    """Bind canonical tokenized prompt and sampling/grammar configuration for one supervised request."""

    config: ChatConfig
    prompt_ids: Annotated[tuple[NonnegativeInt, ...], Field(min_length=1, max_length=65536)]

    @model_validator(mode="after")
    def bounded_generation(self):
        """Keep declared serving inside its saved full-prefix profile and bound all text/config work before model calls."""
        generation_bounds(self.config)
        return self


def generation_bounds(config: ChatConfig) -> None:
    """Validate serving options and serialized configuration before model loading or prompt construction."""
    sampling_bounds(config)
    if config.context_multiplier != 1 or (config.max_tool_calls or 0) > 32:
        raise ValueError("declared serving requires trained context and at most 32 tool calls")
    if len(config.model_dump_json().encode("utf-8")) > 4 << 20:
        raise ValueError("generation configuration exceeds its private byte budget")


class GeneratedTool(Contract):
    """A parsed known tool call for presentation/coordination, never an instruction to this worker."""

    tool: Annotated[str, Field(min_length=1, max_length=128)]
    argument: Annotated[str, Field(max_length=MAX_REQUEST_CHARACTERS)]


class GenerationResult(Contract):
    """The existing stateless chat summary, bounded before publication as data-only JSON."""

    model_name: Annotated[str, Field(min_length=1, max_length=255)]
    text: Annotated[str, Field(max_length=MAX_REQUEST_CHARACTERS)]
    prompt_tokens: Annotated[NonnegativeInt, Field(le=65536)]
    generated_tokens: Annotated[NonnegativeInt, Field(le=131072)]
    drained_blocks: Annotated[NonnegativeInt, Field(le=65536)]
    tool_calls: Annotated[tuple[GeneratedTool, ...], Field(max_length=32)]
    stopped: Literal["length", "marker"]
    elapsed_secs: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class MeasurementInput(Contract):
    """Bound likelihood/generation input; bench-engine continues to own corpus metrics and scoring policy."""

    operation: Literal["language"] = "language"
    config: EvalConfig
    text: Annotated[str, Field(max_length=MAX_REQUEST_CHARACTERS)]
    prompts: Annotated[tuple[Annotated[str, Field(max_length=32768)], ...], Field(max_length=32)]

    @model_validator(mode="after")
    def bounded_measurement(self):
        """Cap aggregate prompt characters and generation work while validating every sampler option."""
        sampling_bounds(self.config)
        if sum(map(len, self.prompts)) > MAX_REQUEST_CHARACTERS or len(self.prompts) * self.config.max_tokens > MAX_MEASUREMENT_TOKENS:
            raise ValueError("measurement exceeds aggregate prompt or generation budget")
        return self


class MeasurementModel(Contract):
    """Report real generic LM facts; unsupported fixed-layout dimensions remain unavailable."""

    id: Annotated[str, Field(min_length=1, max_length=255)]
    name: Annotated[str, Field(min_length=1, max_length=255)]
    layers: None = None
    dimensions: None = None
    heads: None = None
    context: NonnegativeInt
    vocab: NonnegativeInt
    parameters: NonnegativeInt


class LikelihoodResult(Contract):
    """Aggregate natural-log likelihood without benchmark scoring policy or unbounded per-token arrays."""

    total: Annotated[float, Field(le=0, allow_inf_nan=False)]
    count: Annotated[NonnegativeInt, Field(le=MAX_MEASUREMENT_TOKENS)]
    unscored_bytes: NonnegativeInt


class MeasuredSample(Contract):
    """Retain one bounded unconstrained continuation paired with its original prompt."""

    prompt: Annotated[str, Field(max_length=32768)]
    text: Annotated[str, Field(max_length=MAX_REQUEST_CHARACTERS)]


class MeasurementResult(Contract):
    """Return actual LM measurements while leaving unsupported legacy ablation absent."""

    model: MeasurementModel
    likelihood: LikelihoodResult
    samples: Annotated[tuple[MeasuredSample, ...], Field(max_length=32)]
    ablation: None = None
    cancelled: StrictBool = False


class AttentionInput(Contract):
    """Interpret one bounded attention request inside the numerical worker, with no browser paths or execution hooks."""

    operation: Literal["attention"] = "attention"
    model_name: Annotated[str, Field(min_length=1, max_length=255)]
    selection: AttentionSelection


class InspectionInput(Contract):
    """Read declared candidate metadata and existing worker-lifetime proof bounds from the resident model."""

    operation: Literal["inspection_points"] = "inspection_points"
    model_name: Annotated[str, Field(min_length=1, max_length=255)]


class InspectionRefusal(Contract):
    """Publish a classified observation refusal with static public-safe wording rather than custom source exception text."""

    refused: Literal[True] = True
    code: Literal["unsupported", "selection", "input", "budget", "validation"]

    @property
    def message(self) -> str:
        """Return bounded actionable wording for a known observation failure without exposing source diagnostics."""
        return {
            "unsupported": "This model has no supported attention view for that selection.",
            "selection": "Select an available attention point or a head within its actual range.",
            "input": "Attention text must fit the saved vocabulary and the model's attention token limit.",
            "budget": "Attention inspection exceeds admitted memory; use fewer tokens.",
            "validation": "This attention view failed its capture or numerical checks.",
        }[self.code]


MeasurementRequest = Annotated[MeasurementInput | AttentionInput | InspectionInput, Field(discriminator="operation")]
MEASUREMENT_REQUEST = TypeAdapter(MeasurementRequest)
