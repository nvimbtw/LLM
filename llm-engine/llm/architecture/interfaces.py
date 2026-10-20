"""Version-one causal language-model and optional worker-local capabilities.

These declarations describe tensors and model behavior, not service/storage or
worker transport. Serialized declarations are validated by architecture.spec.
Capability discovery requires both the saved declaration and a matching protocol;
structural runtime checks alone do not establish semantic validity.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from torch import Tensor


@dataclass(frozen=True)
class LanguageModelFacts:
    """Tokenizer-bound vocabulary count and trained maximum per-row position span."""

    vocabulary_size: int
    context_window: int


@dataclass(frozen=True)
class ExecutionOccurrence:
    """One resolved module instance and its flow/loop call path, using zero-based indices.

    Distinct call paths share parameter ownership when instance_path is equal,
    but receive separate request-cache and diagnostic identities.
    """

    instance_path: str
    call_path: tuple[int, ...]


@dataclass(frozen=True)
class CacheIdentity:
    """Parent-supplied owner identity preventing cache reuse across requests/definitions."""

    request_id: UUID
    definition_digest: str


@dataclass(frozen=True)
class RequestCache:
    """Request-owned tensor state, indexed by execution occurrence and named state port.

    The owning inference worker releases these references at completion, disconnect
    or cancellation. A forward returns a fresh mapping and must not mutate input
    cache tensors; tensors remain on the model's device and never travel over IPC.
    """

    identity: CacheIdentity
    entries: Mapping[ExecutionOccurrence, Mapping[str, Tensor]]


@dataclass(frozen=True)
class CachedResult:
    """Vocabulary logits for the supplied tokens and the next explicit request state."""

    logits: Tensor
    cache: RequestCache


@dataclass(frozen=True)
class BlockCachedResult:
    """Named internal block outputs and fresh named state without whole-LM assumptions."""

    outputs: Mapping[str, Tensor]
    state: Mapping[str, Tensor]


@dataclass(frozen=True)
class InspectionPoint:
    """An available diagnostic tensor at an actual execution occurrence and named port."""

    id: str
    occurrence: ExecutionOccurrence
    port: str
    kind: Literal["hidden", "attention"]
    title: str


@dataclass(frozen=True)
class TokenRowResult:
    """Token initialization outcome; unsupported models keep all parameters unchanged."""

    status: Literal["initialized", "skipped_unsupported"]
    initialized_ids: tuple[int, ...] = ()
    skipped_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExtendedContext:
    """Request-owned derived position state without changing the trained context.

    The mapping and its tensor values are immutable inputs to model computation;
    they are regenerated from the saved definition and explicit requested length.
    """

    context_window: int
    state: Mapping[str, Tensor]


@runtime_checkable
class LanguageModel(Protocol):
    """Minimum model adapter; torch.nn.Module separately owns parameters and mode."""

    @property
    def facts(self) -> LanguageModelFacts:
        """Return final tokenizer-bound vocabulary/context values without model execution."""
        ...

    def forward(
        self, tokens: Tensor, *, positions: Tensor | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """Return floating [batch, sequence, vocabulary] logits for int64 token inputs.

        Positions are int64 [batch, sequence]; key_mask is bool [batch, sequence]
        with True meaning visible. Live outputs must be causal, independent of
        padding and other rows. Inputs/persistent state are unchanged; normal
        PyTorch training-mode RNG is permitted. Eval remains deterministic with
        gradients enabled. The boundary validates IDs, shapes and resource limits.
        """
        ...


@runtime_checkable
class CachedLanguageModel(Protocol):
    """Optional serving capability, enabled only after full-prefix equivalence evidence."""

    def prefill(
        self, tokens: Tensor, *, identity: CacheIdentity,
        positions: Tensor | None = None, key_mask: Tensor | None = None,
    ) -> CachedResult:
        """Evaluate a nonempty prefix and create occurrence-scoped request cache tensors.

        Preserve the supplied identity. Positions and padding follow LanguageModel;
        no cache state is retained on the persistent module or shared across requests.
        """
        ...

    def decode(
        self, tokens: Tensor, cache: RequestCache, *, identity: CacheIdentity,
        positions: Tensor,
        key_mask: Tensor | None = None,
    ) -> CachedResult:
        """Return next logits/cache for int64 [batch, 1] tokens at explicit positions.

        key_mask covers past plus the new key. Reject cache.identity != identity,
        batch/device/dtype or context budget before allocation. Each occurrence
        keeps independent state even when module parameters are shared.
        """
        ...


@runtime_checkable
class CachedBlock(Protocol):
    """Optional hidden/attention-block state owned by its assembled execution occurrence."""

    def cached_forward(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor],
    ) -> BlockCachedResult:
        """Compute declared named outputs and fresh tensor state without mutating inputs.

        An empty state mapping starts a prefix. The assembled model validates the
        signature and owns request/occurrence identity, device, bounds and cleanup.
        Sharing Parameters never shares state between execution occurrences.
        """
        ...


@runtime_checkable
class ModelIntrospection(Protocol):
    """Optional diagnostic views; absent attention/activation measurements stay absent."""

    def inspection_points(self) -> tuple[InspectionPoint, ...]:
        """List declared occurrence/port views without executing a forward pass."""
        ...

    def inspect(
        self, tokens: Tensor, *, point_id: str,
        positions: Tensor | None = None, key_mask: Tensor | None = None,
    ) -> Tensor:
        """Measure one supported view within the caller's admitted diagnostic budget.

        Reject unknown IDs and preserve RNG, mode, parameters, buffers and caller
        tensors. Attention shape is [batch, heads, queries, keys]; hidden shape is
        [batch, sequence, channels]. Return no fabricated substitute for absent data.
        """
        ...


@runtime_checkable
class RecomputableModel(Protocol):
    """Optional memory trade using explicit occurrences and controlled RNG replay."""

    def recomputation_targets(self) -> tuple[ExecutionOccurrence, ...]:
        """Return legal independently configurable recomputation occurrences."""
        ...

    def configure_recomputation(self, occurrences: tuple[ExecutionOccurrence, ...]) -> None:
        """Select validated occurrences between steps, rejecting unknown or duplicate IDs.

        Preserve forward values, gradients, dropout RNG replay and parameter/cache
        ownership. Do not change saved architecture identity or enable recomputation
        in cached inference. The owning worker configures this mutable execution policy.
        """
        ...


@runtime_checkable
class TokenRowInitializer(Protocol):
    """Optional SFT token-row initialization with explicit tied/untied table ownership."""

    def initialize_token_rows(self, token_ids: tuple[int, ...]) -> TokenRowResult:
        """Initialize valid unique requested rows under no_grad and report affected IDs.

        The capability owns the numerical policy for every affected input/output
        table; tied parameters change once. Out-of-vocabulary IDs are reported as
        skipped to preserve legacy behavior. No assumed embedding attribute is used.
        Models lacking this capability skip initialization and remain usable for SFT.
        """
        ...


@runtime_checkable
class ContextExtensibleModel(Protocol):
    """Optional positional extension, authorized for an explicit profile and signature."""

    def extended_context(self, target_context_window: int) -> ExtendedContext:
        """Derive bounded position tensors for the target without modifying saved buffers.

        Reject an unsupported length/position scheme before allocating. The trained
        LanguageModelFacts.context_window and structural definition remain unchanged.
        """
        ...

    def forward_extended(
        self, tokens: Tensor, *, context: ExtendedContext,
        positions: Tensor | None = None, key_mask: Tensor | None = None,
    ) -> Tensor:
        """Evaluate full-prefix logits with validated request-local positional state.

        Preserve all LanguageModel causality/mutation contracts. Separate cache and
        extension support does not authorize their combination: extended cached
        signatures need explicit joint equivalence/capture evidence and an adapter.
        """
        ...
