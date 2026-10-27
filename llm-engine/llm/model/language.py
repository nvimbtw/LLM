"""The shared logits/facts boundary for legacy Transformers and declared recipe language models."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn

from llm.architecture.interfaces import LanguageModel, LanguageModelFacts
from llm.model.transformer import Transformer


def model_facts(model: nn.Module) -> LanguageModelFacts:
    """Read explicit vocabulary/context authority; unsupported models never inherit guessed Transformer dimensions."""
    if isinstance(model,Transformer):
        facts = LanguageModelFacts(model.vocab,model.ctx_window)
    elif isinstance(model,LanguageModel):
        facts = model.facts
    else:
        raise ValueError("model does not implement the language-model contract")
    if (not isinstance(facts,LanguageModelFacts) or type(facts.vocabulary_size) is not int
            or type(facts.context_window) is not int or min(facts.vocabulary_size,facts.context_window) < 1):
        raise ValueError("language model facts require positive vocabulary and context")
    return facts


def logits(model: nn.Module, tokens: Tensor, *, positions: Tensor | None = None, key_mask: Tensor | None = None) -> Tensor:
    """Return full-prefix logits with explicit position/padding semantics, retaining the legacy tuple adapter deliberately."""
    if isinstance(model,Transformer):
        return model(tokens,positions=positions,key_mask=key_mask)[0]
    return model(tokens,positions=positions,key_mask=key_mask)


@contextmanager
def evaluating(model: nn.Module | None) -> Iterator[None]:
    """Use deterministic evaluation behavior while preserving every module's original mode and the caller's autograd context.

    This changes only mode, so policy recomputation can accumulate gradients.
    Sampling separately owns no_grad. Heterogeneous child modes restore on error.
    """
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    try:
        if model is not None:
            model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
