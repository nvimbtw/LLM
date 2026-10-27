"""RL boundary refusals before saved model construction or external gym acquisition."""

from uuid import uuid4

import pytest

from llm.config import RlConfig
from llm.train.model_preflight import rl_profile
from llm.train.rl_records import RlPreparation, validate_configuration
from tests.test_modular_worker import NumericalContext, preparation


@pytest.mark.parametrize("field,value", [
    ("lr", float("nan")), ("beta", -0.1), ("grad_clip", float("inf")),
    ("temperature", float("inf")), ("top_p", 1.1), ("top_k", -1), ("max_tokens", 65537),
    ("seed_base", -1), ("holdout_base", 2**64), ("tasks_per_batch", 129),
    ("micro_batch", 0), ("iterations", 100000), ("context_multiplier", 2.0),
])
def test_invalid_effective_rl_configuration_refuses_before_acquisition(field, value):
    """Nonfinite math, unbounded batches/history, invalid seeds and unevidenced extension cannot enter an RL numerical lifetime."""
    cfg = RlConfig(model_name=str(uuid4()), base_model=str(uuid4()), task_types=["fixture"],
        tasks_per_batch=1, group_size=2, iterations=2)
    with pytest.raises(ValueError):
        validate_configuration(cfg.model_copy(update={field: value}))


@pytest.mark.parametrize("violation", ["reference", "reference_format", "tokenizer", "precision", "batch", "context"])
def test_rl_preparation_requires_exact_inherited_inputs_and_runtime_profile(tmp_path, violation):
    """Interpreted worker preparation rejects missing original reference, role confusion and stale exact runtime/tokenizer facts."""
    context = NumericalContext(tmp_path / "records")
    source, _command = preparation(context)
    cfg = RlConfig(model_name=str(uuid4()), base_model=str(uuid4()), task_types=["fixture"],
        tasks_per_batch=1, group_size=2, iterations=2)
    snapshot = source.validation.model_copy(update={"request": source.validation.request.model_copy(update={
        "profile": rl_profile(cfg, 8, "cpu")})})
    prep = RlPreparation(request=cfg, validation=snapshot, tokenizer=source.tokenizer)
    raw = prep.model_dump(mode="json")
    if violation in {"reference", "reference_format"}:
        raw["request"]["resume_from"] = str(uuid4())
        if violation == "reference_format":
            raw["reference"] = {**raw["tokenizer"], "format": "checkpoint_v1"}
    elif violation == "tokenizer":
        raw["tokenizer"]["sha256"] = "0" * 64
    elif violation == "precision":
        raw["validation"]["request"]["profile"]["precision"] = "bf16"
    elif violation == "batch":
        raw["validation"]["request"]["profile"]["domains"][0]["batch_max"] = 4
    else:
        raw["validation"]["request"]["profile"]["domains"][0]["sequence_max"] = 9
    with pytest.raises(ValueError):
        RlPreparation.model_validate(raw)
