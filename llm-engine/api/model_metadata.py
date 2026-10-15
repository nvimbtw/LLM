"""Resolve model facts from saved results, legacy metadata and bounded ancestry."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from llm import storage
from store import queries as q
from store.schema import Model

ARCHITECTURE_FIELDS = ("dimensions", "heads", "layers", "ff_multiplier", "ctx_window")
MAX_ANCESTRY_HOPS = 25


@dataclass(frozen=True)
class ModelMetadata:
    """Available model facts; missing counts and incomplete shapes stay unavailable."""

    architecture: dict[str, int] | None
    context: int | None
    vocabulary_size: int | None
    parameter_count: int | None


def read_model_meta(directory: Path | None) -> dict:
    """Read a model's JSON metadata without loading weights; absent or invalid JSON yields no facts."""
    if directory is None or not (directory / "meta.json").is_file():
        return {}
    try:
        value = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def positive_count(value: object) -> int | None:
    """Accept positive integral model measurements, excluding booleans and unsafe browser integers."""
    return value if type(value) is int and 0 < value <= 2**53 - 1 else None


def _facts(parameters: dict, meta: dict) -> tuple[dict[str, int], int | None, int | None]:
    """Collect local shape and measured counts, preferring resolved results to requested config."""
    results = [parameters.get("result") or {}, meta.get("result") or {}, meta.get("dataset") or {}]
    configs = [parameters.get("config") or {}, meta.get("config") or {}]
    sources = [result.get("architecture") or {} for result in results] + configs
    architecture: dict[str, int] = {}
    for source in sources:
        for key in ARCHITECTURE_FIELDS:
            if (value := positive_count(source.get(key))) is not None:
                architecture.setdefault(key, value)
    vocabulary = next(
        (value for result in results for candidate in (
            result.get("vocab"), result.get("vocab_size"),
            (result.get("architecture") or {}).get("vocab"),
            (result.get("dataset") or {}).get("vocab_size"),
        ) if (value := positive_count(candidate)) is not None),
        None,
    )
    parameters_count = next(
        (value for result in results if (value := positive_count(result.get("parameters"))) is not None),
        None,
    )
    return architecture, vocabulary, parameters_count


def resolve_model_metadata(db: Session, model: Model | None, meta: dict) -> ModelMetadata:
    """Resolve a model's own facts, filling legacy gaps through base/resume models for at most 25 hops.

    Does not mutate rows or copy ancestor training data. Cycles, malformed IDs and deleted
    ancestors stop the walk; the model's resolved facts always override inherited ones.
    """
    architecture: dict[str, int] = {}
    vocabulary = parameter_count = context = None
    seen: set[uuid.UUID] = set()
    for _ in range(MAX_ANCESTRY_HOPS):
        parameters = (model.parameters or {}) if model else {}
        shape, local_vocabulary, local_count = _facts(parameters, meta)
        for key, value in shape.items():
            architecture.setdefault(key, value)
        vocabulary = vocabulary or local_vocabulary
        parameter_count = parameter_count or local_count
        context = context or shape.get("ctx_window") or positive_count(model.context if model else None)
        if len(architecture) == len(ARCHITECTURE_FIELDS) and vocabulary and parameter_count:
            break
        config = parameters.get("config") or meta.get("config") or {}
        result = parameters.get("result") or meta.get("result") or {}
        ancestor = config.get("resume_from") or config.get("base_model") or result.get("base_model")
        try:
            ancestor_id = uuid.UUID(str(ancestor))
        except ValueError:
            break
        if ancestor_id in seen:
            break
        seen.add(ancestor_id)
        model = q.get_model(db, ancestor_id)
        meta = read_model_meta(storage.model_dir(str(ancestor_id)))
        if model is None and not meta:
            break
    if context is not None:
        architecture.setdefault("ctx_window", context)
    return ModelMetadata(
        architecture=architecture if len(architecture) == len(ARCHITECTURE_FIELDS) else None,
        context=context,
        vocabulary_size=vocabulary,
        parameter_count=parameter_count,
    )
